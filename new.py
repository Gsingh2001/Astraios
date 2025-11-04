import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast # Use new torch.amp
import cv2 

# =======================================================
# --- 1. Configuration and Paths ---
# =======================================================

# --- GLOBAL DEFINITIONS (MUST BE FIRST) ---
CSV_COLUMNS = [
    'source_pcd', 'target_pcd', 
    'source_img', 'target_img',
    'imu_data_path',
    'tx','ty','tz','rx','ry','rz'
]
N_POINTS = 8192 # Must match training (Updated from 2048)
IMG_HEIGHT = 512 # Must match training (Updated from 128)
IMG_WIDTH = 512  # Must match training (Updated from 128)

BASE_DIR = r"G:\backup\papers\Dataset\hidrive_file"
LIDAR_DIR = os.path.join(BASE_DIR, "lidar")
IMAGE_DIR = os.path.join(BASE_DIR, "images", "left") # Path used for images
TRAINING_CSV = os.path.join(BASE_DIR, "training_data.csv")
MODEL_SAVE_PATH = r"G:\backup\papers\Dataset\s3li_umf_odometry.pth"
IMU_SAVE_DIR = os.path.join(BASE_DIR, "imu_data_for_training")

# --- PARAMETERS ---
MAP_DOWNSAMPLE_RATE = 10 
MAP_VOXEL_SIZE = 0.2    
# --- UPDATED BATCH SIZE ---
BATCH_SIZE = 6

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# --- HELPERS ---
def dof6_to_mat(d):
    """Convert 6-DoF pose (tx,ty,tz,rx,ry,rz) to a 4x4 matrix."""
    T = np.eye(4, dtype=np.float32)
    t = d[:3]
    r = Rotation.from_euler('xyz', d[3:], degrees=False)
    T[:3, :3] = r.as_matrix().astype(np.float32)
    T[:3, 3] = t
    return T

def load_pcd_for_mapping(path):
    """Loads a full PCD file for global map generation."""
    try:
        pcd = o3d.io.read_point_cloud(path)
        # Downsample the single frame before adding to global map
        return pcd.voxel_down_sample(voxel_size=0.1)
    except Exception:
        return o3d.geometry.PointCloud()

# ==============================================================
# --- 2. Dataset and Model Architecture ---
# (Copied from working training script for correct inference)
# ==============================================================

def load_pcd_numpy(path, num_points):
    """Load PCD and sample/pad for network input."""
    try:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.shape[0] > num_points:
            idx = np.random.choice(pts.shape[0], num_points, replace=False)
            pts = pts[idx]
        elif pts.shape[0] < num_points:
            idx = np.random.choice(pts.shape[0], num_points - pts.shape[0], replace=True)
            pts = np.vstack((pts, pts[idx])) if pts.shape[0]>0 else np.zeros((num_points,3),dtype=np.float32)
        return pts
    except:
        return np.zeros((num_points,3),dtype=np.float32)

def load_image(path):
    """Loads and samples an image file."""
    img = cv2.imread(path)
    if img is None:
        return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT)).astype(np.float32) / 255.0
    return img

class LIV_Inference_Dataset(Dataset):
    def __init__(self, csv_file):
        # --- FIX: CSV_COLUMNS is now defined globally and accessible ---
        self.data = pd.read_csv(csv_file, names=CSV_COLUMNS, skiprows=1)
        self.imu_dir = os.path.dirname(IMU_SAVE_DIR)
        print(f"Loaded {len(self.data)} inference samples.")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. LiDAR Input (for network)
        src_pc = load_pcd_numpy(os.path.join(LIDAR_DIR, row['source_pcd']), N_POINTS)
        tgt_pc = load_pcd_numpy(os.path.join(LIDAR_DIR, row['target_pcd']), N_POINTS)
        combined_pc = np.vstack((src_pc, tgt_pc)).astype(np.float32)
        combined_pc -= np.mean(combined_pc, axis=0, keepdims=True)
        pc_tensor = torch.tensor(combined_pc, dtype=torch.float32).permute(1, 0) # (3, 2*N)

        # 2. IMU Input
        imu = np.load(os.path.join(self.imu_dir, row['imu_data_path'])) if os.path.exists(os.path.join(self.imu_dir, row['imu_data_path'])) else np.zeros((1,6), dtype=np.float32)
        imu_tensor = torch.tensor(imu, dtype=torch.float32)

        # 3. Vision Input
        src_img = load_image(os.path.join(IMAGE_DIR, row['source_img']))
        tgt_img = load_image(os.path.join(IMAGE_DIR, row['target_img']))
        combined_img = np.concatenate((src_img, tgt_img), axis=2)
        img_tensor = torch.tensor(combined_img, dtype=torch.float32).permute(2, 0, 1) # (6, H, W)
        
        # 4. Ground Truth and Path Data (for mapping)
        pose_6d = torch.tensor(row[['tx', 'ty', 'tz', 'rx', 'ry', 'rz']].values.astype(np.float32))
        source_pcd_path = os.path.join(LIDAR_DIR, row['source_pcd'])
        
        return pc_tensor, imu_tensor, img_tensor, pose_6d, source_pcd_path

def liv_collate_fn(batch):
    """Custom collate for padding IMU sequences and extracting mapping data."""
    pc_tensors = [item[0] for item in batch]
    imu_tensors = [item[1] for item in batch]
    img_tensors = [item[2] for item in batch]
    poses = [item[3] for item in batch]
    pcd_paths = [item[4] for item in batch]

    pc_batch = torch.stack(pc_tensors, dim=0)
    img_batch = torch.stack(img_tensors, dim=0)
    pose_batch = torch.stack(poses, dim=0)
    imu_padded = pad_sequence(imu_tensors, batch_first=True, padding_value=0.0)
    
    return pc_batch, imu_padded, img_batch, pose_batch, pcd_paths

class LIV_OdometryNet(nn.Module):
    """Multimodal Odometry Network with Transformer Fusion (UMF-inspired)"""
    def __init__(self, num_points=N_POINTS, img_h=IMG_HEIGHT, img_w=IMG_WIDTH, d_model=512, n_head=8):
        super().__init__()
        d_model = 512
        self.lidar_fe = nn.Sequential(nn.Conv1d(3,64,1), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64,128,1), nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128,1024,1), nn.BatchNorm1d(1024), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
        self.imu_rnn = nn.GRU(6, 128, 2, batch_first=True, dropout=0.1)
        self.visual_fe = nn.Sequential(nn.Conv2d(6,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU(), nn.Conv2d(64,128,3,2,1), nn.ReLU(), nn.Conv2d(128,256,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.lidar_proj = nn.Linear(1024, d_model)
        self.imu_proj = nn.Linear(128, d_model)
        self.visual_proj = nn.Linear(256, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model * 4, dropout=0.1, activation='gelu', batch_first=True)
        self.fusion_transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Sequential(nn.Linear(d_model * 3, 1024), nn.ReLU(), nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 6))

    def forward(self, pc, imu, img):
        B = pc.shape[0]
        lidar_feat = self.lidar_fe(pc).view(B, -1)
        _, h_n = self.imu_rnn(imu)
        imu_feat = h_n[-1] 
        vis_feat = self.visual_fe(img).view(B, -1)
        
        tok_l = self.lidar_proj(lidar_feat)
        tok_i = self.imu_proj(imu_feat)
        tok_v = self.visual_proj(vis_feat)
        
        tokens = torch.stack([tok_l, tok_i, tok_v], dim=1)
        fused = self.fusion(tokens).reshape(B, -1)
        
        return self.head(fused)


# =======================================================
# --- 5. Inference, Mapping, and Visualization ---
# =======================================================

def build_and_visualize_map():
    """
    Reads poses, chains them to build trajectories, and accumulates the 3D map.
    """
    if not os.path.exists(TRAINING_CSV):
        print(f"Error: CSV not found at: {TRAINING_CSV}")
        return
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model not found at {MODEL_SAVE_PATH}")
        return

    # --- Dataset Setup ---
    ds = LIV_Inference_Dataset(TRAINING_CSV)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, collate_fn=liv_collate_fn)

    # --- Model Setup ---
    model = LIV_OdometryNet().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Accumulators
    predicted_poses = [np.eye(4, dtype=np.float32)] # 4x4 matrices
    ground_truth_poses = [np.eye(4, dtype=np.float32)]
    global_map_pcd = o3d.geometry.PointCloud()
    
    current_T_pred = np.eye(4, dtype=np.float32)
    current_T_gt = np.eye(4, dtype=np.float32)

    print("\nRunning inference and accumulating 3D map...")
    with torch.no_grad():
        for pc, imu, img, gt_pose_6d, pcd_paths in tqdm(loader, desc="[Inference]"):
            pc, imu, img = pc.to(device), imu.to(device), img.to(device)

            # 1. Predict the relative pose (6 DoF)
            with autocast(device_type=device.type, dtype=torch.float16):
                predicted_pose_6d_batch = model(pc, imu, img).cpu().numpy()
            
            gt_pose_6d_batch = gt_pose_6d.numpy()

            # 2. Chain Poses and Build Map
            for i, (p_6d, gt_6d) in enumerate(zip(predicted_pose_6d_batch, gt_pose_6d_batch)):
                
                T_delta_gt = dof6_to_mat(gt_6d)
                T_delta_pred = dof6_to_mat(p_6d) # Use model's prediction
                
                # Update Poses
                current_T_gt = current_T_gt @ T_delta_gt
                current_T_pred = current_T_pred @ T_delta_pred
                
                ground_truth_poses.append(current_T_gt.copy())
                predicted_poses.append(current_T_pred.copy())

                # Accumulate Map (Sparsely, using GT pose for map quality)
                if len(predicted_poses) % 50 == 0: # Accumulate every 50th frame
                    pcd_path = pcd_paths[i]
                    pcd = load_pcd_for_mapping(pcd_path)
                    
                    # Transform the current PCD into the global frame using the accurate GT pose
                    pcd.transform(current_T_gt)
                    global_map_pcd += pcd


    print(f"\nMap accumulation finished. Total points before downsample: {len(global_map_pcd.points)}")
    
    # 3. Final Map Downsampling
    global_map_pcd = global_map_pcd.voxel_down_sample(voxel_size=0.2)
    print(f"Final map points: {len(global_map_pcd.points)}")
    
    # 4. Visualization
    visualize_3d_trajectory_o3d(predicted_poses, ground_truth_poses, global_map_pcd)


def visualize_3d_trajectory_o3d(predicted_poses, ground_truth_poses, global_map):
    """
    Displays the 3D map, trajectories, and coordinate frames using Open3D.
    """
    
    geometries = [global_map]
    
    # --- Helper to create trajectories as LineSets ---
    def create_line_set(poses, color, line_name):
        points = np.array([p[:3, 3] for p in poses])
        lines = [[i, i+1] for i in range(len(points) - 1)]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector([color for i in range(len(lines))])
        return line_set

    # 1. Trajectory 1: Ground Truth (Green/Blue)
    gt_line = create_line_set(ground_truth_poses, [0, 0.7, 0], 'Ground Truth') 
    geometries.append(gt_line)

    # 2. Trajectory 2: Predicted (Red/Dashed style)
    pred_line = create_line_set(predicted_poses, [1, 0, 0], 'Predicted')
    geometries.append(pred_line)

    # 3. Coordinate Frames (Aesthetics)
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0))

    # Add coordinate frames sparsely (every 50th frame)
    for i in range(0, len(ground_truth_poses), 50):
        # GT Frames (Green/Blue, showing sensor orientation)
        coord_gt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        coord_gt.transform(ground_truth_poses[i])
        geometries.append(coord_gt)
        
    print(f"\n--- Launching Open3D Visualization ({len(geometries)} geometries) ---")
    print("GT Trajectory: Green | Predicted Trajectory: Red")

    o3d.visualization.draw_geometries(geometries, 
                                      window_name="DLR S3LI Multimodal Odometry: Trajectory and Map",
                                      zoom=0.1,
                                      front=[1, 0, 0],
                                      lookat=[0, 0, 0],
                                      up=[0, 0, 1])

if __name__ == "__main__":
    build_and_visualize_map()
