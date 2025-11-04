import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import open3d as o3d
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast
import matplotlib.pyplot as plt
import cv2 

# =======================================================
# --- 1. Settings & Architecture (Must Match Training)---
# =======================================================

# --- Local Paths ---
# NOTE: These paths must match the paths used during data generation.
BASE_DIR = r'G:\backup\papers\Dataset\hidrive_file'
LIDAR_DIR = os.path.join(BASE_DIR, "lidar")
IMAGE_DIR = os.path.join(BASE_DIR, "images", "left")
IMU_SAVE_DIR = os.path.join(BASE_DIR, "imu_data_for_training")
TRAINING_CSV_PATH = os.path.join(BASE_DIR, "training_data.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "s3li_umf_odometry.pth")

# --- Model & Data Parameters ---
N_POINTS = 2048
BATCH_SIZE = 16 
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_WORKERS = 4 # Use 0 if multiprocessing errors occur locally

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =======================================================
# --- 2. Utility Functions ---
# =======================================================

def dof6_to_matrix(pose_6d):
    """Converts 6 DoF pose (tx, ty, tz, rx, ry, rz) to a 4x4 matrix."""
    T = np.eye(4, dtype=np.float32)
    t = pose_6d[0:3]
    r_euler = pose_6d[3:6]
    r = Rotation.from_euler('xyz', r_euler, degrees=False)
    T[:3, :3] = r.as_matrix().astype(np.float32)
    T[:3, 3] = t
    return T

def load_pcd(path, num_points):
    """Loads and samples a PCD file."""
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

# =======================================================
# --- 3. Dataset (LIV - Must Match Training) ---
# =======================================================

class LIV_Dataset(Dataset):
    """LiDAR-Inertial-Visual Odometry Dataset (Inference Mode)"""
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.imu_dir = IMU_SAVE_DIR
        print(f"Loaded {len(self.data)} inference samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. --- LiDAR ---
        src_pc = load_pcd(os.path.join(LIDAR_DIR, row['source_pcd']), N_POINTS)
        tgt_pc = load_pcd(os.path.join(LIDAR_DIR, row['target_pcd']), N_POINTS)
        combined_pc = np.vstack((src_pc, tgt_pc)).astype(np.float32)
        combined_pc -= np.mean(combined_pc, axis=0, keepdims=True)
        pc_tensor = torch.tensor(combined_pc, dtype=torch.float32).permute(1, 0)

        # 2. --- IMU ---
        imu = np.load(os.path.join(IMU_SAVE_DIR, row['imu_data_path'])) if os.path.exists(os.path.join(IMU_SAVE_DIR, row['imu_data_path'])) else np.zeros((1,6), dtype=np.float32)
        imu_tensor = torch.tensor(imu, dtype=torch.float32)

        # 3. --- Vision ---
        src_img = load_image(os.path.join(IMAGE_DIR, row['source_img']))
        tgt_img = load_image(os.path.join(IMAGE_DIR, row['target_img']))
        combined_img = np.concatenate((src_img, tgt_img), axis=2)
        img_tensor = torch.tensor(combined_img, dtype=torch.float32).permute(2, 0, 1)

        # 4. --- Target Pose (Ground Truth) ---
        pose_6d = torch.tensor(row[['tx', 'ty', 'tz', 'rx', 'ry', 'rz']].values.astype(np.float32))
        
        return pc_tensor, imu_tensor, img_tensor, pose_6d

def liv_collate_fn(batch):
    """Custom collate for padded IMU data and stacked PC/Image data."""
    pc_tensors = [item[0] for item in batch]
    imu_tensors = [item[1] for item in batch]
    img_tensors = [item[2] for item in batch]
    poses = [item[3] for item in batch]
    
    pc_batch = torch.stack(pc_tensors, dim=0)
    img_batch = torch.stack(img_tensors, dim=0)
    pose_batch = torch.stack(poses, dim=0)
    
    # Pad IMU sequences
    imu_padded = pad_sequence(imu_tensors, batch_first=True, padding_value=0.0)
    
    return pc_batch, imu_padded, img_batch, pose_batch

# =======================================================
# --- 4. Model Architecture (Must Match Training) ---
# =======================================================

class LIV_OdometryNet(nn.Module):
    """
    Multimodal Odometry Network with Transformer Fusion (UMF-inspired)
    """
    def __init__(self, num_points=N_POINTS, img_h=IMG_HEIGHT, img_w=IMG_WIDTH, d_model=512, n_head=8):
        super().__init__()
        
        # 1. Feature Extractors
        self.lidar_fe = nn.Sequential(nn.Conv1d(3,64,1), nn.BatchNorm1d(64), nn.ReLU(), nn.Conv1d(64,128,1), nn.BatchNorm1d(128), nn.ReLU(), nn.Conv1d(128,1024,1), nn.BatchNorm1d(1024), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
        self.imu_rnn = nn.GRU(6, 128, 2, batch_first=True, dropout=0.1)
        self.visual_fe = nn.Sequential(nn.Conv2d(6,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU(), nn.Conv2d(64,128,3,2,1), nn.ReLU(), nn.Conv2d(128,256,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        
        # 2. Projection Layers
        self.lidar_proj = nn.Linear(1024, d_model)
        self.imu_proj = nn.Linear(128, d_model)
        self.visual_proj = nn.Linear(256, d_model)
        
        # 3. Transformer Fusion
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4, batch_first=True, dropout=0.1, activation='gelu')
        self.fusion = nn.TransformerEncoder(enc_layer, num_layers=2)
        
        # 4. Regression Head
        self.head = nn.Sequential(nn.Linear(d_model*3,1024),nn.ReLU(), nn.Linear(1024,256),nn.ReLU(), nn.Linear(256,6))

    def forward(self, pc, imu, img):
        B = pc.shape[0]
        
        # Feature Extraction
        lidar_feat = self.lidar_fe(pc).view(B, -1)
        _, h_n = self.imu_rnn(imu)
        imu_feat = h_n[-1] # Use the final hidden state of the GRU
        vis_feat = self.visual_fe(img).view(B, -1)
        
        # Projection
        tok_l = self.lidar_proj(lidar_feat)
        tok_i = self.imu_proj(imu_feat)
        tok_v = self.visual_proj(vis_feat)
        
        # Fusion
        tokens = torch.stack([tok_l, tok_i, tok_v], dim=1) # (B, 3 tokens, d_model)
        fused = self.fusion(tokens).reshape(B, -1) # Flatten (B, 3*d_model)
        
        # Pose Regression
        return self.head(fused)

# =======================================================
# --- 5. Inference and Trajectory Chaining ---
# =======================================================

def run_inference():
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Model not found at {MODEL_SAVE_PATH}")
        print("Please ensure the training script ran fully.")
        return

    # Load Model
    model = LIV_OdometryNet().to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Load Dataset
    ds = LIV_Dataset(TRAINING_CSV_PATH)
    # Ensure shuffle=False for correct sequence chaining
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=liv_collate_fn)

    # Initialize Trajectories (starting at [0,0,0])
    predicted_traj = [np.zeros(3, dtype=np.float32)]
    ground_truth_traj = [np.zeros(3, dtype=np.float32)]
    
    # Pose matrices (4x4)
    current_predicted_pose = np.eye(4)
    current_ground_truth_pose = np.eye(4)

    print("\nRunning inference and chaining poses...")
    with torch.no_grad():
        for pc, imu, img, gt_pose_6d in tqdm(loader, desc="[Inference]"):
            pc, imu, img = pc.to(device), imu.to(device), img.to(device)

            # 1. Predict the relative pose (6 DoF)
            # Use autocast for potential speed boost even in inference
            with autocast(device_type=device.type, dtype=torch.float16):
                predicted_pose_6d_batch = model(pc, imu, img).cpu().numpy()
            
            gt_pose_6d_batch = gt_pose_6d.numpy()

            # 2. Chain Poses (iterate over batch)
            for p_6d, gt_6d in zip(predicted_pose_6d_batch, gt_pose_6d_batch):
                
                # Convert 6 DoF delta pose to 4x4 matrix
                T_pred_delta = dof6_to_matrix(p_6d)
                T_gt_delta = dof6_to_matrix(gt_6d)
                
                # Accumulate the global pose: T_world = T_world @ T_delta
                current_predicted_pose = current_predicted_pose @ T_pred_delta
                current_ground_truth_pose = current_ground_truth_pose @ T_gt_delta

                # Extract translation for plotting
                predicted_traj.append(current_predicted_pose[:3, 3].tolist())
                ground_truth_traj.append(current_ground_truth_pose[:3, 3].tolist())

    print("Inference complete. Plotting trajectories.")
    plot_trajectories(np.array(predicted_traj), np.array(ground_truth_traj))


def plot_trajectories(predicted_traj, ground_truth_traj):
    """Plots the 2D trajectory (X vs Y)."""
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Ground Truth (Blue)
    ax.plot(ground_truth_traj[:, 0], ground_truth_traj[:, 1], 
            label='Ground Truth (ICP)', color='blue', linewidth=2)
    ax.scatter(ground_truth_traj[::10, 0], ground_truth_traj[::10, 1], 
               color='blue', marker='.', s=50, alpha=0.6)

    # Plot Predicted Trajectory (Red)
    ax.plot(predicted_traj[:, 0], predicted_traj[:, 1], 
            label='Predicted (LIV Model)', color='red', linestyle='--', linewidth=1.5)
    ax.scatter(predicted_traj[::10, 0], predicted_traj[::10, 1], 
               color='red', marker='x', s=50, alpha=0.8)

    # Styling
    ax.set_title('Predicted Trajectory vs. Ground Truth (XY Plane)')
    ax.set_xlabel('X Position (meters)')
    ax.set_ylabel('Y Position (meters)')
    ax.legend()
    ax.axis('equal') # Important for visualizing accurate motion scale
    ax.grid(True)
    plt.show()


if __name__ == "__main__":
    run_inference()
