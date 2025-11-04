import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import open3d as o3d
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt
import cv2

# ===============================
# --- 1. Parameters ---
# ===============================

base_dir = r"G:\backup\papers\Dataset\hidrive_file"
lidar_dir = os.path.join(base_dir, "lidar")
image_dir = os.path.join(base_dir, "images", "left")
associations_path = os.path.join(base_dir, "associations.txt")
training_csv_path = os.path.join(base_dir, "training_data.csv")
imu_data_path = os.path.join(base_dir, "imu", "imu_data.csv")
imu_save_dir = os.path.join(base_dir, "imu_data_for_training")
model_save_path = os.path.join(base_dir, "liv_odometry_model.pth")

# Training Params
N_POINTS = 2048
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-4
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_WORKERS = 4

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ===============================
# --- 2. Data Functions ---
# ===============================

def load_associations(path):
    associations = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or len(line.strip())==0:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                lidar_ts = float(parts[4])
                img_left_name = os.path.basename(parts[1])
                pcd_name = os.path.basename(parts[5])
                associations.append((pcd_name, img_left_name, lidar_ts))
            except:
                continue
    print(f"Loaded {len(associations)} associations")
    return associations


def parse_imu_column(col):
    def parse_entry(x):
        try:
            # Remove parentheses if exist, split by comma, take first value (or all if needed)
            if isinstance(x, str) and x.startswith("("):
                vals = x.strip("()").split(",")
                return [float(v) for v in vals]  # returns all numbers
            return float(x)
        except:
            return 0.0
    return col.apply(parse_entry)

def load_imu_csv(path):
    df = pd.read_csv(path, header=None, on_bad_lines='skip')

    numeric_cols = [4,5,6,7,11,12,13]  # adjust as needed
    processed = []
    for c in numeric_cols:
        col_data = parse_imu_column(df[c])
        # Flatten if column entries are lists (from tuple strings)
        if isinstance(col_data.iloc[0], list):
            col_data = col_data.apply(lambda x: x[0])  # take first element or handle differently
        processed.append(col_data.astype(np.float32))

    imu_data = np.stack(processed, axis=1)
    timestamps = df[0].to_numpy()
    print(f"Loaded IMU: {len(imu_data)} readings")
    return timestamps, imu_data



def matrix_to_6dof(matrix):
    t = matrix[0:3,3]
    r = Rotation.from_matrix(matrix[0:3,0:3]).as_euler('xyz', degrees=False)
    return np.concatenate((t,r))

def estimate_lidar_odometry(pcd_path_source, pcd_path_target):
    try:
        src = o3d.io.read_point_cloud(pcd_path_source)
        tgt = o3d.io.read_point_cloud(pcd_path_target)
        if not src.has_points() or not tgt.has_points():
            return None
        voxel_size = 0.1
        src_down = src.voxel_down_sample(voxel_size)
        tgt_down = tgt.voxel_down_sample(voxel_size)
        src_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=30))
        tgt_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,max_nn=30))
        reg = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down, 0.5, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        if reg.fitness<0.1:
            return None
        return reg.transformation
    except Exception as e:
        print(f"ICP error: {e}")
        return None

def find_imu_indices(timestamps, start, end):
    start_idx = np.searchsorted(timestamps, start, side='left')
    end_idx = np.searchsorted(timestamps, end, side='right')
    return start_idx, end_idx

def create_training_dataset(debug_max=100):
    associations = load_associations(associations_path)
    imu_ts, imu_data = load_imu_csv(imu_data_path)
    if not associations or imu_data is None:
        print("No data found. Check files.")
        return None
    os.makedirs(imu_save_dir, exist_ok=True)
    data_rows = []

    success_count = 0
    fail_count = 0
    skip_count = 0

    print(f"Generating training data for max {debug_max} frames (debug mode)...")
    for i in tqdm(range(min(len(associations)-1, debug_max)), desc="Frames"):
        pcd_i, img_i, ts_i = associations[i]
        pcd_i1, img_i1, ts_i1 = associations[i+1]

        paths_exist = all(os.path.exists(os.path.join(p, f)) for p,f in 
                          zip([lidar_dir,lidar_dir,image_dir,image_dir],
                              [pcd_i,pcd_i1,img_i,img_i1]))
        if not paths_exist:
            skip_count += 1
            tqdm.write(f"Frame {i}: ⏭ Skipped missing files")
            continue

        T = estimate_lidar_odometry(os.path.join(lidar_dir, pcd_i),
                                    os.path.join(lidar_dir, pcd_i1))
        if T is None:
            fail_count += 1
            tqdm.write(f"Frame {i}: ❌ ICP failed")
            continue
        pose_6d = matrix_to_6dof(T)

        start_idx, end_idx = find_imu_indices(imu_ts, ts_i, ts_i1)
        if start_idx >= end_idx:
            fail_count += 1
            tqdm.write(f"Frame {i}: ❌ No IMU data")
            continue
        imu_seq = imu_data[start_idx:end_idx]
        imu_filename = f"{ts_i}_to_{ts_i1}.npy"
        imu_filepath = os.path.join(imu_save_dir, imu_filename)
        np.save(imu_filepath, imu_seq.astype(np.float32))
        imu_relative = os.path.join(os.path.basename(imu_save_dir), imu_filename)

        data_rows.append([pcd_i, pcd_i1, img_i, img_i1, imu_relative, *pose_6d.tolist()])
        success_count += 1

        tqdm.write(f"Frame {i}: ✅ Success | Total ✅ {success_count}, ❌ {fail_count}, ⏭ {skip_count}")

    df = pd.DataFrame(data_rows, columns=['source_pcd','target_pcd','source_img','target_img','imu_data_path',
                                          'tx','ty','tz','rx','ry','rz'])
    df.to_csv(training_csv_path, index=False)
    print(f"\nDebug training dataset created: {len(df)} samples")
    return training_csv_path


def main_train(debug_max=100):
    csv_path = create_training_dataset(debug_max)
    if csv_path is None:
        print("Training dataset generation failed. Abort.")
        return
    ds = LIV_Dataset(csv_path)
    if len(ds) == 0:
        print("Dataset empty. Abort.")
        return
    train_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, collate_fn=liv_collate_fn)

    model = LIV_OdometryNet(N_POINTS, IMG_HEIGHT, IMG_WIDTH).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = GradScaler()

    print("Starting debug training for 1 epoch...")
    model.train()
    for pc, imu, img, pose in tqdm(train_loader, desc="[Debug Train]"):
        pc, imu, img, pose = pc.to(device), imu.to(device), img.to(device), pose.to(device)
        optimizer.zero_grad()
        with autocast(device_type=device.type, dtype=torch.float16):
            preds = model(pc, imu, img)
            loss = criterion(preds, pose)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        tqdm.write(f"Batch loss: {loss.item():.6f}")

    print("Debug training completed.")


# ===============================
# --- 3. Dataset ---
# ===============================

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        return np.zeros((IMG_HEIGHT,IMG_WIDTH,3),dtype=np.float32)
    img = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT)).astype(np.float32)/255.0
    return img

def load_pcd(path, num_points):
    try:
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points,dtype=np.float32)
        if pts.shape[0] > num_points:
            idx = np.random.choice(pts.shape[0], num_points, replace=False)
            pts = pts[idx]
        elif pts.shape[0]<num_points:
            idx = np.random.choice(pts.shape[0], num_points-pts.shape[0], replace=True)
            pts = np.vstack((pts, pts[idx])) if pts.shape[0]>0 else np.zeros((num_points,3),dtype=np.float32)
        return pts
    except:
        return np.zeros((num_points,3),dtype=np.float32)

class LIV_Dataset(Dataset):
    def __init__(self,csv_file):
        self.data = pd.read_csv(csv_file)
        print(f"LIV dataset loaded: {len(self.data)} samples")
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        row = self.data.iloc[idx]
        src_pc = load_pcd(os.path.join(lidar_dir,row['source_pcd']),N_POINTS)
        tgt_pc = load_pcd(os.path.join(lidar_dir,row['target_pcd']),N_POINTS)
        pc = np.vstack((src_pc,tgt_pc))
        pc -= np.mean(pc,axis=0,keepdims=True)
        pc_tensor = torch.tensor(pc,dtype=torch.float32).permute(1,0)

        imu = np.load(os.path.join(imu_save_dir,row['imu_data_path'])) if os.path.exists(os.path.join(imu_save_dir,row['imu_data_path'])) else np.zeros((1,6),dtype=np.float32)
        imu_tensor = torch.tensor(imu,dtype=torch.float32)

        src_img = load_image(os.path.join(image_dir,row['source_img']))
        tgt_img = load_image(os.path.join(image_dir,row['target_img']))
        img = torch.tensor(np.concatenate((src_img,tgt_img),axis=2),dtype=torch.float32).permute(2,0,1)

        pose = torch.tensor(row[['tx','ty','tz','rx','ry','rz']].values.astype(np.float32))
        return pc_tensor, imu_tensor, img, pose

def liv_collate_fn(batch):
    pc_tensors = [b[0] for b in batch]
    imu_tensors = [b[1] for b in batch]
    img_tensors = [b[2] for b in batch]
    pose_tensors = [b[3] for b in batch]
    pc_batch = torch.stack(pc_tensors)
    img_batch = torch.stack(img_tensors)
    pose_batch = torch.stack(pose_tensors)
    imu_batch = pad_sequence(imu_tensors,batch_first=True,padding_value=0.0)
    return pc_batch, imu_batch, img_batch, pose_batch

# ===============================
# --- 4. Model ---
# ===============================

class LIV_OdometryNet(nn.Module):
    def __init__(self,num_points, img_h, img_w, d_model=512, n_head=8):
        super().__init__()
        self.lidar_fe = nn.Sequential(
            nn.Conv1d(3,64,1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64,128,1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128,1024,1), nn.BatchNorm1d(1024), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1)
        )
        self.imu_rnn = nn.GRU(6,128,2,batch_first=True,dropout=0.1)
        self.visual_fe = nn.Sequential(
            nn.Conv2d(6,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(),
            nn.Conv2d(128,256,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.lidar_proj = nn.Linear(1024,d_model)
        self.imu_proj = nn.Linear(128,d_model)
        self.visual_proj = nn.Linear(256,d_model)
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model,nhead=n_head,dim_feedforward=d_model*4,batch_first=True),
            num_layers=2
        )
        self.head = nn.Sequential(nn.Linear(d_model*3,1024),nn.ReLU(),
                                  nn.Linear(1024,256),nn.ReLU(),
                                  nn.Linear(256,6))
    def forward(self,x_pc,x_imu,x_img):
        B = x_pc.shape[0]
        lidar_feat = self.lidar_fe(x_pc).view(B,-1)
        _,h_n = self.imu_rnn(x_imu)
        imu_feat = h_n[-1]
        visual_feat = self.visual_fe(x_img).view(B,-1)
        lidar_proj = self.lidar_proj(lidar_feat)
        imu_proj = self.imu_proj(imu_feat)
        visual_proj = self.visual_proj(visual_feat)
        tokens = torch.stack([lidar_proj,imu_proj,visual_proj],dim=1)
        fused = self.fusion_transformer(tokens).reshape(B,-1)
        return self.head(fused)

# ===============================
# --- 5. Training ---
# ===============================

def main_train():
    csv_path = create_training_dataset()
    if csv_path is None:
        print("Training dataset generation failed. Abort.")
        return
    ds = LIV_Dataset(csv_path)
    if len(ds)==0:
        print("Dataset empty. Abort.")
        return
    train_size = int(0.8*len(ds))
    val_size = len(ds)-train_size
    train_ds,val_ds = torch.utils.data.random_split(ds,[train_size,val_size])
    train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,pin_memory=True,collate_fn=liv_collate_fn)
    val_loader = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,pin_memory=True,collate_fn=liv_collate_fn)

    model = LIV_OdometryNet(N_POINTS,IMG_HEIGHT,IMG_WIDTH).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(),lr=LEARNING_RATE)
    scaler = GradScaler()
    best_val = float('inf')
    history={'train_loss':[],'val_loss':[]}

    for epoch in range(EPOCHS):
        model.train()
        train_loss=0
        for pc,imu,img,pose in tqdm(train_loader,desc=f"[Train] Epoch {epoch+1}/{EPOCHS}"):
            pc,imu,img,pose = pc.to(device),imu.to(device),img.to(device),pose.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type,dtype=torch.float16):
                preds = model(pc,imu,img)
                loss = criterion(preds,pose)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        avg_train = train_loss/len(train_loader)
        history['train_loss'].append(avg_train)

        model.eval()
        val_loss=0
        with torch.no_grad():
            for pc,imu,img,pose in val_loader:
                pc,imu,img,pose = pc.to(device),imu.to(device),img.to(device),pose.to(device)
                with autocast(device_type=device.type,dtype=torch.float16):
                    preds = model(pc,imu,img)
                    loss = criterion(preds,pose)
                val_loss += loss.item()
        avg_val = val_loss/len(val_loader)
        history['val_loss'].append(avg_val)
        print(f"Epoch {epoch+1}/{EPOCHS} -> Train {avg_train:.6f} | Val {avg_val:.6f}")

        if avg_val<best_val:
            best_val=avg_val
            torch.save(model.state_dict(),model_save_path)
            print(f"Saved best model at epoch {epoch+1}")

    plt.figure()
    plt.plot(history['train_loss'],label='Train')
    plt.plot(history['val_loss'],label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

if __name__=="__main__":
    main_train()
