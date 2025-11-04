import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.transform import Rotation
import open3d as o3d

# ===============================
# --- 1. Parameters ---
# ===============================

base_dir = r"G:\backup\papers\Dataset\hidrive_file"  # Path to extracted bag folder
lidar_dir = os.path.join(base_dir, "lidar")
associations_path = os.path.join(base_dir, "associations.txt")
training_csv_path = os.path.join(base_dir, "training_data.csv")

# ===============================
# --- 2. Utility Functions ---
# ===============================

def load_associations(associations_path):
    """Loads synced frames from associations.txt."""
    associations = []
    with open(associations_path, 'r') as f:
        for line in f:
            if line.startswith('#'): 
                continue
            parts = line.split()
            if len(parts) == 6:
                img_left_name = os.path.basename(parts[1])
                img_right_name = os.path.basename(parts[3])
                pcd_name = os.path.basename(parts[5])
                associations.append((img_left_name, img_right_name, pcd_name))
    print(f"Loaded {len(associations)} associations.")
    return associations

def matrix_to_6dof(matrix):
    """Converts 4x4 transformation matrix to 6 DoF pose (tx, ty, tz, rx, ry, rz)."""
    translation = matrix[0:3, 3]
    rotation_matrix = matrix[0:3, 0:3]
    r = Rotation.from_matrix(rotation_matrix)
    rotation_euler = r.as_euler('xyz', degrees=False)
    return np.concatenate((translation, rotation_euler))

def estimate_lidar_odometry(pcd_path_source, pcd_path_target):
    """
    Runs ICP on two point clouds and returns 4x4 transform.
    Returns None if ICP fails.
    """
    try:
        source = o3d.io.read_point_cloud(pcd_path_source)
        target = o3d.io.read_point_cloud(pcd_path_target)
        if not source.has_points() or not target.has_points():
            return None

        # Downsample and estimate normals
        voxel_size = 0.1
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        source_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        target_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))

        # ICP Registration (Point-to-Plane)
        threshold = 0.5
        reg = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
        )
        if reg.fitness < 0.1:
            return None

        return reg.transformation

    except Exception as e:
        print(f"ICP Error ({pcd_path_source} -> {pcd_path_target}): {e}")
        return None

# ===============================
# --- 3. Create Training Dataset ---
# ===============================

def create_training_dataset():
    associations = load_associations(associations_path)
    if not associations or len(associations) < 2:
        print("Not enough frames to create training data.")
        return

    training_data = []

    print("\nGenerating training data using ICP...")
    for i in tqdm(range(len(associations) - 1)):
        _, _, pcd_i = associations[i]
        _, _, pcd_i1 = associations[i+1]

        pcd_path_i = os.path.join(lidar_dir, pcd_i)
        pcd_path_i1 = os.path.join(lidar_dir, pcd_i1)

        if not os.path.exists(pcd_path_i) or not os.path.exists(pcd_path_i1):
            continue

        T = estimate_lidar_odometry(pcd_path_i, pcd_path_i1)
        if T is not None:
            pose_6d = matrix_to_6dof(T)
            training_data.append([pcd_i, pcd_i1] + pose_6d.tolist())

    if not training_data:
        print("No training data generated. ICP failed on all frames.")
        return

    df = pd.DataFrame(training_data, columns=['source_pcd','target_pcd','tx','ty','tz','rx','ry','rz'])
    df.to_csv(training_csv_path, index=False)
    print(f"Training dataset saved to: {training_csv_path} ({len(df)} samples)")

# ===============================
# --- 4. Main ---
# ===============================

if __name__ == "__main__":
    create_training_dataset()
