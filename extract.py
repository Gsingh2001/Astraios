import os
import csv
import yaml
from pathlib import Path
from tqdm import tqdm

# ROS imports
try:
    import rosbag
    import sensor_msgs.msg
    import geometry_msgs.msg
    import std_msgs.msg
except ImportError:
    raise ImportError("You need ROS Python packages (rosbag, sensor_msgs, geometry_msgs) installed.")

# -----------------------------
# 1. Paths
# -----------------------------
bag_file = r"G:\backup\papers\Dataset\hidrive_file.bag"  # Use raw string for Windows paths
out_root = r"G:\backup\papers\Dataset\hidrive_file"

# Create output folders
dirs = {
    'left': os.path.join(out_root, "images/left"),
    'right': os.path.join(out_root, "images/right"),
    'lidar': os.path.join(out_root, "lidar"),
    'camera_info': os.path.join(out_root, "camera_info"),
    'imu': os.path.join(out_root, "imu"),
    'tf': os.path.join(out_root, "tf")
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

# -----------------------------
# 2. Open bag and list topics
# -----------------------------
print(f"[INFO] Opening bag file: {bag_file}")
bag = rosbag.Bag(bag_file)

print("Topics in bag:")
for topic, topic_info in bag.get_type_and_topic_info()[1].items():
    msg_type = topic_info.msg_type
    msg_count = topic_info.message_count
    print(f"{topic} | {msg_type} | {msg_count} messages")

# -----------------------------
# 3. IMU CSV writers
# -----------------------------
imu_topics = ['/imu/data', '/imu/dq', '/imu/dv', '/imu/mag', '/imu/time_ref']
imu_files = {}
imu_writers = {}
for t in imu_topics:
    fname = os.path.join(dirs['imu'], t.strip('/').replace('/', '_') + '.csv')
    f = open(fname, 'w', newline='')
    imu_files[t] = f
    w = csv.writer(f)
    imu_writers[t] = w
    w.writerow(['timestamp', 'data...'])  # generic header

# -----------------------------
# 4. Process bag messages
# -----------------------------
print("[INFO] Starting extraction...")

# Keep track of files for associations.txt
left_images = []
right_images = []
lidar_scans = []
tf_lines = []

for topic, msg, t in tqdm(bag.read_messages(), total=bag.get_message_count()):
    timestamp = t.to_nsec()
    
    # ---------- LIDAR ----------
    if topic == '/bf_lidar/points_raw':
        pcd_fname = f"{timestamp}.pcd"
        pcd_path = os.path.join(dirs['lidar'], pcd_fname)
        # Convert PointCloud2 to PCD using rosbag library
        try:
            from sensor_msgs import point_cloud2
            import open3d as o3d
            points = list(point_cloud2.read_points(msg, skip_nans=True))
            if points:
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector([[p[0], p[1], p[2]] for p in points])
                o3d.io.write_point_cloud(pcd_path, pc)
                lidar_scans.append((timestamp, pcd_fname))
        except Exception as e:
            print(f"Failed to convert PointCloud2: {e}")
    
    # ---------- IMU ----------
    elif topic in imu_topics:
        writer = imu_writers[topic]
        # Flatten the message into a list of floats
        data_row = [timestamp]
        if hasattr(msg, '__slots__'):
            for field in msg.__slots__:
                val = getattr(msg, field)
                if hasattr(val, '__slots__'):
                    # Nested message
                    for sub in val.__slots__:
                        data_row.append(getattr(val, sub))
                else:
                    data_row.append(val)
        writer.writerow(data_row)
    
    # ---------- Camera images ----------
    elif topic == '/stereo/left/image_rect':
        from cv_bridge import CvBridge
        import cv2
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        fname = f"{timestamp}.png"
        path = os.path.join(dirs['left'], fname)
        cv2.imwrite(path, img)
        left_images.append((timestamp, fname))
    
    elif topic == '/stereo/right/image_rect':
        from cv_bridge import CvBridge
        import cv2
        bridge = CvBridge()
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        fname = f"{timestamp}.png"
        path = os.path.join(dirs['right'], fname)
        cv2.imwrite(path, img)
        right_images.append((timestamp, fname))
    
    # ---------- Camera info ----------
    elif topic.endswith('camera_info'):
        fname = f"{timestamp}.yaml"
        path = os.path.join(dirs['camera_info'], fname)
        try:
            with open(path, 'w') as f:
                yaml.dump(msg.__slots__, f)
        except Exception:
            pass  # skip if cannot write

# -----------------------------
# 5. Create associations.txt
# -----------------------------
assoc_path = os.path.join(out_root, "associations.txt")
with open(assoc_path, 'w') as f:
    for l, r, pcd in zip(left_images, right_images, lidar_scans):
        f.write(f"{l[0]} {l[1]} {r[0]} {r[1]} {pcd[0]} {pcd[1]}\n")

# -----------------------------
# 6. Close IMU files
# -----------------------------
for f in imu_files.values():
    f.close()

bag.close()
print("[INFO] Extraction completed!")
