from pathlib import Path
import numpy as np
import cv2


class LuSNARDataset:
    def __init__(self, root):
        self.root = Path(root)

        # Folders
        self.color_dir = self.root / "image0" / "color"
        self.depth_dir = self.root / "image0" / "depth"
        self.lidar_dir = self.root / "LiDAR"

        # Timestamp files
        self.color_ts = self._load_timestamp_csv(self.root / "image0" / "color_timestamp.txt")
        self.depth_ts = self._load_timestamp_csv(self.root / "image0" / "depth_timestamp.txt")

        # IMU + GT
        self.imu = self._load_table(self.root / "imu.txt")
        self.gt = self._load_table(self.root / "gt.txt")

        print(f"[LuSNAR] Loaded {len(self.color_ts)} frames.")


    # --- timestamp parser that ignores header ---
    def _load_timestamp_csv(self, path):
        timestamps = []
        with open(path, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue  # skip header
                parts = line.strip().split(",")
                if len(parts) >= 1 and parts[0].isdigit():
                    timestamps.append(int(parts[0]))
        return np.array(timestamps, dtype=np.int64)


    # --- imu, gt loaders ---
    def _load_table(self, path):
        return np.loadtxt(path, delimiter=",")


    # --- RGB image ---
    def load_rgb(self, ts):
        img = cv2.imread(str(self.color_dir / f"{ts}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.color_dir / f"{ts}.png")
        return img


    # --- depth PFM reader ---
    def load_depth(self, ts):
        return self._read_pfm(self.depth_dir / f"{ts}.pfm")


    def _read_pfm(self, path):
        with open(path, "rb") as f:
            header = f.readline().decode().strip()
            dims = f.readline().decode().strip()
            width, height = map(int, dims.split())
            scale = float(f.readline().decode().strip())
            endian = "<" if scale < 0 else ">"
            data = np.fromfile(f, endian + "f")
            data = np.reshape(data, (height, width))
            return np.flipud(data)


    # --- LiDAR ---
    def load_lidar(self, ts):
        pts = np.loadtxt(self.lidar_dir / f"{ts}.txt", delimiter=",")
        return pts.astype(np.float32)


    # --- Get one full frame ---
    def get_frame(self, idx):
        ts = self.color_ts[idx]
        rgb = self.load_rgb(ts)
        depth = self.load_depth(ts)
        lidar = self.load_lidar(ts)
        return ts, rgb, depth, lidar
