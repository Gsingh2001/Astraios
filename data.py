import subprocess

# HiDrive file URL
url = "https://my.hidrive.com/api/file?attachment=true&pid=b1584010878.594&access_token=RyFEVdAf5mBFmdmFydB6"

# Output filename
output_file = "hidrive_file.bag"  # Change as needed

# Aria2c command
cmd = [
    "aria2c",
    url,
    "-o", output_file,
    "-x", "16",    # 16 connections per server
    "-s", "16",    # 16 splits
    "-k", "1M",    # 1 MB per split
    "--console-log-level=error",
    "--summary-interval=5"
]

try:
    subprocess.run(cmd, check=True)
    print("Download completed successfully!")
except subprocess.CalledProcessError:
    print("Download failed. Check URL or aria2c installation.")
