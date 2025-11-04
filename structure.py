import os
from collections import defaultdict

def print_folder_tree_limited(startpath, prefix=""):
    try:
        items = os.listdir(startpath)
    except PermissionError:
        print(prefix + "└── [Permission Denied]")
        return

    file_groups = defaultdict(list)
    dirs = []

    for item in items:
        path = os.path.join(startpath, item)
        if os.path.isdir(path):
            dirs.append(item)
        else:
            ext = os.path.splitext(item)[1] or "no_ext"
            file_groups[ext].append(item)

    # Print directories first
    for i, d in enumerate(dirs):
        path = os.path.join(startpath, d)
        connector = "└── " if i == len(dirs) - 1 and not file_groups else "├── "
        print(prefix + connector + d)
        extension = "    " if i == len(dirs) - 1 and not file_groups else "│   "
        print_folder_tree_limited(path, prefix + extension)

    # Print files grouped by type with limit
    for ext, files in file_groups.items():
        for idx, file in enumerate(files):
            if idx >= 2:
                if idx == 2:
                    print(prefix + f"└── +{len(files)-2} more {ext} files")
                break
            connector = "└── " if idx == len(files)-1 or len(files) <= 2 else "├── "
            print(prefix + connector + file)


# ======================================
# ✅ Example usage
# Use a *raw string* (r"") for Windows paths
# ======================================
root_folder = r"G:\backup\papers\Dataset\hidrive_file"
print(f"📁 Folder structure for: {root_folder}\n")
print_folder_tree_limited(root_folder)
