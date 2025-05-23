import os
import cv2
import rerun as rr
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import rerun.blueprint as rrb

# --- Configuration ---
dataset_folder = r"Record\recordings\objectsonthefloor"
rgb_folder = os.path.join(dataset_folder, "sRGB")
depth_folder = os.path.join(dataset_folder, "depth")
trajectory_file = os.path.join(dataset_folder, "groundtruth.txt")

# Initialize rerun
rr.init("slam_visualization", spawn=True)


# Read trajectory file
poses = []
with open(trajectory_file, "r") as f:
    first = True
    for line in f:
        if first:
            first = False
            continue
        parts = line.strip().split()
        if len(parts) != 8:
            print("Length is not the expected one", len(parts), line)
            break
        frame, timestamp, x, y, z, rx, ry, rz = map(float, parts)
        poses.append({
            "frame": int(frame), 
            "timestamp": timestamp, 
            "position": np.array([x, y, z]),
            "rotation": np.array([rx, ry, rz])
        })

# Log world origin for reference
rr.log("world/origin", rr.Transform3D(translation=[0, 0, 0]))

trajectory_so_far = []
# Iterate through poses and visualize
for ind, pose in enumerate(poses):
    frame = pose["frame"]
    time = pose["timestamp"]
    rotation = R.from_euler('xyz', pose["rotation"], degrees=False)
    # print("position",pose["position"],pose["position"].shape,pose["position"][0])
    # print("rotation",pose["rotation"])
    # if ind == 0:
    #     break
    # quat_xyzw = rotation.as_quat()  # [x, y, z, w]
    # quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # [w, x, y, z]
    # rotation=rr.RotationQuat(wxyz=quat_wxyz)

    # Load trajectory
    trajectory_so_far.append(pose["position"])

    # Load RGB and depth images
    rgb_path = os.path.join(rgb_folder, f"{frame}.png")
    depth_path = os.path.join(depth_folder, f"{frame}.png")

    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f"Missing RGB or depth image for frame {frame}")
        continue

    if not os.path.exists(rgb_path):
        print(f"Missing RGB image at {rgb_path}")
        continue
    rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    # Log images and depth
    rr.set_time_sequence("frame", frame)
    rr.log(f"images/rgb", rr.Image(rgb))
    rr.log(f"images/depth", rr.DepthImage(depth, meter=0.001))  # depth scale
    
    # Log growing trajectory line
    rr.log("trajectory", rr.LineStrips3D([trajectory_so_far]))
    # Log camera pose
    # rr.log(
    #     f"camera/{frame}",
    #     rr.Transform3D(
    #         translation=pose["position"],
    #         rotation=rr.RotationQuat(wxyz=quat_wxyz),
    #     ),
    # )
    # rr.log("points/test", rr.Points3D(positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]]))

    # Show orientation axes
    # rr.log("camera/axes", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)


