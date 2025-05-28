import os
import numpy as np
from tqdm import tqdm
import csv
import zipfile

def extract_solid_data_from_csv(
    dataset_folder, csv_name, solid_name, motiontracker_frame, final_limit
):
    csv_file = os.path.join(dataset_folder, csv_name)
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return None

    with open(csv_file, newline="") as csvfile:
        reader = list(csv.reader(csvfile))

    # Data starts at row 7 (index 6 is the header)
    data_rows = reader[7:]

    if final_limit is None:
        final_limit = len(data_rows)  # Adjust for header rows
        print(final_limit)
    results = {}
    print("Processing in the CSV file...")
    for row in data_rows:
        frame = row[0]
        values = row[1:]

        if any(v == "" or v is None for v in values):
            print("Frame", frame, "is empty or incomplete, skipping...")
            continue  # skip incomplete rows

        # try:
        time = float(values[0])
        if solid_name == "solid1":
            rotx, roty, rotz = map(float, values[1:4])
            x, y, z = map(float, values[4:7])
        elif solid_name == "solid2":
            rotx, roty, rotz = map(float, values[8:11])
            x, y, z = map(float, values[11:14])
        else:
            raise ValueError(f"Unknown solid name: {solid_name}")

        results[int(frame)] = {
            "time": time,
            "x": x,
            "y": y,
            "z": z,
            "rotx": rotx,
            "roty": roty,
            "rotz": rotz,
        }

    # Sort frames
    results = {
        frame: data
        for frame, data in results.items()
        if motiontracker_frame <= int(frame) <= final_limit
    }

    sorted_frames = sorted(results.items())

    if len(sorted_frames) != final_limit - motiontracker_frame + 1:
        print("THERE ARE WRONG VALUES IN IMPORTANT PARTS OF THE CSV FILE!")

    # Write to file
    output_path = os.path.join(dataset_folder, f"{solid_name}_motion_data.txt")
    print("")
    print("Writing motion data to file...")
    with open(output_path, "w") as f:
        for frame, data in sorted_frames:
            if int(frame) == motiontracker_frame:
                reference_time = data["time"]
            accumulated_time = (data["time"] - reference_time) * 1e3
            f.write(
                f"{frame} {accumulated_time:.1f} {data['x']} {data['y']} {data['z']} {data['rotx']} {data['roty']} {data['rotz']}\n"
            )

    print(f"Saved motion data for '{solid_name}' to: {output_path}")
    return output_path


def generate_gt(
    dataset_folder, motion_tracker_times, camera_frame, cut_beginning, cut_end
):
    camera_times = os.path.join(dataset_folder, "time_camera.txt")
    if not os.path.exists(camera_times):
        print(f"❌ Camera times file not found: {camera_times}")
        return None

    with open(camera_times, "r") as fc:
        camera_lines = fc.readlines()

    with open(motion_tracker_times, "r") as fmt:
        motiontracker_lines = fmt.readlines()

    motion_data = []
    for line in motiontracker_lines:
        parts = line.strip().split()
        frame = int(parts[0])
        time = float(parts[1])
        motion_data.append((frame, time, line.strip()))

    # Go through the camera times and save motion tracker whose times correspond to the camera times
    matched_lines = []
    used_frames = []
    first = True
    for line in camera_lines[
        camera_frame:
    ]:  # We start from the camera frame of synchronization
        cam_frame, cam_time = line.strip().split()
        cam_frame = int(cam_frame)
        cam_time = float(cam_time)

        # Set to 0 the first time of camera
        if first:
            firts_time_cam = cam_time
            first = False

        if cut_beginning is not None and cam_frame < cut_beginning:
            continue

        if cut_end is not None and cam_frame > cut_end:
            continue

        cam_time = cam_time - firts_time_cam

        # Save useful frames
        used_frames.append(cam_frame)

        # Find the closest motion tracker time
        closest = min(motion_data, key=lambda x: abs(x[1] - cam_time))

        # Rename the frame numbers column to match the camera frames number
        list_to_save = closest[2].split()
        list_to_save[0] = str(cam_frame)

        matched_lines.append(list_to_save)
        # print(f"Camera frame {cam_frame} time {cam_time} -> matched motion frame {closest[0]} time {closest[1]}")

    # Fix phase of rotation to avoid jumps of 179 to -179
    rx = [float(row[5]) for row in matched_lines]
    ry = [float(row[6]) for row in matched_lines]
    rz = [float(row[7]) for row in matched_lines]

    rx_unwrapped = np.degrees(np.unwrap(np.radians(rx)))
    ry_unwrapped = np.degrees(np.unwrap(np.radians(ry)))
    rz_unwrapped = np.degrees(np.unwrap(np.radians(rz)))

    for i, row in enumerate(matched_lines):
        row[5] = str(rx_unwrapped[i])
        row[6] = str(ry_unwrapped[i])
        row[7] = str(rz_unwrapped[i])

    # Create gt.txt file
    gt_path = os.path.join(dataset_folder, "groundtruth.txt")
    print("")
    print("Writing gt file in", gt_path)
    with open(gt_path, "w") as fgt:
        fgt.write(
            "# Camera and depth frames used | Resetted Time (ms) | x (m) | y (m) | z (m) | rotx (degrees)| roty (degrees)| rotz (degrees) \n"
        )
        for line in matched_lines:
            fgt.write(" ".join(line) + "\n")

    return used_frames


# def create_zip(used_frames, dataset_folder, groundtruth_path, scene_name):
def create_zip(used_frames, dataset_folder, scene_name):
    print("✍🏼 Creating zip file...")
    output_zip = os.path.join(dataset_folder, scene_name + ".zip")

    groundtruth_path = os.path.join(dataset_folder, "groundtruth.txt")

    with zipfile.ZipFile(output_zip, "w") as zipf:
        # Add groundtruth.txt
        zipf.write(groundtruth_path, arcname="groundtruth.txt")

        # Add corresponding frame files from rgb and depth folders
        for folder_name in ["sRGB", "rgb", "depth"]:
            folder_path = os.path.join(dataset_folder, folder_name)
            print(f"Uploading {folder_name} folder...")
            for frame in tqdm(used_frames):
                frame_file = os.path.join(folder_path, f"{frame}.png")
                if os.path.exists(frame_file):
                    arcname = os.path.join(folder_name, f"{frame}.png")
                    zipf.write(frame_file, arcname=arcname)
                else:
                    print(f"Warning: {folder_name}/{frame}.png not found.")

    print(f"Zip file created: {output_zip}")


# Path to files
dataset_folder = r"C:\Users\marin\OneDrive - UNIVERSIDAD DE SEVILLA\Escritorio\Thesis\Our dataset\Record\recordings\dark_kitchen"
csv_name = "dark_kitchen.csv"

# Manually synchronized frames
camera_frame = 90
motiontracker_frame = 1030

# Cut the video at the beginning and end, set to None if you don't want to cut it (in camera frames)
cut_beginning, cut_end = 205, 546

# Number of frames from which the CSV starts being empty or with ,,,,,  (Set to None if there aren't)
final_limit = None  

# Convert CSV to motion tracker times .txt file
solid_name = "solid2"  # Tracked solid

assert final_limit is None or final_limit > motiontracker_frame, (
    "Final limit must be less than motion tracker frame"
)

motion_tracker_times = extract_solid_data_from_csv(
    dataset_folder, csv_name, solid_name, motiontracker_frame, final_limit
)

# Synchronize times and get a final gt
used_frames = generate_gt(
    dataset_folder, motion_tracker_times, camera_frame, cut_beginning, cut_end
)

# Create a .zip file with everything
scene_name = csv_name[:-4]
# create_zip(used_frames, dataset_folder, scene_name)
