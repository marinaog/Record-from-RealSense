import os
import numpy as np
from tqdm import tqdm
import csv

def extract_solid_data_from_csv(csv_file, output_folder, solid_name, motiontracker_frame, final_limit):
    with open(csv_file, newline='') as csvfile:
        reader = list(csv.reader(csvfile))

    # Data starts at row 7 (index 6 is the header)
    data_rows = reader[7:]

    results = {}
    print("Processing in the CSV file...")
    for row in tqdm(data_rows):
        frame = row[0]
        values = row[1:]

        # if any(v == '' or v is None for v in values):
        #     print("Frame", frame, "is empty or incomplete, skipping...")
        #     continue  # skip incomplete rows

        try:
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
                'time': time,
                'x': x,
                'y': y,
                'z': z,
                'rotx': rotx,
                'roty': roty,
                'rotz': rotz
            }
        except Exception as e:
            print(f"Skipping row {frame} due to error: {e}")
            continue

    # Sort frames
    sorted_frames = sorted(results.items())
    print("sorted frames", sorted_frames[motiontracker_frame:motiontracker_frame+10])
    # Write to file
    output_path = os.path.join(output_folder, f'{solid_name}_motion_data.txt')
    with open(output_path, 'w') as f:
        print("")
        print("Writing motion data to file...")
        i=0
        for frame, data in tqdm(sorted_frames[motiontracker_frame:final_limit]):
            print("frame",frame)
            if i == 0:
                print("frame",frame, "motiontracker_frame", motiontracker_frame)
                i+=1
            if frame == motiontracker_frame:
                reference_time = data['time']
            accumulated_time = (data['time'] - reference_time) * 1e3
            f.write(f"{frame} {accumulated_time:.1f} {data['x']} {data['y']} {data['z']} {data['rotx']} {data['roty']} {data['rotz']}\n")

    print(f"Saved motion data for '{solid_name}' to: {output_path}")
    return output_path



def generate_gt(camera_times, motion_tracker_times, camera_frame, output_folder):
    
    with open(camera_times, 'r') as fc:
        camera_lines = fc.readlines()

    with open(motion_tracker_times, 'r') as fmt:
        motiontracker_lines = fmt.readlines()
    
    motion_data = []
    for line in motiontracker_lines:
        parts = line.strip().split()
        frame = int(parts[0])
        time = float(parts[1])
        motion_data.append((frame, time, line.strip()))

    # Go through the camera times and save motion tracker whose times correspond to the camera times
    matched_lines = []
    
    first = True
    for line in camera_lines[camera_frame:]: # We start from the camera frame of synchronization
        cam_frame, cam_time = line.strip().split()
        cam_frame = int(cam_frame)
        cam_time = float(cam_time)

        # Set to 0 the first time of camera
        if first:
            firts_time_cam = cam_time
            first = False
        
        cam_time = cam_time - firts_time_cam


        # Find the closest motion tracker time
        closest = min(motion_data, key=lambda x: abs(x[1] - cam_time))

        # Rename the frame numbers column to match the camera frames number
        list_to_save = closest[2].split()
        list_to_save[0] = str(cam_frame) 
        
        matched_lines.append(list_to_save)
        # print(f"Camera frame {cam_frame} time {cam_time} -> matched motion frame {closest[0]} time {closest[1]}")

    # Create gt.txt file
    gt_path = os.path.join(output_folder, 'groundtruth.txt')
    print("")
    print("Writing gt file in", gt_path)
    with open(gt_path, 'w') as fgt:
        fgt.write(f"# Camera and depth frames used | Resetted Time (ms) | x (m) | y (m) | z (m) | rotx | roty | rotz \n")
        for line in matched_lines:
            fgt.write(" ".join(line) + "\n")


# Path to files
camera_times = r"C:\Users\marin\OneDrive - UNIVERSIDAD DE SEVILLA\Escritorio\Thesis\Our dataset\Record\recordings\one_device\recording\camera_time.txt"
csv_file = r"C:\Users\marin\OneDrive - UNIVERSIDAD DE SEVILLA\Escritorio\Thesis\Our dataset\recording.csv"

# Manually synchronized frames
camera_frame = 244
motiontracker_frame = 1530

# Where to save the times of the Motion Tracker
output_folder = r"C:\Users\marin\OneDrive - UNIVERSIDAD DE SEVILLA\Escritorio\Thesis\Our dataset"

# Convert CSV to motion tracker times .txt file
solid_name = "solid1" # Tracked solid
final_limit = 7114    # Number of frames from which the CSV starts being empty or with ,,,,,  (Set to None if there aren't)

assert final_limit is None or final_limit > motiontracker_frame, "Final limit must be less than motion tracker frame"

motion_tracker_times = extract_solid_data_from_csv(csv_file, output_folder, solid_name, motiontracker_frame, final_limit)

# Synchronize times and get a final gt
generate_gt(camera_times, motion_tracker_times, camera_frame, output_folder)