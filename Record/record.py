import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import time
import keyboard

def find_devices():
    """Finds and assigns D435 as device A and L515 as device B."""
    ctx = rs.context()
    devices = ctx.query_devices()

    d435 = None
    l515 = None

    for dev in devices:
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)

        if "D435" in name:
            d435 = serial
        elif "L515" in name:
            l515 = serial

    not_connected = []
    if not d435:
        not_connected.append("d435")
    else:
        print(f"✅ D435 (Device A): {d435}")
    if not l515:
        not_connected.append("l515")
    else:
       print(f"✅ L515 (Device B): {l515}")
        
    if len(not_connected) > 0:
        for device in not_connected:
            print(f"❌ {device} not connected.")
        exit()

    return d435, l515

def create_directories():
    """Creates a timestamped directory structure for storing data."""
    timestamp = datetime.datetime.now().strftime("%m.%d-%H.%M")
    base_folder = f"recordings/{timestamp}"

    depth_folder = os.path.join(base_folder, "depth")
    raw_folder = os.path.join(base_folder, "raw")
    imu_folder = os.path.join(base_folder, "imu")

    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(imu_folder, exist_ok=True)

    return base_folder, depth_folder, raw_folder, imu_folder

def main():
    # Identify devices
    d435_serial, l515_serial = find_devices()
    
    # Create directories
    base_folder, depth_folder, raw_folder, imu_folder = create_directories()

    # Create pipelines
    pipeline_A = rs.pipeline()
    pipeline_B = rs.pipeline()
    
    config_A = rs.config()
    config_B = rs.config()

    # Configure D435 (Device A) - Depth + RAW
    config_A.enable_device(d435_serial)
    config_A.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config_A.enable_stream(rs.stream.color, 1920, 1080, rs.format.raw16, 30)  # RAW

    # Configure L515 (Device B) - IMU
    config_B.enable_device(l515_serial)
    config_B.enable_stream(rs.stream.gyro)
    config_B.enable_stream(rs.stream.accel)

    try:
        print("⏳ Starting D435...")
        pipeline_A.start(config_A)
        time.sleep(1)  # Delay to avoid conflicts

        print("⏳ Starting L515...")
        pipeline_B.start(config_B)

        print(f"✅ Recording started! Data will be saved in: {base_folder}")
        print("Press 'q' to stop.")

        frame_count = 0

        while True:
            frames_A = pipeline_A.wait_for_frames()
            frames_B = pipeline_B.wait_for_frames()

            depth_frame = frames_A.get_depth_frame()
            raw_frame = frames_A.get_color_frame()

            accel_frame = frames_B.first_or_default(rs.stream.accel)
            gyro_frame = frames_B.first_or_default(rs.stream.gyro)

            if not accel_frame or not gyro_frame or not depth_frame or not raw_frame:
                print("Error")
                break

            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()

            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
    
            raw_image = np.asanyarray(raw_frame.get_data(), dtype=np.uint16)
            
            
            # Save raw binary files
            depth_image.tofile(depth_folder + f"/{frame_count}.raw")
            raw_image.tofile(raw_folder + f"/{frame_count}.raw")

            frame_count += 1
            
            if keyboard.is_pressed('q'):
                break

    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        print("🔄 Stopping pipelines...")
        pipeline_A.stop()
        pipeline_B.stop()
        print("🔄 Recording stopped.")

if __name__ == "__main__":
    main()