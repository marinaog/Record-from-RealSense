import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import time

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

    if not d435 or not l515:
        print("❌ Error: Both D435 and L515 must be connected!")
        exit()

    print(f"✅ D435 (Device A): {d435}")
    print(f"✅ L515 (Device B): {l515}")
    return d435, l515

def create_directories():
    """Creates a timestamped directory structure for storing data."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    base_folder = f"recordings/{timestamp}"

    depth_folder = os.path.join(base_folder, "depth")
    color_raw_folder = os.path.join(base_folder, "color_raw")
    color_rgb_folder = os.path.join(base_folder, "color_rgb")
    imu_folder = os.path.join(base_folder, "imu")

    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(color_raw_folder, exist_ok=True)
    os.makedirs(color_rgb_folder, exist_ok=True)
    os.makedirs(imu_folder, exist_ok=True)

    return base_folder, depth_folder, color_raw_folder, color_rgb_folder, imu_folder

def main():
    # Identify devices
    d435_serial, l515_serial = find_devices()

    # Create directories
    base_folder, depth_folder, color_raw_folder, color_rgb_folder, imu_folder = create_directories()

    # Create pipelines
    pipeline_A = rs.pipeline()
    pipeline_B = rs.pipeline()
    
    config_A = rs.config()
    config_B = rs.config()

    # Configure D435 (Device A) - RGB + Depth + RAW
    config_A.enable_device(d435_serial)
    config_A.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config_A.enable_stream(rs.stream.color, 1920, 1080, rs.format.raw16, 30)  # RAW
    config_A.enable_stream(rs.stream.color, 1920, 1080, rs.format.rgb8, 30)  # RGB

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
            # Fetch frames from both devices
            frames_A = pipeline_A.wait_for_frames()
            frames_B = pipeline_B.wait_for_frames()

            # Extract Depth & Color from A (D435)
            depth_frame = frames_A.get_depth_frame()
            color_raw_frame = frames_A.get_color_frame()
            color_rgb_frame = frames_A.get_color_frame()

            if depth_frame and color_raw_frame and color_rgb_frame:
                depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
                color_raw_image = np.asanyarray(color_raw_frame.get_data(), dtype=np.uint16)
                color_rgb_image = np.asanyarray(color_rgb_frame.get_data(), dtype=np.uint8)

                print(f"✅ Captured Frame {frame_count}")

                # Save raw binary files
                depth_image.tofile(f"{depth_folder}/frame_{frame_count}.raw")
                color_raw_image.tofile(f"{color_raw_folder}/frame_{frame_count}.raw")

                # Save RGB image
                corrected_RGB = cv2.cvtColor(color_rgb_image, cv2.COLOR_BGR2RGB)
                cv2.imwrite(f"{color_rgb_folder}/frame_{frame_count}.png", corrected_RGB)

                # Display images
                depth_vis = cv2.convertScaleAbs(depth_image, alpha=0.03)
                cv2.imshow("Depth Stream (D435)", depth_vis)
                cv2.imshow("Color RAW Stream (D435)", color_raw_image)
                cv2.imshow("Color RGB Stream (D435)", corrected_RGB)

            # Extract IMU Data from B (L515)
            accel_frame = frames_B.first_or_default(rs.stream.accel)
            gyro_frame = frames_B.first_or_default(rs.stream.gyro)

            if accel_frame and gyro_frame:
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()

                # Save IMU data
                imu_data = f"{frame_count}, {accel_data.x}, {accel_data.y}, {accel_data.z}, {gyro_data.x}, {gyro_data.y}, {gyro_data.z}\n"
                with open(f"{imu_folder}/imu_data.csv", "a") as f:
                    f.write(imu_data)

                print(f"✅ IMU Data: {imu_data.strip()}")

            frame_count += 1

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        print("🔄 Stopping pipelines...")
        try:
            pipeline_A.stop()
            pipeline_B.stop()
        except RuntimeError:
            pass  # Ignore stop() errors if pipeline never started
        cv2.destroyAllWindows()
        print("🔄 Recording stopped.")

if __name__ == "__main__":
    main()