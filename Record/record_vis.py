import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec  

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

def draw_imu_visualization(frame, accel, gyro_data):
    """Draws IMU acceleration data as a horizontal bar graph on the frame."""
    h, w, _ = frame.shape
    bar_height = h // 10  # Height of each bar
    center_x = w // 2
    center_y = h // 2
    max_width = w // 3  # Maximum bar width
    scale = max_width / 4  # Scaling factor for better visualization

    # Acceleration values
    accel_values = [accel.x, accel.y, accel.z]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # Blue (X), Green (Y), Red (Z)

    # Base positions for bars (stacked vertically)
    base_y = [center_y - bar_height, center_y, center_y + bar_height]
    bar_ends = [int(center_x + scale * val) for val in accel_values]

    # Draw bars
    for i in range(3):
        cv2.rectangle(frame, 
                      (center_x, base_y[i] - bar_height // 2),  
                      (bar_ends[i], base_y[i] + bar_height // 2),  
                      colors[i], 
                      thickness=-1)  

        # Add labels
        cv2.putText(frame, f"{accel_values[i]:.2f}", 
                    (bar_ends[i] + 10 if accel_values[i] >= 0 else bar_ends[i] - 50, base_y[i] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 1)

    # Draw axis line
    cv2.line(frame, (center_x, center_y - 2 * bar_height), (center_x, center_y + 2 * bar_height), (255, 255, 255), 2)

    return frame


def draw_imu_visualization_3d(accel, gyro):
    """
    Draws a 3D visualization of accelerometer and gyroscope data.
    :param accel: Tuple (ax, ay, az) for acceleration values
    :param gyro: Tuple (gx, gy, gz) for gyroscope values
    """
    fig = plt.figure(figsize=(10, 5))
    
    # Accelerometer plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.quiver(0, 0, 0, accel[0], accel[1], accel[2], color=['r', 'g', 'b'], length=1, normalize=True)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([-1, 1])
    ax1.set_zlim([-1, 1])
    ax1.set_title("Accelerometer")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    
    # Gyroscope plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.quiver(0, 0, 0, gyro[0], gyro[1], gyro[2], color=['r', 'g', 'b'], length=1, normalize=True)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-1, 1])
    ax2.set_zlim([-1, 1])
    ax2.set_title("Gyroscope")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    plt.show()

def main():
    # Identify devices
    d435_serial, l515_serial = find_devices()
    
    # Create directories
    base_folder, depth_folder, color_raw_folder, imu_folder = create_directories()

    # Create pipelines
    pipeline_A = rs.pipeline()
    pipeline_B = rs.pipeline()
    
    config_A = rs.config()
    config_B = rs.config()

    # Configure D435 (Device A) - RGB + Depth + RAW
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

        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(2, 2, height_ratios=[1, 0.5])   
        ax1 = fig.add_subplot(gs[0, 0])  
        ax2 = fig.add_subplot(gs[0, 1])  
        ax3 = fig.add_subplot(gs[1, :])   


        frame_count = 0
        while True:
            frames_A = pipeline_A.wait_for_frames()
            frames_B = pipeline_B.wait_for_frames()

            depth_frame = frames_A.get_depth_frame()
            color_raw_frame = frames_A.get_color_frame()

            imu_display = np.zeros((300, 1200, 3), dtype=np.uint8)
            accel_frame = frames_B.first_or_default(rs.stream.accel)
            gyro_frame = frames_B.first_or_default(rs.stream.gyro)

            if accel_frame and gyro_frame:
                accel_data = accel_frame.as_motion_frame().get_motion_data()
                gyro_data = gyro_frame.as_motion_frame().get_motion_data()
                imu_display = draw_imu_visualization(imu_display, accel_data, gyro_data)

            if depth_frame and color_raw_frame:
                depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                
                color_raw_image = np.asanyarray(color_raw_frame.get_data(), dtype=np.uint16)
                
                               
                ax1.clear()
                ax1.imshow(depth_colormap, cmap='gray')
                ax1.set_title("Depth Image")
                ax1.axis('off')
                
                ax2.clear()
                ax2.imshow(color_raw_image, cmap='gray')
                ax2.set_title("Raw Image")
                ax2.axis('off')
                
                ax3.clear()
                ax3.imshow(imu_display)
                ax3.set_title("IMU Visualization")
                ax3.axis('off')
                
                plt.draw()
                plt.pause(0.001)

            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        print("🔄 Stopping pipelines...")
        pipeline_A.stop()
        pipeline_B.stop()
        plt.ioff()
        plt.show()
        print("🔄 Recording stopped.")

if __name__ == "__main__":
    main()