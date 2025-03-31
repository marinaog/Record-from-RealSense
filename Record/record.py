import pyrealsense2 as rs
import numpy as np
import json
import os
import datetime
import time
import keyboard
import matplotlib.pyplot as plt
from pidng.core import RPICAM2DNG, BaseCameraModel, DNGTags, Tag
from argparse import ArgumentParser

# For metadata in json files  ->  sRGB-to-XYZ color spaces conversion matrix
_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041]
])

def find_devices(two_devices):
    if two_devices:
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
                print(f"Devices connected:")
                for dev in devices:
                    print(dev.get_info(rs.camera_info.name))
            exit()

        return d435, l515
    else:
        """Finds and assigns D435i as the only device."""
        ctx = rs.context()
        devices = ctx.query_devices()

        d435i = None

        for dev in devices:
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)

            if "D435I" in name:
                d435i = serial

        not_connected = []
        if not d435i:
            not_connected.append("d435i")
        else:
            print(f"✅ D435i: {d435i}")
            
        if len(not_connected) > 0:
            for device in not_connected:
                print(f"❌ {device} not connected.")
                print(f"Devices connected:")
                for dev in devices:
                    print(dev.get_info(rs.camera_info.name))
            exit()

        return d435i
    

def create_directories():
    """Creates a timestamped directory structure for storing data."""
    timestamp = datetime.datetime.now().strftime("%m.%d-%H.%M")
    base_folder = f"Record/recordings/{timestamp}"

    depth_folder = os.path.join(base_folder, "depth")
    raw_folder = os.path.join(base_folder, "raw")
    imu_folder = os.path.join(base_folder, "imu")

    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(imu_folder, exist_ok=True)

    return base_folder, depth_folder, raw_folder, imu_folder

class D435CameraModel(BaseCameraModel):
    def __init__(self):
        super().__init__()
        # You would define the specific camera tags and format here
        self.fmt = {"size": (1920, 1080), "stride": 1920, "bpp": 16, "format": "raw16"}  # example values
        self.tags = self.create_tags()  # Create or assign your tags here

    def create_tags(self):
        # Dummy implementation - populate with actual D435 tags
        tags = DNGTags()
        tags.set(Tag.ImageWidth, 1920)
        tags.set(Tag.ImageLength, 1080)
        tags.set(Tag.BitsPerSample, 16)
        tags.set(Tag.Compression, 1)   
        tags.set(Tag.Software, "RealSense D435")
        return tags
    
def main():
    parser = ArgumentParser(description="Recording data from intel RealSense devices")
    parser.add_argument("--two_devices", default=False, type=bool)
    parser.add_argument("--show", default=False, type=bool)
    
    args = parser.parse_args()
    
    # Identify devices
    if args.two_devices:
        d435_serial, l515_serial = find_devices(args.two_devices)
    else:
        d435_serial = find_devices(args.two_devices)
    
    # Create directories
    base_folder, depth_folder, raw_folder, imu_folder = create_directories()

    # Create pipelines for D435(i) (Device A)
    pipeline_A = rs.pipeline()
    config_A = rs.config()

    # Configure D435 or D435i (Device A) - Depth + RAW
    config_A.enable_device(d435_serial)
    config_A.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config_A.enable_stream(rs.stream.color, 1920, 1080, rs.format.raw16, 30)  

    if args.two_devices:
        # Create pipeline for L515 (Device B)
        pipeline_B = rs.pipeline()
        config_B = rs.config()

        # Configure L515 (Device B) - IMU
        config_B.enable_device(l515_serial)
        config_B.enable_stream(rs.stream.gyro)
        config_B.enable_stream(rs.stream.accel)
    else:
        # Configure D435i (Device A) - IMU
        config_A.enable_stream(rs.stream.gyro)
        config_A.enable_stream(rs.stream.accel)

    # Show RAW data
    if args.show:
        plt.ion()
        fig = plt.figure()

    try:
        if args.two_devices:
            print("⏳ Starting D435...")
            pipeline_A.start(config_A)
            time.sleep(1)  # Delay to avoid conflicts

            print("⏳ Starting L515...")
            pipeline_B.start(config_B)
        else:
            print("⏳ Starting D435i...")
            pipeline_A.start(config_A)

        print(f"✅ Recording started! Data will be saved in: {base_folder}")
        print("Press 'q' to stop.")

        frame_count = 0

        while True:
            if keyboard.is_pressed('q'):
                break

            if args.show and plt.fignum_exists(fig.number) == False:
                break

            # Receive data from devices
            frames_A = pipeline_A.wait_for_frames()

            depth_frame = frames_A.get_depth_frame()
            raw_frame = frames_A.get_color_frame()

            if args.two_devices:           
                frames_B = pipeline_B.wait_for_frames()
                # imu data            
                accel_frame = frames_B.first_or_default(rs.stream.accel)
                gyro_frame = frames_B.first_or_default(rs.stream.gyro)
            else: 
                accel_frame = frames_A.first_or_default(rs.stream.accel)
                gyro_frame = frames_A.first_or_default(rs.stream.gyro)


            if not accel_frame or not gyro_frame or not depth_frame or not raw_frame:
                print("Error")
                break

            time_of_arrival = raw_frame.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
            dt = time_of_arrival - previous_time_of_arrival if frame_count > 0 else 0
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            imu_data = [dt, accel_data.x, accel_data.y, accel_data.z, gyro_data.x, gyro_data.y, gyro_data.z]

            if frame_count > 0:
                print("frame count", frame_count,"dt",dt, time_of_arrival, previous_time_of_arrival)
            else:
                print("frame count", frame_count,"dt",dt, time_of_arrival)

            previous_time_of_arrival = time_of_arrival

            np.save(os.path.join(imu_folder + f"/{frame_count}.npy"), np.array(imu_data))

            # Depth 
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            depth_image.tofile(depth_folder + f"/{frame_count}.raw")

            # Raw images
            raw_frame = frames_A.get_color_frame()
            raw_image = np.asanyarray(raw_frame.get_data(), dtype=np.uint16)
            camera_model = D435CameraModel()
            dng = RPICAM2DNG(camera_model)
            dng.options(path=raw_folder, compress=False)
            dng.convert(raw_image, filename=f"{frame_count}.dng")
            
            if args.show:
                plt.imshow(raw_image, cmap='gray')
                plt.title("Live Raw Data")
                plt.draw()
                plt.pause(0.01)

            # Metadata
            exposure_us = raw_frame.get_frame_metadata(rs.frame_metadata_value.actual_exposure) if raw_frame.supports_frame_metadata(rs.frame_metadata_value.actual_exposure) else None
            gain = (raw_frame.get_frame_metadata(rs.frame_metadata_value.gain_level)
                    if raw_frame.supports_frame_metadata(rs.frame_metadata_value.gain_level)
                    else 1  # Default gain if not available
                )            
            base_exposure = 100  # Assume ISO 100 at 100µs exposure
            iso = (gain * (exposure_us / base_exposure)) if exposure_us else None

            metadata = {
                "timestamp": time.time(),
                "frame_number": frame_count,
                "exposure": exposure_us,
                "gain": raw_frame.get_frame_metadata(rs.frame_metadata_value.gain_level) if raw_frame.supports_frame_metadata(rs.frame_metadata_value.gain_level) else None,
                "blacklevel": 0, # raw_frame.get_frame_metadata(rs.frame_metadata_value.black_level) if raw_frame.supports_frame_metadata(rs.frame_metadata_value.black_level) else None,
                "whitelevel": 65535,  #raw_frame.get_frame_metadata(rs.frame_metadata_value.white_level) if raw_frame.supports_frame_metadata(rs.frame_metadata_value.white_level) else None,
                "brightness": raw_frame.get_frame_metadata(rs.frame_metadata_value.brightness) if raw_frame.supports_frame_metadata(rs.frame_metadata_value.brightness) else None,
                "white_balance": raw_frame.get_frame_metadata(rs.frame_metadata_value.white_balance) if raw_frame.supports_frame_metadata(rs.frame_metadata_value.white_balance) else None,
                "shutter_speed": f"1/{int(round(1_000_000 / exposure_us))}" if exposure_us else None,
                "ISO": iso,
                }
            as_shot_neutral = (
                np.array([1.0, 1.0, 1.0])  # Default neutral white balance
                if not raw_frame.supports_frame_metadata(rs.frame_metadata_value.white_balance)
                else np.array(raw_frame.get_frame_metadata(rs.frame_metadata_value.white_balance))
            )

            try:
                color_matrix_2 = np.array(raw_frame.get_frame_metadata(rs.frame_metadata_value.color_matrix)).reshape(3, 3)
            except AttributeError:
                # print("⚠️ ColorMatrix2 not found, using default matrix.")
                color_matrix_2 = np.eye(3)  # Use an identity matrix as fallback


            # Store them properly in metadata
            metadata["AsShotNeutral"] = as_shot_neutral.tolist()
            metadata["ColorMatrix2"] = color_matrix_2.tolist()
            whitebalance = np.array(metadata['AsShotNeutral']).reshape(-1, 3)
            cam2camwb = np.array([np.diag(1. / x) for x in whitebalance])

            xyz2camwb = np.array(metadata['ColorMatrix2']).reshape(-1, 3, 3)
            rgb2camwb = xyz2camwb @ _RGB2XYZ
            rgb2camwb /= rgb2camwb.sum(axis=-1, keepdims=True)

            try:
                cam2rgb = np.linalg.inv(rgb2camwb) @ cam2camwb
                metadata['cam2rgb'] = cam2rgb.tolist()  # Convert to list for JSON storage
            except np.linalg.LinAlgError:
                print("⚠️ Warning: Color correction matrix is singular, using identity matrix.")
                metadata['cam2rgb'] = np.eye(3).tolist()


            json_path = os.path.join(raw_folder, f"{frame_count}.json")
            with open(json_path, "w") as json_file:
                json.dump(metadata, json_file, indent=4)

            frame_count += 1



    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        print("🔄 Stopping pipelines...")
        pipeline_A.stop()
        if args.two_devices:
            pipeline_B.stop()     

        if args.show:
            plt.ioff()   
            if plt.fignum_exists(fig.number):
                plt.close(fig)
        print("🔄 Recording stopped.")

if __name__ == "__main__":
    main()