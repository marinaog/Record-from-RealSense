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
import rawpy
import imageio


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
    image_folder = os.path.join(base_folder, "images")

    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(imu_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)

    return base_folder, depth_folder, raw_folder, imu_folder, image_folder

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
    
def extract_metadata(dng_path, json_path, image_folder, file_name):
    try:
        # Open JSON file
        with open(json_path, "r") as json_file:
            metadata = json.load(json_file) 

        # Extract DNG metadata
        with rawpy.imread(dng_path) as raw:
            # print(dir(raw))
            metadata.update({
                "BlackLevel": min(raw.black_level_per_channel),
                "WhiteLevel": raw.white_level,
                "cam2rgb": raw.color_matrix.tolist()
            })

            
            # Post-process RAW image and save as PNG
            processed_image = raw.postprocess()
            img_path = os.path.join(image_folder, file_name.replace('.json', '.png'))
            imageio.imwrite(img_path, processed_image)

        # Save updated metadata
        with open(json_path, "w") as json_file:
            json.dump(metadata, json_file, indent=4)

        print(f"▫️ Metadata and images extracted for {file_name}")

    except json.JSONDecodeError:
        print(f"❌ Error: {json_path} is empty or corrupt.")
    except rawpy.LibRawFileUnsupportedError:
        print(f"❌ Error: {dng_path} is not a valid RAW file.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


    
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
    base_folder, depth_folder, raw_folder, imu_folder, image_folder = create_directories()

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

            # IMU with fps
            if args.two_devices:           
                frames_B = pipeline_B.wait_for_frames()        
                accel_frame = frames_B.first_or_default(rs.stream.accel)
                gyro_frame = frames_B.first_or_default(rs.stream.gyro)
            else: 
                accel_frame = frames_A.first_or_default(rs.stream.accel)
                gyro_frame = frames_A.first_or_default(rs.stream.gyro)

            time_of_arrival = raw_frame.get_frame_metadata(rs.frame_metadata_value.time_of_arrival)
            dt = time_of_arrival - previous_time_of_arrival if frame_count > 0 else 0
            if frame_count > 0:
                print("frame count", frame_count,"dt",dt, time_of_arrival, previous_time_of_arrival)
            else:
                print("frame count", frame_count,"dt",dt, time_of_arrival)
            previous_time_of_arrival = time_of_arrival

            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            imu_data = [dt, accel_data.x, accel_data.y, accel_data.z, gyro_data.x, gyro_data.y, gyro_data.z]

            np.save(os.path.join(imu_folder + f"/{frame_count}.npy"), np.array(imu_data))

            # Depth 
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)
            depth_image.tofile(depth_folder + f"/{frame_count}.raw")

            # Raw images
            raw_frame = frames_A.get_color_frame()
            raw_image = np.asanyarray(raw_frame.get_data(), dtype=np.uint16)
            raw_image.tofile(raw_folder + f"/{frame_count}meta.dng")
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
            metadata = {}

            """Exposure can be obtained directly from RealSense,
            for the rest of metadata need to be accessed with Rawpy"""
            try:
                metadata["exposure"] = raw_frame.get_frame_metadata(rs.frame_metadata_value.auto_exposure)
            except:                
                print("❌ Warning: Exposure metadata not available for this frame.")
                for metadata_key in rs.frame_metadata_value.__members__.values():
                    if raw_frame.supports_frame_metadata(metadata_key):
                        value = raw_frame.get_frame_metadata(metadata_key)
                        print(f"{metadata_key.name}: {value}")
             
            metadata["ISO"] = None

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
        print("")
        print("🔄 Metadata recording started.")
        for file_name in os.listdir(raw_folder):
            if file_name.endswith('.json'):
                json_path = os.path.join(raw_folder, file_name)
                dng_path = os.path.join(raw_folder, file_name.replace('.json', 'meta.dng'))
                extract_metadata(dng_path, json_path, image_folder, file_name)
        print("🏁 Metadata recording finished.")
            

if __name__ == "__main__":
    main()