import pyrealsense2 as rs
import numpy as np
import cv2
import os
import datetime
import time
import keyboard
import json
from argparse import ArgumentParser
from utils import (
    extract_intrinsics,
    RAW2RGB,
    RAW2sRGB,
    testing_record,
    extract_metadata,
)


def find_devices(two_devices, devices):
    if two_devices:
        """Finds and assigns D435 as device A and L515 as device B."""
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
                print("Devices connected:")
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
                print("Devices connected:")
                for dev in devices:
                    print(dev.get_info(rs.camera_info.name))
            exit()

        return d435i


def create_directories(two_devices):
    """Creates a timestamped directory structure for storing data."""
    timestamp = datetime.datetime.now().strftime("%m.%d-%H.%M")
    if two_devices:
        base_folder = f"Record/recordings/two_devices/{timestamp}"
    else:
        base_folder = f"Record/recordings/one_device/{timestamp}"

    depth_folder = os.path.join(base_folder, "depth")
    raw_folder = os.path.join(base_folder, "raw")
    imu_folder = os.path.join(base_folder, "imu")
    image_folder = os.path.join(base_folder, "rgb")
    srgb_folder = os.path.join(base_folder, "sRGB")
    metadata_folder = os.path.join(base_folder, "metadata")

    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(imu_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(srgb_folder, exist_ok=True)
    os.makedirs(metadata_folder, exist_ok=True)

    time_file_path = os.path.join(base_folder, "time_camera.txt")
    time_file = open(time_file_path, "w")

    return (
        base_folder,
        depth_folder,
        raw_folder,
        imu_folder,
        image_folder,
        srgb_folder,
        time_file,
        metadata_folder,
    )


def intrinsics(base_folder, profile_A):
    rgb_profile = rs.video_stream_profile(profile_A.get_stream(rs.stream.color))
    rgb_intrinsics = rgb_profile.get_intrinsics()

    depth_sensor = profile_A.get_device().first_depth_sensor()

    extract_intrinsics(base_folder, rgb_intrinsics, depth_sensor)


def main():
    parser = ArgumentParser(description="Recording data from intel RealSense devices")
    parser.add_argument("--depthas", default="png", choices=["png", "raw"])

    args = parser.parse_args()

    # Check how many devices are connected
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 2:
        two_devices = True
    elif len(devices) == 1:
        two_devices = False
    else:
        print("❌ No devices connected.")
        exit()

    # Identify devices
    if two_devices:
        d435_serial, l515_serial = find_devices(two_devices, devices)
    else:
        d435_serial = find_devices(two_devices, devices)

    # Create directories
    (
        base_folder,
        depth_folder,
        raw_folder,
        imu_folder,
        image_folder,
        srgb_folder,
        time_file,
        metadata_folder,
    ) = create_directories(two_devices)

    # Create pipelines for D435(i) (Device A)
    pipeline_A = rs.pipeline()
    config_A = rs.config()

    # Configure D435 or D435i (Device A) - Depth + RAW
    config_A.enable_device(d435_serial)
    config_A.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
    config_A.enable_stream(rs.stream.color, 1920, 1080, rs.format.raw16, 30)

    # Create align object to align depth to color
    align_to = rs.stream.color
    align = rs.align(align_to)

    if two_devices:
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

    try:
        if two_devices:
            print("⏳ Starting D435...")
            profile_A = pipeline_A.start(config_A)
            time.sleep(1)  # Delay to avoid conflicts

            print("⏳ Starting L515...")
            pipeline_B.start(config_B)
        else:
            print("⏳ Starting D435i...")
            profile_A = pipeline_A.start(config_A)

        print(f"🎥 Recording started! Data will be saved in: {base_folder}")
        print("Press 'q' to stop.")

        frame_count = 0

        while True:
            if keyboard.is_pressed("q"):
                break

            # Receive data from devices
            frames_A = pipeline_A.wait_for_frames()

            # Align frames
            aligned_frames = align.process(frames_A)

            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            raw_frame = aligned_frames.get_color_frame()

            # IMU with fps
            if two_devices:
                frames_B = pipeline_B.wait_for_frames()
                accel_frame = frames_B.first_or_default(rs.stream.accel)
                gyro_frame = frames_B.first_or_default(rs.stream.gyro)
            else:
                accel_frame = frames_A.first_or_default(rs.stream.accel)
                gyro_frame = frames_A.first_or_default(rs.stream.gyro)

            time_of_arrival = raw_frame.get_frame_metadata(
                rs.frame_metadata_value.time_of_arrival
            )
            dt = time_of_arrival - previous_time_of_arrival if frame_count > 0 else 0
            if frame_count > 0:
                print("frame count", frame_count, ",    dt", dt)
            else:
                print("frame count", frame_count, ",    dt", dt)
            previous_time_of_arrival = time_of_arrival

            accel_data = accel_frame.as_motion_frame().get_motion_data()
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            imu_data = [
                dt,
                accel_data.x,
                accel_data.y,
                accel_data.z,
                gyro_data.x,
                gyro_data.y,
                gyro_data.z,
            ]

            np.save(
                os.path.join(imu_folder + f"/{frame_count}.npy"), np.array(imu_data)
            )

            # Time text
            if frame_count == 0:
                time_of_reference = time_of_arrival

            time_to_record = time_of_arrival - time_of_reference

            time_file.write(f"{frame_count} {time_to_record}\n")

            # Depth
            depth_image = np.asanyarray(depth_frame.get_data(), dtype=np.uint16)

            if args.depthas == "raw":
                depth_image.tofile(depth_folder + f"/{frame_count}.raw")
            else:
                cv2.imwrite(depth_folder + f"/{frame_count}.png", depth_image)

            # Raw images
            raw_frame = frames_A.get_color_frame()
            raw_image = np.asanyarray(raw_frame.get_data(), dtype=np.uint16)
            if raw_image.size != 2073600:
                raise KeyError(
                    "❌ Error: Raw image size mismatch. Expected 2073600 pixels."
                )

            raw_image.tofile(raw_folder + f"/{frame_count}.dng")

            # Metadata
            metadata = {}
            """Exposure can be obtained directly from RealSense,
            for the rest of metadata need to be accessed with Rawpy"""
            try:
                metadata["exposure"] = raw_frame.get_frame_metadata(
                    rs.frame_metadata_value.auto_exposure
                )
            except Exception:
                print(
                    "❌ Warning: Exposure metadata not available for frame {frame_count}."
                )

            print("Supported metadata keys:")
            if frame_count == 0:
                for metadata_key in rs.frame_metadata_value.__members__.values():
                    if raw_frame.supports_frame_metadata(metadata_key):
                        value = raw_frame.get_frame_metadata(metadata_key)
                        print(f"{metadata_key.name}: {value}")

            metadata["ISO"] = None

            json_path = os.path.join(metadata_folder, f"{frame_count}.json")
            with open(json_path, "w") as json_file:
                json.dump(metadata, json_file, indent=4)

            frame_count += 1

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        print("")
        print("🔄 Creating intrinsics file.")
        intrinsics(base_folder, profile_A)
        print("")

        print("🛑 Stopping pipelines...")
        pipeline_A.stop()
        if two_devices:
            pipeline_B.stop()

        time_file.close()

        print("🏁 Recording stopped.")
        print("")

        print("🔄 Images extraction started.")
        RAW2RGB(raw_folder, image_folder)
        print("")

        print("🔄 sRGB extraction started.")
        RAW2sRGB(raw_folder, srgb_folder)
        print("")

        print(" Testing recording...")
        testing_record(raw_folder, image_folder, srgb_folder)
        print("")

        print("🔄 Metadata extraction started.")
        extract_metadata(raw_folder, metadata_folder)
        print("")

        print("🏁 Dataset extraction finished.")


if __name__ == "__main__":
    main()
