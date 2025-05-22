import os
import cv2
import numpy as np
import rawpy
import imageio
from tqdm import tqdm
import math
import yaml
import pyrealsense2 as rs


def extract_intrinsics(base_folder, rgb_intrinsics, depth_sensor):
    fx = rgb_intrinsics.fx
    fy = rgb_intrinsics.fy
    cx = rgb_intrinsics.ppx
    cy = rgb_intrinsics.ppy
    width = rgb_intrinsics.width
    height = rgb_intrinsics.height
    dist_coeffs = np.asarray(rgb_intrinsics.coeffs)

    
    depth_scale = depth_sensor.get_depth_scale()

    camera_info = {
        "Intrinsics": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "k1": float(dist_coeffs[0]),
            "k2": float(dist_coeffs[1]),
            "p1": float(dist_coeffs[2]),
            "p2": float(dist_coeffs[3]),
            "k3": float(dist_coeffs[4]),
            "distorted": True,
            "width": int(width),
            "height": int(height),
            "depth_scale": float(depth_scale)
            }
        }   
    
    file_path = os.path.join(base_folder, "my_dataset.yaml")

    with open(file_path, 'w') as file:
        yaml.dump(camera_info, file, default_flow_style=False)


def load_raw_image(file_path, width, height):
    """
    Load a 16-bit raw grayscale image and return it as a NumPy array.
    """
    with open(file_path, 'rb') as f:
        raw_data = np.fromfile(f, dtype=np.uint16)
    if raw_data.shape[0] != height*width:
        print("Raw data shape:", raw_data.shape, "Expected shape:", height*width) 
    else:
        image = raw_data.reshape((height, width))
        return image

def RAW2RGB(input_folder, output_folder, width = 1920, height = 1080, bayer_pattern=cv2.COLOR_BAYER_GR2BGR):
    """
    Process all .raw files in the input folder, demosaic, and save as RGB.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in tqdm(os.listdir(input_folder)):
        if file_name.endswith('.dng'):
            input_path = os.path.join(input_folder, file_name)
            
            raw_image = load_raw_image(input_path, width, height)
            bgr_image = cv2.cvtColor(raw_image, bayer_pattern)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

            output_path = os.path.join(output_folder, file_name.replace('.dng', '.png'))
            cv2.imwrite(output_path, rgb_image)


def RAW2sRGB(dng_path, output_folder):    
    for file_name in tqdm(os.listdir(dng_path)):
        if file_name.endswith('.dng'):
            input_path = os.path.join(dng_path, file_name)
            with rawpy.imread(input_path) as raw:
                # Process the DNG file into an sRGB image
                rgb_srgb = raw.postprocess(
                    use_camera_wb=True,            # Use white balance from metadata
                    output_color=rawpy.ColorSpace.sRGB,  # Convert to sRGB color space
                    gamma=(2.2, 0.0),              # Apply sRGB gamma (standard 2.2)
                    no_auto_bright=True,           # Do not apply automatic brightness adjustments
                    output_bps=8                   # Output 8-bit per channel (standard for images)
                )
                rgb_srgb = rgb_srgb[..., [2, 1, 0]]  # Convert from BGR to RGB

                # Save the sRGB image as a PNG or JPG
                output_path = os.path.join(output_folder, file_name.replace('.dng', '.png'))
                imageio.imwrite(output_path, rgb_srgb)  

def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def testing_record(raw_folder, rgb_folder, srgb_folder):
    raw_count = sum(1 for entry in os.listdir(raw_folder))

    # Test RGB images
    rgb_count = sum(1 for entry in os.listdir(rgb_folder))
    
    rgb_sample = cv2.imread(os.path.join(rgb_folder, os.listdir(rgb_folder)[0]), cv2.IMREAD_UNCHANGED)
    rgb_bit_depth = rgb_sample.dtype.itemsize * 8

    if rgb_count != raw_count:
        print(f"❌ Error:  Some RGB images haven't been converted. RGB images:({rgb_count}) vs. RAW images: ({raw_count})")
        print("Missing RGB images:")
        for i in range(raw_count):
            if not os.path.exists(os.path.join(rgb_folder, f"{i}.png")):
                print(f"{i}.png missing")
    else:
        if rgb_bit_depth != 16:
            print(f"❌ Error: RGB images bit depth ({rgb_bit_depth}) is not 16 bits.")
        else:
            print(f"✅ RGB images are correct and 16 bits.")

    # Test sRGB images
    srgb_count = sum(1 for entry in os.listdir(srgb_folder))

    srgb_sample = cv2.imread(os.path.join(srgb_folder, os.listdir(srgb_folder)[0]), cv2.IMREAD_UNCHANGED)
    srgb_bit_depth = srgb_sample.dtype.itemsize * 8

    if srgb_count != raw_count:
        print(f"❌ Error: Some sRGB images haven't been converted. sRGB images: ({srgb_count}) vs. RAW images: ({raw_count})")
        print("Missing sRGB images:")
        for i in range(raw_count):
            if not os.path.exists(os.path.join(srgb_folder, f"{i}.png")):
                print(f"{i}.png missing")
    else:
        if srgb_bit_depth != 8:
            print(f"❌ Error: sRGB images bit depth ({srgb_bit_depth}) is not 8 bits.")
        else:
            print(f"✅ sRGB images are correct and 8 bits.")
    
