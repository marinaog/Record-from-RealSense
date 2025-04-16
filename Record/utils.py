import os
import cv2
import numpy as np
import rawpy
import imageio

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

def printing(rgb_image):
    # Inspect dtype and shape
    print("Data type:", rgb_image.dtype)
    print("Shape:", rgb_image.shape)  # (height, width, channels)

    # Bits per channel
    bits_per_channel = rgb_image.dtype.itemsize * 8
    print("Bits per channel:", bits_per_channel)

    # # Number of channels
    # num_channels = rgb_image.shape[2] if len(rgb_image.shape) == 3 else 1

    # # Total bits per pixel
    # bits_per_pixel = bits_per_channel * num_channels
    # print("Bits per pixel:", bits_per_pixel)


def RAW2RGB(input_folder, output_folder, width = 1920, height = 1080, print_things = False, bayer_pattern=cv2.COLOR_BAYER_GR2BGR):
    """
    Process all .raw files in the input folder, demosaic, and save as RGB.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.dng'):
            input_path = os.path.join(input_folder, file_name)
            
            raw_image = load_raw_image(input_path, width, height)
            bgr_image = cv2.cvtColor(raw_image, bayer_pattern)
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            if print_things:
                print("image:")
                printing(rgb_image)
                print("")

            output_path = os.path.join(output_folder, file_name.replace('.dng', '.png'))
            cv2.imwrite(output_path, rgb_image)
            print(f"Saved image: {output_path}")
            if print_things:
                print("")
                print("saved image:")
                saved_image = cv2.imread(output_path, cv2.IMREAD_UNCHANGED)
                printing(saved_image)
                print("")


def RAW2sRGB(dng_path, output_folder):    
    for file_name in os.listdir(dng_path):
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
                print(f"Saved the sRGB image to: {output_path}")

if __name__ == "__main__":
    input_folder = "Record/recordings/two_devices/04.08-08.49_test/raw"  # Change this to your input folder path
    output_folder = "Record/recordings/two_devices/04.08-08.49_test/images"  # Change this to your output folder path
    image_width = 1920  # Replace with actual width
    image_height = 1080  # Replace with actual height
    print_things = False
    
    RAW2RGB(input_folder, output_folder, image_width, image_height, print_things)