import os
import cv2
import rawpy
import imageio
from tqdm import tqdm
import numpy as np

from Record.utils import load_raw_image

def RAW2RGB(
    frame,
    input_file,
    output_folder,
    width=1920,
    height=1080,
    bayer_pattern=cv2.COLOR_BAYER_GR2BGR,
):
    """
    Process .dng files demosaic, and save as RGB.
    """
    output_path = os.path.join(output_folder, str(frame) + '.png')
    if os.path.isfile(output_path):
        return output_path
    raw_image = load_raw_image(input_file, width, height)
    bgr_image = cv2.cvtColor(raw_image, bayer_pattern)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    cv2.imwrite(output_path, rgb_image)
    return output_path


def RAW2sRGB(frame, input_file, sRGB_folder):
    """
    sRGB processing from RealSensethat I cant use later
    Process .dng files and converts them into sRGB in the way realsense makes it!.
    """
    with rawpy.imread(input_file) as raw:
        rgb_srgb = raw.postprocess(
            use_camera_wb=True,  # Use white balance from metadata
            output_color=rawpy.ColorSpace.sRGB,  # Convert to sRGB color space
            gamma=(2.2, 0.0),  # Apply sRGB gamma (standard 2.2)
            no_auto_bright=True,  # Do not apply automatic brightness adjustments
            output_bps=8,  # Output 8-bit per channel (standard for images)
        )
        rgb_srgb = rgb_srgb[..., [2, 1, 0]]  # Convert from BGR to RGB

        # Save the sRGB image as a PNG 
        output_path = os.path.join(sRGB_folder,str(frame) + '.png')
        imageio.imwrite(output_path, rgb_srgb)


def gamma_encode_srgb(x):
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * x ** (1 / 2.4) - a)

def RGB2sRGB(frame, rgb_file, srgb_folder):    
    output_path = srgb_folder + '/' + str(frame) + '.png'
    if not os.path.isfile(output_path):
        img_demosaicked = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED) 
        img_demosaicked = cv2.cvtColor(img_demosaicked, cv2.COLOR_BGR2RGB)

        # Step 1: Normalize
        if img_demosaicked.dtype == np.uint8:
            img = img_demosaicked.astype(np.float32) / 255.0
        elif img_demosaicked.dtype == np.uint16:
            img = img_demosaicked.astype(np.float32) / 65535.0

        # Step 2: Apply white balance (generic daylight)
        wb = np.array([2.0, 1.0, 1.5])
        img_wb = img * wb  # broadcast over channels

        # Step 3: Color correction matrix (generic)
        ccm = np.array([
            [1.8, -0.6, -0.2],
            [-0.3, 1.3, 0.0],
            [0.0, -0.6, 1.6]
        ])
        img_ccm = np.tensordot(img_wb, ccm.T, axes=1)

        # Step 4: Clip
        img_ccm = np.clip(img_ccm, 0, 1)

        # Step 5: Gamma encode (sRGB)
        img_srgb = gamma_encode_srgb(img_ccm)
        img_srgb = np.clip(img_srgb, 0, 1)

        # Save
        img_to_save = (img_srgb * 255).astype(np.uint8)
        img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, img_to_save)


def testing_record(gt_lines, rgb_folder, srgb_folder):
    # First frame
    first_frame = gt_lines[0].strip().split()[0]

    # The last line might be empty
    i = 0
    while True:
        line = gt_lines[-i-1]
        parts = line.strip().split()
        if len(parts) == 8:
            break
        else:
            print(f"#{i+1} last line of groundtruth.txt is empty")
            i += 1

    num_frames = int(parts[0]) - int(first_frame) + 1

    # Test RGB images    
    rgb_count = sum(1 for entry in os.listdir(rgb_folder))

    rgb_sample = cv2.imread(
        os.path.join(rgb_folder, os.listdir(rgb_folder)[0]), cv2.IMREAD_UNCHANGED
    )
    rgb_bit_depth = rgb_sample.dtype.itemsize * 8

    if rgb_count != num_frames:
        print(
            f"❌ Error:  Some RGB images haven't been converted. RGB images:({rgb_count}) vs. RAW images: ({num_frames})"
        )
        print("Missing RGB images:")
        for i in range(int(first_frame), int(parts[0])+1):
            if not os.path.exists(os.path.join(rgb_folder, f"{i}.png")):
                print(f"{i}.png missing")
    else:
        if rgb_bit_depth != 16:
            print(f"❌ Error: RGB images bit depth ({rgb_bit_depth}) is not 16 bits.")
        else:
            print(f"✅ RGB images are correct and 16 bits.")

    # Test sRGB images
    srgb_count = sum(1 for entry in os.listdir(srgb_folder))

    srgb_sample = cv2.imread(
        os.path.join(srgb_folder, os.listdir(srgb_folder)[0]), cv2.IMREAD_UNCHANGED
    )
    srgb_bit_depth = srgb_sample.dtype.itemsize * 8

    if srgb_count != num_frames:
        print(
            f"❌ Error: Some sRGB images haven't been converted. sRGB images: ({srgb_count}) vs. RAW images: ({num_frames})"
        )
        print("Missing sRGB images:")
        for i in range(int(first_frame), int(parts[0])+1):
            if not os.path.exists(os.path.join(srgb_folder, f"{i}.png")):
                print(f"{i}.png missing")
    else:
        if srgb_bit_depth != 8:
            print(f"❌ Error: sRGB images bit depth ({srgb_bit_depth}) is not 8 bits.")
        else:
            print(f"✅ sRGB images are correct and 8 bits.")




def main():     
    main_folder = r"/home/morozco/datasets/my_dataset/kitchen2"

    raw_folder = os.path.join(main_folder, "raw")
    image_folder = os.path.join(main_folder, "rgb")
    srgb_folder = os.path.join(main_folder, "sRGB")
    metadata_folder = os.path.join(main_folder, "metadata")
    print(image_folder)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(srgb_folder, exist_ok=True)
    os.makedirs(metadata_folder, exist_ok=True)

    groundtruth = os.path.join(main_folder, "groundtruth.txt")

    # Loop over groundtruth frames only
    with open(groundtruth, "r") as f:
        gt_lines = f.readlines()
    gt_lines = gt_lines[1:] #skip first line with title

    print("🔴 Extracting RGB and sRGB images")
    for line in tqdm(gt_lines):
        parts = line.strip().split()
        if len(parts) != 8:
            continue # Skip empty row (last ones)
        frame = int(parts[0])

        raw_file = os.path.join(raw_folder, str(frame) + '.dng')
        rgb_file = RAW2RGB(frame, raw_file, image_folder)
        # RAW2sRGB(frame, raw_file, srgb_folder) #RealSense
        RGB2sRGB(frame, rgb_file, srgb_folder)

    
    print("🔎 Inspecting files")
    testing_record(gt_lines, image_folder, srgb_folder)


    
if __name__ == "__main__":
    main()
