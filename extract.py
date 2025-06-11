import os
import cv2
import rawpy
import imageio
from tqdm import tqdm
import numpy as np

from Record.utils import load_raw_image

def gamma_encode_srgb(x):
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * x ** (1 / 2.4) - a)

def green2sRGB(img_demosaicked):
    img_demosaicked = cv2.cvtColor(img_demosaicked, cv2.COLOR_BGR2RGB)
    # Step 1: Normalize
    if img_demosaicked.dtype == np.uint8:
        img = img_demosaicked.astype(np.float32) / 255.0
    elif img_demosaicked.dtype == np.uint16:
        img = img_demosaicked.astype(np.float32) / 65535.0

    # Step 2: Apply white balance (generic daylight)
    wb = np.array([2.0, 1.0, 1.5])
    img_wb = img * wb  

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

    return img_srgb


def process_RAW(frame, input_file, srgb_folder, raw_folder, green, width=1920, height=1080, bayer_pattern=cv2.COLOR_BAYER_GR2BGR):
    "Convert .dng RAW to green RGB RAW"

    # 1. Process .dng RAW image
    raw_image = load_raw_image(input_file, width, height)
    bgr_image = cv2.cvtColor(raw_image, bayer_pattern)

    # 2. Demosaic RAW image
    green_raw_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # If green, save green RAW image, if it doesn't already exist
    if green:
        output_raw_path = os.path.join(raw_folder, str(frame) + '.png')
        if not os.path.isfile(output_raw_path): 
            cv2.imwrite(output_raw_path, green_raw_image)

    # 3. Convert green RAW image to sRGB RAW image
    srgb_raw_image = green2sRGB(green_raw_image)

    # If not green, save 16 bits sRGB RAW image, if it doesn't already exist
    if not green:
        output_raw_path = os.path.join(raw_folder, str(frame) + '.png')
        if not os.path.isfile(output_raw_path): 
            img_to_save = (srgb_raw_image * 65535).astype(np.uint16)
            img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_raw_path, img_to_save)
    
    # Save 8 bits SRGB LDR image, if it doesn't already exist
    output_ldr_path = os.path.join(srgb_folder, str(frame) + '.png')
    # if not os.path.isfile(output_ldr_path):
    img_to_save = (srgb_raw_image * 255).astype(np.uint8)
    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_ldr_path, img_to_save)
        
def testing_record(gt_lines, raw_folder, srgb_folder):
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

    # Test RAW images    
    raw_count = sum(1 for entry in os.listdir(raw_folder))

    raw_sample = cv2.imread(
        os.path.join(raw_folder, os.listdir(raw_folder)[0]), cv2.IMREAD_UNCHANGED
    )
    rgb_bit_depth = raw_sample.dtype.itemsize * 8

    if raw_count != num_frames:
        print(
            f"❌ Error:  Some RAW images haven't been converted. RAW images:({raw_count}) vs. recorded images in gt.txt: ({num_frames})"
        )
        print("Missing RGB images:")
        for i in range(int(first_frame), int(parts[0])+1):
            if not os.path.exists(os.path.join(raw_folder, f"{i}.png")):
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
    # Select dataset   
    main_folder = r"/home/morozco/datasets/my_dataset/kitchen2"
    green = False # True for green RAW, False for color sRGB RAW

    # Create and load directories 
    dng_folder = os.path.join(main_folder, "raw")
    print(dng_folder)
    
    srgb_folder = os.path.join(main_folder, "sRGB")
    if green:
        raw_folder = os.path.join(main_folder, "raw_green")
        print("🟢 Green RAW")
    else:
        raw_folder = os.path.join(main_folder, "raw_sRGB")
        print("🌈 Non green RAW")
    metadata_folder = os.path.join(main_folder, "metadata") # Not in use rightnow
    
    os.makedirs(raw_folder, exist_ok=True)
    os.makedirs(srgb_folder, exist_ok=True)
    os.makedirs(metadata_folder, exist_ok=True)

    groundtruth = os.path.join(main_folder, "groundtruth.txt")

    # Loop over groundtruth frames to extract images
    with open(groundtruth, "r") as f:
        gt_lines = f.readlines()
    gt_lines = gt_lines[1:] 

    print("🔴 Extracting RAW and LDR images")
    for line in tqdm(gt_lines):
        parts = line.strip().split()
        if len(parts) != 8:
            continue # Skip empty row (last ones)
        frame = int(parts[0])

        raw_file = os.path.join(dng_folder, str(frame) + '.dng')

        process_RAW(frame, raw_file, srgb_folder, raw_folder, green)

    
    print("🔎 Inspecting files")
    testing_record(gt_lines, raw_folder, srgb_folder) 


    
if __name__ == "__main__":
    main()
