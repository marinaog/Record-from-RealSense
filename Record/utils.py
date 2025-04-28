import os
import cv2
import numpy as np
import rawpy
import imageio
from tqdm import tqdm
import math
import yaml

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
        "Dataset": {
            "Calibration": {
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
        },

        "Results": {
            "save_results": True,
            "save_dir": "results",
            "save_trj": True,
            "save_trj_kf_intv": 5,
            "use_gui": True,
            "eval_rendering": False,
            "use_wandb": False
        },

        "Training": {
            "init_itr_num": 1050,
            "init_gaussian_update": 100,
            "init_gaussian_reset": 500,
            "init_gaussian_th": 0.005,
            "init_gaussian_extent": 30,
            "tracking_itr_num": 100,
            "mapping_itr_num": 150,
            "gaussian_update_every": 150,
            "gaussian_update_offset": 50,
            "gaussian_th": 0.7,
            "gaussian_extent": 1.0,
            "gaussian_reset": 2001,
            "size_threshold": 20,
            "kf_interval": 5,
            "window_size": 8,
            "pose_window": 3,
            "edge_threshold": 1.1,
            "rgb_boundary_threshold": 0.01,
            "alpha": 0.9,
            "kf_translation": 0.08,
            "kf_min_translation": 0.05,
            "kf_overlap": 0.9,
            "kf_cutoff": 0.3,
            "prune_mode": "slam",
            "single_thread": False,
            "spherical_harmonics": False,
            "lr": {
                "cam_rot_delta": 0.003,
                "cam_trans_delta": 0.001
            }
        },

        "opt_params": {
            "iterations": 30000,
            "position_lr_init": 0.00016,
            "position_lr_final": 0.0000016,
            "position_lr_delay_mult": 0.01,
            "position_lr_max_steps": 30000,
            "feature_lr": 0.0025,
            "opacity_lr": 0.05,
            "scaling_lr": 0.001,
            "rotation_lr": 0.001,
            "percent_dense": 0.01,
            "lambda_dssim": 0.2,
            "densification_interval": 100,
            "opacity_reset_interval": 3000,
            "densify_from_iter": 500,
            "densify_until_iter": 15000,
            "densify_grad_threshold": 0.0002
        },

        "model_params": {
            "sh_degree": 0,
            "source_path": "",
            "model_path": "",
            "resolution": -1,
            "white_background": False,
            "data_device": "cuda"
        },

        "pipeline_params": {
            "convert_SHs_python": False,
            "compute_cov3D_python": False
        }
    }   
    
    file_path = os.path.join(base_folder, "my_dataset.yaml")

    with open(file_path, 'w') as file:
        yaml.dump(camera_info, file, default_flow_style=False)
