from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
import qwen_vl_utils
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Literal
from vllm.sampling_params import GuidedDecodingParams
import re
import os
import numpy as np
from tqdm import tqdm
import gc
import json
from time import sleep
import cv2
import re
import pandas as pd
import cv2
import os
import pandas as pd
from pathlib import Path
import numpy as np



#for full documentation, please refer to https://github.com/QwenLM/Qwen3-VL/

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto",attn_implementation="flash_attention_2"
)



processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")



def convert_text(text_str):
    # Split on pattern like "1.", "2.", etc. at the start of a line
    bullet_points = re.split(r'\n(?=\d+\.)',text_str.strip())

    # Clean up each bullet point
    bullet_points = [bp.strip()[2:] for bp in bullet_points if bp.strip()]
    return bullet_points

def to_float(value):
    """
    Convert a string like ' 0%' or '45 %' or '3.5' to float.
    Returns None if conversion fails.
    """
    if value is None:
        return None
    
    # Convert to string (handles numbers and None)
    s = str(value).strip()
    
    # Remove percent sign if present
    s = s.replace('%', '').strip()
    
    try:
        return float(s)
    except ValueError:
        return value  # or return float('nan') if using pandas

def classify_outcome(text):
    # Normalize to lowercase for case-insensitive match
    text = text.lower()

    if "partial success" in text:
        return "ps"
    elif "success" in text:
        return "s"
    elif "fail" in text:
        return "f"
    else:
        return 'ucn'  # or return "" if you prefer empty


def resample_video_simple(video_path: str, output_path: str, scale_factor: float = 0.5) -> dict:
    """
    Simple function to downsample and upsample a video with quality metrics.
    
    Args:
        video_path: Path to input video
        output_path: Path to save output
        scale_factor: Scale for downsampling (0.5 = half size, 0.25 = quarter size)
        
    Returns:
        Dictionary with processing information and quality metrics
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate new dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Create output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    
    # Lists to store metrics from sampled frames
    mse_values = []
    psnr_values = []
    frame_num = 0
    
    # Sample every Nth frame for metrics (to avoid processing every frame)
    sample_interval = max(1, total_frames // 54)  # Sample ~100 frames
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Downsample
        down = cv2.resize(frame, (new_width, new_height), 
                         interpolation=cv2.INTER_AREA)
        
        # Upsample back to original size
        up = cv2.resize(down, (width, height), 
                       interpolation=cv2.INTER_CUBIC)
        
        out.write(up)
        
        # Calculate metrics for sampled frames
        if frame_num % sample_interval == 0:
            metrics = compare_frames(frame, up)
            mse_values.append(metrics['mse'])
            if metrics['psnr'] != float('inf'):
                psnr_values.append(metrics['psnr'])
        
        frame_num += 1
    
    cap.release()
    out.release()
    
    # Calculate average metrics
    avg_mse = np.mean(mse_values) if mse_values else 0
    avg_psnr = np.mean(psnr_values) if psnr_values else float('inf')
    
    
    return {
        'avg_mse': avg_mse,
        'min_mse': min(mse_values) if mse_values else 0,
        'max_mse': max(mse_values) if mse_values else 0,
        'avg_psnr': avg_psnr,
        'min_psnr': min(psnr_values) if psnr_values else float('inf'),
        'max_psnr': max(psnr_values) if psnr_values else float('inf'),
    }



def compare_frames(original: np.ndarray, resampled: np.ndarray) -> dict:
    """
    Compare original and resampled frames.
    
    Args:
        original: Original frame
        resampled: Resampled frame
        
    Returns:
        Dictionary with comparison metrics
    """
    # Calculate MSE
    mse = np.mean((original.astype(float) - resampled.astype(float)) ** 2)
    
    # Calculate PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return {
        'mse': mse,
        'psnr': psnr
    }

def generate_row_input(trial, generated_text):
    try:
        converted_text =convert_text(generated_text)
        perval = to_float(converted_text[3])
        trial_data = {
            'trial':trial,
            'tongue_contact':converted_text[0],
            'water_drop_stable':converted_text[1],
            'water_spilled': converted_text[2],
            'percentage_consumed': perval,
            'outcome':classify_outcome(converted_text[4]),
            'outcome_classification':converted_text[4],
            'raw_answers':generated_text,
            'confidence':converted_text[6],
        }
    except Exception as e:
        trial_data = {
            'trial':trial,
            'tongue_contact':np.nan,
            'water_drop_stable':np.nan,
            'water_spilled': np.nan,
            'percentage_consumed': np.nan,
            'outcome':np.nan,
            'outcome_classification':np.nan,
            'raw_answers':generated_text,
            'confidence':np.nan
        }
    return trial_data
def generate_msg(vid_path,sys_instruct,qs):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": sys_instruct}
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": vid_path,
                    "max_pixels": 4096 * 28 * 28,  # ← VERY HIGH resolution
                    "total_pixels": 259000 * 28 * 28,  # ← Allocate ~92% of context to video
                    "fps":1
                },            
                {"type": "text", "text": qs},
            ],
        },

    ]
    #inputs = [prepare_inputs_for_vllm(message, processor) for message in [messages]]
    return messages
def check_video_integrity(video_path: str, thorough_check: bool = True) -> (bool, str):
    """
    Checks the integrity of a video file using a multi-step process.

    This function verifies:
    1. File existence and non-zero size.
    2. That OpenCV can open the file container.
    3. That the video reports a valid frame count and FPS.
    4. (Optional) That at least the first frame can be successfully decoded.

    Args:
        video_path (str): The full path to the video file.
        thorough_check (bool): If True, attempts to read the first frame to ensure
                               the video stream is decodable. Defaults to True.

    Returns:
        tuple[bool, str]: A tuple containing:
            - A boolean indicating if the video is valid (True) or not (False).
            - A string message describing the result or the error found.
    """
    # --- Check 1: Basic file system checks (Fastest) ---
    if not os.path.exists(video_path):
        return False, f"Error: File does not exist at path: {video_path}"

    if os.path.getsize(video_path) == 0:
        return False, "Error: File is empty (0 bytes)."

    cap = None
    try:
        # --- Check 2: Attempt to open the video file with OpenCV ---
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False, "Error: OpenCV could not open the video file. It may be corrupted or in an unsupported format."

        # --- Check 3: Check for valid metadata ---
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        if frame_count <= 0 or fps <= 0:
            return False, f"Error: Video metadata is invalid (Frames: {frame_count}, FPS: {fps})."

        # --- Check 4: Thorough check - attempt to read a frame (Most reliable) ---
        if thorough_check:
            success, frame = cap.read()
            if not success:
                return False, "Error: Failed to decode the first frame. The video stream is likely corrupted."
            # You could add an extra check here to see if the frame is not just a black screen
            # if frame is None or frame.sum() == 0:
            #     return False, "Warning: The first frame is empty or completely black."

    except Exception as e:
        # Catch any other unexpected errors during processing
        return False, f"An unexpected error occurred: {e}"
    finally:
        # --- Cleanup: Always release the capture object ---
        if cap is not None:
            cap.release()

    return True, "Video integrity check passed."

def extract_trial_number(path):
    filename = os.path.basename(path)
    match = re.search(r'trial(\d+)', filename, re.IGNORECASE)
    return int(match.group(1)) if match else None


def read_dlc_csv(dlc_path):
    """
    Read DLC CSV file and return cleaned dataframe.
    
    Returns:
        DataFrame with multi-level columns (bodypart, coordinate)
    """
    df = pd.read_csv(dlc_path, header=[1, 2], index_col=0)
    return df

def get_bodypart_groups():
    """
    Define the three body part groups.
    
    Returns:
        Dictionary with group names and their body parts
    """
    groups = {
        'right_paw': ['r1_tip_A', 'r1_mid_A', 'r1_joint_A', 'r_mid_A',
                      'r2_tip_A', 'r2_mid_A', 'r2_joint_A',
                      'r3_tip_A', 'r3_mid_A', 'r3_joint_A',
                      'r4_tip_A', 'r4_mid_A', 'r4_joint_A'],
        'left_paw': ['l1_tip_A', 'l1_mid_A', 'l1_joint_A', 'l_mid_A',
                     'l2_tip_A', 'l2_mid_A', 'l2_joint_A',
                     'l3_tip_A', 'l3_mid_A', 'l3_joint_A',
                     'l4_tip_A', 'l4_mid_A', 'l4_joint_A'],
        'snout': ['snout_A', 'mouth_tip_A']
    }
    return groups

def get_group_center(df, frame_idx, bodyparts, likelihood_threshold=0.5):
    """
    Calculate the center of a group of body parts for a specific frame.
    
    Args:
        df: DLC dataframe
        frame_idx: Frame index
        bodyparts: List of body part names
        likelihood_threshold: Minimum likelihood to include point
    
    Returns:
        (center_x, center_y) or None if no valid points
    """
    valid_x = []
    valid_y = []
    
    for bodypart in bodyparts:
        try:
            x = df.loc[frame_idx, (bodypart, 'x')]
            y = df.loc[frame_idx, (bodypart, 'y')]
            likelihood = df.loc[frame_idx, (bodypart, 'likelihood')]
            
            if likelihood >= likelihood_threshold and not (np.isnan(x) or np.isnan(y)):
                valid_x.append(x)
                valid_y.append(y)
        except KeyError:
            continue
    
    if valid_x and valid_y:
        center_x = int(np.mean(valid_x))
        center_y = int(np.mean(valid_y))
        return (center_x, center_y)
    else:
        return None

def blur_region(frame, center, blur_size=51, blur_sigma=0):
    """
    Apply Gaussian blur to a circular region around center.
    
    Args:
        frame: Input image frame
        center: (x, y) tuple for center of blur
        blur_size: Size of blur kernel (must be odd)
        blur_sigma: Gaussian kernel standard deviation (0 = auto)
    
    Returns:
        Frame with blurred region
    """
    if center is None:
        return frame
    
    # Ensure blur_size is odd
    if blur_size % 2 == 0:
        blur_size += 1
    
    x, y = center
    radius = blur_size // 2
    
    # Create a copy of the frame
    output = frame.copy()
    
    # Define the region to blur
    y_min = max(0, y - radius)
    y_max = min(frame.shape[0], y + radius + 1)
    x_min = max(0, x - radius)
    x_max = min(frame.shape[1], x + radius + 1)
    
    # Extract and blur the region
    region = frame[y_min:y_max, x_min:x_max].copy()
    blurred_region = cv2.GaussianBlur(region, (blur_size, blur_size), blur_sigma)
    
    # Create circular mask
    mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)
    mask_center = (radius, radius) if (x - radius >= 0 and y - radius >= 0) else (x - x_min, y - y_min)
    cv2.circle(mask, mask_center, radius, 1, -1)
    
    # Apply mask to blend blurred region
    for c in range(frame.shape[2]):
        output[y_min:y_max, x_min:x_max, c] = (
            mask * blurred_region[:, :, c] + 
            (1 - mask) * region[:, :, c]
        ).astype(np.uint8)
    
    return output

def blur_bodypart_groups(frame, dlc_df, frame_idx, groups_to_blur=['right_paw'], 
                         blur_size=51, likelihood_threshold=0.5):
    """
    Blur specific body part groups in a frame.
    
    Args:
        frame: Input video frame (numpy array)
        dlc_df: DLC dataframe
        frame_idx: Current frame index
        groups_to_blur: List of groups to blur ('right_paw', 'left_paw', 'snout')
        blur_size: Size of Gaussian blur kernel
        likelihood_threshold: Minimum likelihood for valid keypoints
    
    Returns:
        Frame with blurred regions
    """
    all_groups = get_bodypart_groups()
    output_frame = frame.copy()
    
    for group_name in groups_to_blur:
        if group_name not in all_groups:
            print(f"Warning: Unknown group '{group_name}', skipping")
            continue
        
        bodyparts = all_groups[group_name]
        center = get_group_center(dlc_df, frame_idx, bodyparts, likelihood_threshold)
        
        if center is not None:
            output_frame = blur_region(output_frame, center, blur_size)
    
    return output_frame

# ===== USAGE EXAMPLE =====
def get_dlc_path(video_path):
    """
    Given a video path, return the corresponding DLC CSV file path.
    
    Args:
        video_path: Path to video file (e.g., 'trial93.mp4')
    
    Returns:
        Path to DLC CSV file, or None if not found
    """
    video_path = Path(video_path)
    video_stem = video_path.stem  # Gets 'trial93' from 'trial93.mp4'
    video_dir = video_path.parent
    
    # Look for files that start with video name and end with .csv
    pattern = f"{video_stem}DLC*.csv"
    dlc_files = list(video_dir.glob(pattern))
    
    if dlc_files:
        return dlc_files[0]  # Return first match
    else:
        return None

def process_video_with_blur(video_path, output_path,groups_to_blur=['right_paw'], blur_size=51):
    """
    Process entire video and blur specified body part groups.
    
    Args:
        video_path: Path to video file
        groups_to_blur: List of groups to blur
        blur_size: Size of blur kernel
    """
    video_path = Path(video_path)
    
    # Get DLC file
    dlc_path = get_dlc_path(video_path)
    if dlc_path is None:
        raise FileNotFoundError(f"No DLC file found for {video_path}")
    
    # Read DLC data
    print(f"Reading DLC file: {dlc_path.name}")
    dlc_df = read_dlc_csv(dlc_path)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Setup output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Blur specified groups
        blurred_frame = blur_bodypart_groups(
            frame, dlc_df, frame_idx, 
            groups_to_blur=groups_to_blur, 
            blur_size=blur_size
        )
        
        out.write(blurred_frame)
        frame_idx += 1

    
    cap.release()
    out.release()
    return







with open('instructions.json', 'r') as f:
    instructions = json.load(f)
sys_instruct1 = instructions['sys_instruct1']
sys_instruct2 = instructions['sys_instruct2']
sys_instruct3 = instructions['sys_instruct3']
qs = instructions['questions']

# path to video folders
videos = ['AZ_R2_2024-12-14_1','K_R2_2025-02-04_1','FJ_R2_2024-07-15_1','FU_R2_2024-07-18_1','K_R2_2025-01-14_1','FJ_R2_2024-06-29_1',
          'FJ_L2_2024-07-15_1','FJ_L3_2024-07-29_1','FJ_R3_2024-07-29_1'] 

save_name = 'results_8b.csv'
vid_dir = './vids'
sessions = [os.path.join(vid_dir,i) for i in os.listdir(vid_dir)]#if i in videos]
for session in sessions:
     if not os.path.exists(os.path.join(session, save_name)):
        print(session)
        vids2score = [os.path.join(session,i) for i in os.listdir(session) if i[-4:]=='.mp4']
        results1 = []
        pbar = tqdm(total=len(vids2score),leave=True,position=0)
        for vid in vids2score:
            is_valid, message = check_video_integrity(vid)
            if is_valid:
                #metrics = resample_video_simple(vid,'temp.mp4', 0.5) 
                #process_video_with_blur(vid, 'temp.mp4', groups_to_blur=['right_paw'], blur_size=140)
                messages =  generate_msg(vid,sys_instruct1,qs)
            # Preparation for inference
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(model.device)

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=1024,top_p = 0.9,temperature=0.01)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                tr = extract_trial_number(vid)
                results = generate_row_input(tr, output_text[0])
                results1.append(results)
                del inputs
                del generated_ids
                del generated_ids_trimmed
                del output_text
                del messages
                torch.cuda.empty_cache()
                gc.collect()

            pbar.update(1)
        pbar.close()
        df = pd.DataFrame(results1)
        df.to_csv(os.path.join(session,save_name))



































'''
save_name = 'results_30b_resize_10.csv'
vid_dir = './vids'
sessions = [os.path.join(vid_dir,i) for i in os.listdir(vid_dir) if i in videos]
for session in sessions:
     if not os.path.exists(os.path.join(session, save_name)):
        print(session)
        vids2score = [os.path.join(session,i) for i in os.listdir(session)]
        results1 = []
        results2 = []
        pbar = tqdm(total=len(vids2score),leave=True,position=0)
        for vid in vids2score:
            is_valid, message = check_video_integrity(vid)
            if is_valid:
                metrics = resample_video_simple(vid,'temp.mp4', 0.1) 
                messages =  generate_msg('temp.mp4',sys_instruct1,qs)
            # Preparation for inference
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(model.device)

                # Inference: Generation of the output
                generated_ids = model.generate(**inputs, max_new_tokens=1024,top_p = 0.9,temperature=0.01)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                tr = extract_trial_number(vid)
                results = generate_row_input(tr, output_text[0],metrics)
                results1.append(results)
                del inputs
                del generated_ids
                del generated_ids_trimmed
                del output_text
                del messages
                torch.cuda.empty_cache()
                gc.collect()

            pbar.update(1)
        pbar.close()
        df = pd.DataFrame(results1)
        df.to_csv(os.path.join(session,save_name))
'''

