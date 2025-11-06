import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json 
from PIL import Image
from IPython.display import Markdown, clear_output, display, Video
import matplotlib.pyplot as plt
from transformers import AutoConfig, AutoModelForCausalLM
import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, AutoImageProcessor
from tqdm import tqdm
import pandas as pd
import re
import gc
import numpy as np

model_path = "DAMO-NLP-SG/VideoLLaMA3-7B"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # More memory efficient
)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
processor.image_processor.max_tokens = 65536 


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
            'confidence':converted_text[6]
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
    conversation = [
        {
            "role": "system",
            "content": sys_instruct
        },
        {        
            "role": "user",
            "content": [
                {
                    "type": "video", 
                    "video": {"video_path": vid_path, "fps": 1, 
                              "size": 324,"max_frames": 400,}
                },
                {
                    "type": "text", 
                    "text": qs
                },
            ]
        }
    ]

    #inputs = [prepare_inputs_for_vllm(message, processor) for message in [messages]]
    return conversation
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


with open('/home/haozong/projects/Qwen3-VL/instructions.json', 'r') as f:
    instructions = json.load(f)
sys_instruct1 = instructions['sys_instruct1']
sys_instruct2 = instructions['sys_instruct2']
sys_instruct3 = instructions['sys_instruct3']
qs = instructions['questions']




vid_dir = '/home/haozong/projects/Qwen3-VL/vids'
sessions = [os.path.join(vid_dir,i) for i in os.listdir(vid_dir)]
for session in sessions:
     if not os.path.exists(os.path.join(session, 'results_vidlamma.csv')):
        vids2score = [os.path.join(session,i) for i in os.listdir(session)]
        results1 = []
        pbar = tqdm(total=len(vids2score),leave=True,position=0)
        for vid in vids2score:
            is_valid, message = check_video_integrity(vid)
            if is_valid:
                conversation = generate_msg(vid,sys_instruct1,qs)
                # Single-turn conversation
                inputs = processor(conversation=conversation, return_tensors="pt")
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                
                output_ids = model.generate(**inputs, max_new_tokens=2048*2,top_p = 0.9,temperature=0.01)
                response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                tr = extract_trial_number(vid)
                results = generate_row_input(tr, response)
                results1.append(results)
                torch.cuda.empty_cache()
                gc.collect()
            pbar.update(1)
        pbar.close()
        df = pd.DataFrame(results1)
        df.to_csv(os.path.join(session,'results_vidlamma.csv'))
