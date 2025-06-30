import cv2
import numpy as np

""" Expand on this to accompany data preprocessing methods 
    needed to accommodate data passed from Unity"""

from typing import List, Dict

def string_to_image(byte_string):
    # Convert byte string to numpy array
    np_arr = np.frombuffer(byte_string, np.uint8)

    # Decode to image
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # Now 'image' is a NumPy array in BGR format


#scaling and normalization
def resize(img_path, target_size=(640, 640), bboxes=None, to_rgb=True):
    img = cv2.imread(img_path, 3)

    if img is None:
        raise ValueError(f"Failed to load image at {img_path}")

    print('original shape: ', img.shape)

    orig_h, orig_w = img.shape[:2]
    target_w, target_h = target_size

    # Calculate scaling factor to preserve aspect ratio
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)

    # Resize image
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create a blank canvas (black padding)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # Place resized image in the center
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = img_resized

    print("Resized shape:", canvas.shape)

    # Convert to RGB if required
    if to_rgb:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    canvas = canvas / 255.0

    # Scale bounding boxes if provided
    scaled_bboxes = []
    if bboxes is not None:
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            # Scale coordinates to new image size
            x_center = (x_center * orig_w * scale + x_offset) / target_w
            y_center = (y_center * orig_h * scale + y_offset) / target_h
            width = (width * orig_w * scale) / target_w
            height = (height * orig_h * scale) / target_h
            scaled_bboxes.append([class_id, x_center, y_center, width, height])

    return canvas, scaled_bboxes if bboxes else canvas