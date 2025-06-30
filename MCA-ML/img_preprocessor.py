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
    return image


#scaling and normalization
def preprocess(img_path):
    img = cv2.imread(img_path, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img