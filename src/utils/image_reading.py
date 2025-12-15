import numpy as np
import cv2
from datetime import datetime

def process_image(image_path: str) -> tuple[np.ndarray, datetime]:
    """
    Load and preprocess an image from the given path.
    
    Args:
        image_path (str): Path to the image file.
    Returns:
        tuple[np.ndarray, datetime]: Preprocessed image array and the timestamp extracted from the filename.
    """
    img = cv2.imread(image_path)
    file_name = image_path.split("/")[-1].split(".")[0]
    time_frame = datetime.strptime(file_name, "%Y%m%d-%H%M%S")

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), time_frame


