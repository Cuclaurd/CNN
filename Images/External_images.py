from pathlib import Path

import numpy as np
from PIL import Image


def open_image(image_path):
    """
    Opens an image file and returns it as a NumPy array.

    Parameters:
        image_path (str | Path): The path to the image file.

    Returns:
        np.ndarray: The image in HWC format: (height, width, 3).
    """
    with Image.open(image_path) as image:
        image_array = np.array(image.convert("RGB"), dtype=np.float32)

    return image_array
