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
        image_array = np.array(image.convert("RGB"))

    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(
            f"Format d'image inattendu : {image_array.shape}. "
            "Le format attendu est HWC : (hauteur, largeur, 3)."
        )

    return image_array
