from pathlib import Path

import numpy as np


def adaptation_image_centre(image):
    """
    Adapts the image to center it in a 28x28 array.
    """
    new_image = image.copy()
    new_image[3:28, 3:28] = image[:25, :25]
    return new_image


def display_input_image(image):
    """Displays an RGB image and prints every pixel as [R, G, B]."""
    print("Format : RGB")
    print(f"Dimensions du tableau : {image.shape}")
    print(f"Type des valeurs : {image.dtype}")

    with np.printoptions(threshold=np.inf, linewidth=200):
        print("\nPixels RGB [R, G, B] :")
        print(image)
