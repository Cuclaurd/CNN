from pathlib import Path

import numpy as np


def adaptation_image_centre(image):
    """
    Adapts the image to center it in a 28x28 array.
    """
    new_image = image.copy()
    new_image[3:28, 3:28] = image[:25, :25]
    return new_image

def normalize_image(image):
    """Normalizes each RGB channel independently."""
    img_cropped = np.asarray(image, dtype=np.float32)
    height, width, channels = img_cropped.shape
    N = height * width

    img_norm = np.empty_like(img_cropped)
    for channel in range(channels):
        mu = np.mean(img_cropped[:, :, channel])
        sigma = np.std(img_cropped[:, :, channel])
        img_norm[:, :, channel] = (
            img_cropped[:, :, channel] - mu
        ) / max(sigma, 1.0 / np.sqrt(N))

    return img_norm



def display_input_image(image):
    """Displays an RGB image and prints every pixel as [R, G, B]."""
    print("Format : RGB")
    print(f"Dimensions du tableau : {image.shape}")
    print(f"Type des valeurs : {image.dtype}")

    with np.printoptions(threshold=np.inf, linewidth=200):
        print("\nPixels RGB [R, G, B] :")
        print(image)
