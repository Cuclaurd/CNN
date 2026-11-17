from pathlib import Path

import numpy as np


def load_batch(batch_path):
    """
    Loads one binary CIFAR-10 batch and reconstructs its RGB images.

    Parameters:
    batch_path (str | Path): The path to a data_batch_*.bin or test_batch.bin.

    Returns:
    tuple[np.ndarray, np.ndarray]: Images with shape (n, 32, 32, 3) and labels.
    """
    record_size = 1 + 32 * 32 * 3 #1 byte for label + 3072 bytes for image
    expected_size = 10000 * record_size #size of one batch 
    raw_data = Path(batch_path).read_bytes()

    records = np.frombuffer(raw_data, dtype=np.uint8).reshape(10000, record_size)
    labels = records[:, 0].copy()
    data = records[:, 1:]
    images = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def load_label_names(meta_path):
    """Loads CIFAR-10 class names from batches.meta.txt."""
    label_names = [
        line.strip()
        for line in Path(meta_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(label_names) != 10:
        raise ValueError(
            f"Le fichier des labels doit contenir 10 noms, trouvé: {len(label_names)}."
        )
    return label_names

def print_array_coefficients(array):
    """Prints every coefficient of a NumPy array."""
    print("Array Coefficients:")
    with np.printoptions(threshold=np.inf, linewidth=200):
        print(array)


def print_image(image):
    """Prints an image represented as a NumPy array."""
    print("Image:")
    with np.printoptions(threshold=np.inf, linewidth=200):
        print(image)




def main():
    dataset_path = Path("cifar-10-batches-bin")
    batch_path = dataset_path / "data_batch_1.bin"
    meta_path = dataset_path / "batches.meta.txt"


    images, labels = load_batch(batch_path)
    label_names = load_label_names(meta_path)

    image_index = 0
    image = images[image_index]
    label = int(labels[image_index])

    print(f"Nombre d'images dans le batch : {len(images)}")
    print(f"Image sélectionnée : {image_index}")
    print(f"Label : {label} ({label_names[label]})")
    print(f"Forme de l'image : {image.shape}")
    print_image(image)

if __name__ == "__main__":
    main()