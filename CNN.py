import os
import re

import numpy as np
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CNN_COEFFS_PATH = os.path.join(BASE_DIR, "CNN_coeff_3x3.txt")
TEST_BATCH_PATH = os.path.join(BASE_DIR, "cifar-10-batches-bin", "test_batch.bin")

Filter_size = 3  # Taille du filtre pour les convolutions (3x3)

EXPECTED_SHAPES = {
    "conv1/biases": (64,),
    "conv1/weights": (3, 3, 3, 64),
    "conv2/biases": (32,),
    "conv2/weights": (3, 3, 64, 32),
    "conv3/biases": (20,),
    "conv3/weights": (3, 3, 32, 20),
    "local3/biases": (10,),
    "local3/weights": (180, 10),
}

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def preprocess_png_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Fichier introuvable : {image_path}")

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if img.size != (32, 32):
            img = img.resize((32, 32), Image.Resampling.NEAREST)
        img_hwc = np.array(img, dtype=np.float32)

    img_cropped = img_hwc[4:28, 4:28, :]
    norm = np.sqrt(img_cropped.size)
    mu = np.mean(img_cropped)
    sigma = np.std(img_cropped)
    return (img_cropped - mu) / max(sigma, 1.0 / norm)


def parse_cnn_coeffs(filepath):
    # Lecture du fichier contenant tous les poids et biais du réseau.
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Un tenseur est séparé de chaque suivant par la chaîne "tensor_name:".
    tensors = {}
    for block in full_text.split("tensor_name:")[1:]:
        lines = block.strip().splitlines()

        # La première ligne du bloc contient le nom du tenseur.
        tensor_name = lines[0].strip()

        # Les lignes suivantes contiennent ses valeurs numériques.
        numbers_text = " ".join(lines[1:])
        float_strings = re.findall(
            r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?",
            numbers_text,
        )

        # Conversion des nombres en tableau NumPy de nombres flottants 32 bits.
        flat_array = np.array([float(value) for value in float_strings], dtype=np.float32)

        # Redimensionnement du tableau selon la forme attendue du tenseur.
        if tensor_name in EXPECTED_SHAPES:
            tensors[tensor_name] = flat_array.reshape(EXPECTED_SHAPES[tensor_name])
        else:
            tensors[tensor_name] = flat_array

    # Association finale : nom du tenseur -> tableau NumPy.
    return tensors


def conv2d_relu(input_fmap, weights, biases):
    hauteur = input_fmap.shape[0]
    largeur = input_fmap.shape[1]
    nombre_canaux = input_fmap.shape[2]
    nombre_filtres = weights.shape[3]

    # Ajoute une bordure de zéros pour ne réduire ni la hauteur ni la largeur.
    padded = np.pad(
        input_fmap,
        ((1, 1), (1, 1), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )

    sortie = np.zeros(
        (hauteur, largeur, nombre_filtres),
        dtype=np.float32,
    )

    # Parcourt chaque pixel de l'image.
    for ligne in range(hauteur):
        for colonne in range(largeur):
            # Calcul des filtres.
            for filtre in range(nombre_filtres):
                somme = biases[filtre]

                # Parcourt les trois lignes et colonnes du filtre 3 x 3.
                for y in range(Filter_size):
                    for x in range(Filter_size):
                        # Additionne les trois canaux (RGB ou couches précédentes).
                        for canal in range(nombre_canaux):
                            pixel = padded[ligne + y, colonne + x, canal]
                            somme += pixel * weights[y, x, canal, filtre]

                # ReLU : remplace les valeurs négatives par zéro.
                sortie[ligne, colonne, filtre] = max(0.0, somme)

    return sortie


def maxpool(input_fmap, pool_size=3, stride_size=2):
    img_size, _, channels = input_fmap.shape
    out_size = img_size // stride_size
    padded = np.pad(
        input_fmap,
        ((0, 1), (0, 1), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )
    output = np.zeros((out_size, out_size, channels), dtype=np.float32)

    for j in range(0, img_size, stride_size):
        for k in range(0, img_size, stride_size):
            patch = padded[j:j + pool_size, k:k + pool_size, :]
            output[j // stride_size, k // stride_size, :] = np.max(
                patch,
                axis=(0, 1),
            )

    return output


def reshape(input_fmap):
    img_size, _, channels = input_fmap.shape
    output = np.zeros(img_size * img_size * channels, dtype=np.float32)

    for i in range(img_size):
        for j in range(img_size):
            for c in range(channels):
                output[c + j * channels + i * img_size * channels] = input_fmap[i, j, c]

    return output


def fully_connected(input_vec, weights, biases):
    return np.dot(input_vec, weights) + biases


def softmax(x):
    shifted_x = x - np.max(x)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x)


def forward_cnn(img_norm, coeffs):
    c1 = conv2d_relu(
        img_norm,
        coeffs["conv1/weights"],
        coeffs["conv1/biases"],
    )
    p1 = maxpool(c1)

    c2 = conv2d_relu(
        p1,
        coeffs["conv2/weights"],
        coeffs["conv2/biases"],
    )
    p2 = maxpool(c2)

    c3 = conv2d_relu(
        p2,
        coeffs["conv3/weights"],
        coeffs["conv3/biases"],
    )
    p3 = maxpool(c3)

    flat = reshape(p3)
    logits = fully_connected(
        flat,
        coeffs["local3/weights"],
        coeffs["local3/biases"],
    )
    probs = softmax(logits)
    predicted_label = int(np.argmax(probs))
    return predicted_label, probs


def extract_cifar_image(bin_filepath, image_index):
    record_size = 3073

    with open(bin_filepath, "rb") as f:
        f.seek(image_index * record_size)
        raw = f.read(record_size)

    if len(raw) != record_size:
        raise IndexError("Index d'image hors du fichier CIFAR-10.")

    true_label = int(raw[0])
    pixels = np.frombuffer(raw[1:], dtype=np.uint8).astype(np.float32)
    img_hwc = pixels.reshape((3, 32, 32)).transpose((1, 2, 0))
    img_cropped = img_hwc[4:28, 4:28, :]

    mu = np.mean(img_cropped)
    sigma = np.std(img_cropped)
    norm = np.sqrt(img_cropped.size)
    img_norm = (img_cropped - mu) / max(sigma, 1.0 / norm)

    return img_norm, true_label


def predict_png_image(image_path, coeffs):
    img_norm = preprocess_png_image(image_path)
    predicted_label, probs = forward_cnn(img_norm, coeffs)
    confidence = probs[predicted_label] * 100

    print(f"Image : {os.path.basename(image_path)}")
    print(f"Classe prédite : {CIFAR10_CLASSES[predicted_label]} ({predicted_label})")
    print(f"Confiance : {confidence:.1f}%")


def run_cifar_batch(bin_filepath, coeffs, nb_images=100, start_index=0):
    if not os.path.exists(bin_filepath):
        raise FileNotFoundError(f"Fichier batch introuvable : {bin_filepath}")

    correct = 0

    for idx in range(start_index, start_index + nb_images):
        img_norm, true_label = extract_cifar_image(bin_filepath, idx)
        predicted_label, probs = forward_cnn(img_norm, coeffs)
        if predicted_label == true_label:
            correct += 1

        status = "OK" if predicted_label == true_label else "ERREUR"
        print(
            f"Image #{idx:04d} | "
            f"Attendu: {CIFAR10_CLASSES[true_label]:10s} | "
            f"Prédit: {CIFAR10_CLASSES[predicted_label]:10s} "
            f"({probs[predicted_label] * 100:4.1f}%) | "
            f"{status}"
        )

    accuracy = correct / nb_images * 100
    print(f"Résultat : {correct}/{nb_images} reconnues")
    print(f"Précision : {accuracy:.2f}%")
    return accuracy


if __name__ == "__main__":
    coeffs = parse_cnn_coeffs(CNN_COEFFS_PATH)
    run_cifar_batch(TEST_BATCH_PATH, coeffs)
