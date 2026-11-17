import numpy as np
import re

EXPECTED_SHAPES = {
    "conv1/biases":  (64,),
    "conv1/weights": (3, 3, 3, 64),
    "conv2/biases":  (32,),
    "conv2/weights": (3, 3, 64, 32),
    "conv3/biases":  (20,),
    "conv3/weights": (3, 3, 32, 20),
    "local3/biases": (10,),
    "local3/weights": (180, 10)
}

def parse_cnn_coeffs(filepath):
    with open(filepath, 'r') as f:
        full_text = f.read()

    #On decoupe par tensor name
    raw_blocks = full_text.split("tensor_name:")

    tensors = {}


    for block in raw_blocks[1:]:
        lines = block.strip().splitlines()

        #première ligne contient le nom (conv1/viases
        tensor_name = lines[0].strip()

        #le reste : chiffres
        numbers_text = " ".join(lines[1:])

        # on ignore les crochets
        float_strings = re.findall(r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)', numbers_text)

        # Conversion en tableau NumPy 1D de flottants
        flat_array = np.array([float(x) for x in float_strings], dtype=np.float32)

        # Remise à la bonne dimension 4D, 2D ou 1D (Reshape)
        if tensor_name in EXPECTED_SHAPES:
            target_shape = EXPECTED_SHAPES[tensor_name]
            tensors[tensor_name] = flat_array.reshape(target_shape)
        else:

            tensors[tensor_name] = flat_array
            print(f"Attention : '{tensor_name}' non listé dans EXPECTED_SHAPES (taille 1D = {flat_array.size})")

    return tensors

def conv2d_relu(input_fmap, weights, biases):
    """
    input_fmap : tableau 3D (img_size, img_size, in_channels)  ex: (24, 24, 3)
    weights    : tableau 4D (kernel_size, kernel_size, in_channels, out_channels) ex: (3, 3, 3, 64)
    biases     : tableau 1D (out_channels,) ex: (64,)
    """
    img_size, _, in_channels = input_fmap.shape
    kernel_size, _, _, out_channels = weights.shape

    # Zero-padding : on ajoute 1 ligne/colonne de 0 en haut bas gauche droite

    padded_input = np.pad(input_fmap, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0.0)

    # on definie le tableau de sortie de même hauteur/largeur
    output = np.zeros((img_size, img_size, out_channels), dtype=np.float32)

    # Parcours des pixels (k, l) et des canaux de sortie (o)

    for o in range(out_channels):
        for k in range(img_size):
            for l in range(img_size):

                mac = biases[o]
                for m in range(kernel_size):
                    for n in range(kernel_size):
                        for i in range(in_channels):
                            mac += weights[m, n, i, o] * padded_input[k + m, l + n, i]
                # On applique le Relu et on remplace dans l'output

                output[k, l, o] = max(0.0, mac)

    return output

def maxpool(input_fmap, pool_size=3, stride_size=2):
    """
    input_fmap : tableau 3D (img_size, img_size, channels)
    Sortie     : tableau 3D (img_size // 2, img_size // 2, channels)
    """
    img_size, _, channels = input_fmap.shape
    out_size = img_size // stride_size

    # Padding ajouté de 1 ligne en bas et 1 colonne à droite pour que la fenêtre 3x3 ne déborde pas à j=22

    padded_input = np.pad(input_fmap, ((0, 1), (0, 1), (0, 0)), mode='constant', constant_values=0.0)
    output = np.zeros((out_size, out_size, channels), dtype=np.float32)

    for j in range(0, img_size, stride_size):
        for k in range(0, img_size, stride_size):
            for i in range(channels):

                v = -np.inf
                for m in range(pool_size):
                    for n in range(pool_size):
                        v = max(v, padded_input[j + m, k + n, i])
                output[j // stride_size, k // stride_size, i] = v

    return output


def reshape(input_fmap):
    """
    Transforme le cube (3, 3, 20) en vecteur 1D (180,) dans l'ordre HWC
    """

    img_size, _, channels = input_fmap.shape
    output = np.zeros(img_size * img_size * channels, dtype=np.float32)


    for i in range(img_size):
        for j in range(img_size):
            for c in range(channels):
                output[c + j * channels + i * channels * img_size] = input_fmap[i, j, c]

    return output



def fully_connected(input_vec, weights, biases):
    """
    input_vec : (180,)
    weights   : (180, 10)
    biases    : (10,)
    """
    inputs, outputs = weights.shape
    output = np.zeros(outputs, dtype=np.float32)

    for j in range(outputs):
        mac = biases[j]
        for i in range(inputs):
            mac += input_vec[i] * weights[i, j]
        output[j] = mac

    return output

def softmax(x):
    """Applies the softmax function to a 1D array."""
    shifted_x = x - np.max(x)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x)