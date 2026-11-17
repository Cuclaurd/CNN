import Images.External_images as ext_img
import Images.Images_input_CNN as img_input
import numpy as np
import CNN_func as cnn
import Images.CIFAR_10_images as cifar


def print_array_coefficients(array):
    """Prints every coefficient of a NumPy array."""
    print("Array Coefficients:")
    with np.printoptions(threshold=np.inf, linewidth=200):
        print(array)

def main():
    """image_path = "Images/data_test/ship10.png"

    image = ext_img.open_image(image_path)
    image = img_input.adaptation_image_centre(image)
    img_input.display_input_image(image)
    image = img_input.normalize_image(image)
"""

    batch_path = "cifar-10-batches-bin/data_batch_1.bin"
    CNN_coefficients_path = "CNN_coeff_3x3.txt"

    data_set, labels = cifar.load_batch(batch_path)
    label_names = cifar.load_label_names("cifar-10-batches-bin/batches.meta.txt")
    hit = 0

    for i in range(100):
        image = data_set[i]
        image = img_input.normalize_image(image)
        tensors = cnn.parse_cnn_coeffs(CNN_coefficients_path)
        matrice1 = cnn.conv2d_relu(image, tensors["conv1/weights"], tensors["conv1/biases"])
        matrice2 = cnn.maxpool(matrice1)
        matrice3 = cnn.conv2d_relu(matrice2, tensors["conv2/weights"], tensors["conv2/biases"])
        matrice4 = cnn.maxpool(matrice3)
        matrice5 = cnn.conv2d_relu(matrice4, tensors["conv3/weights"], tensors["conv3/biases"])
        matrice6 = cnn.maxpool(matrice5)
        matrice7 = cnn.reshape(matrice6)
        matrice8 = cnn.fully_connected(matrice7, tensors["local3/weights"], tensors["local3/biases"])
        matrice9 = cnn.softmax(matrice8)
        classe_predite = np.argmax(matrice9)
        if (classe_predite == labels[i]):
            hit += 1


    print(f"Nombre de bonnes prédictions : {hit} sur {100}")
    pt = hit / 100 * 100
    print(f"Pourcentage de bonnes prédictions : {pt:.2f}%")




if __name__ == "__main__":
    main()