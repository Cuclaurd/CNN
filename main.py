import Images.External_images as ext_img
import Images.Images_input_CNN as img_input
import numpy as np
import CNN_func as cnn


def print_array_coefficients(array):
    """Prints every coefficient of a NumPy array."""
    print("Array Coefficients:")
    with np.printoptions(threshold=np.inf, linewidth=200):
        print(array)

def main():
    image_path = "Images/data_test/horse.png"
    CNN_coefficients_path = "CNN_coeff_3x3.txt"

    image = ext_img.open_image(image_path)
    image = img_input.adaptation_image_centre(image)
    img_input.display_input_image(image)

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

    print_array_coefficients(matrice9)






if __name__ == "__main__":
    main()