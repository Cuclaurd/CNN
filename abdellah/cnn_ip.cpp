#include "cnn_ip.h"
#include "cnn_coeffs.h"


template <int IMG_SIZE, int IN_CH, int OUT_CH>
void conv2d_relu(
    const data_t input[IMG_SIZE][IMG_SIZE][IN_CH],
    const weight_t weights[3 * 3 * IN_CH * OUT_CH],
    const weight_t biases[OUT_CH],
    data_t output[IMG_SIZE][IMG_SIZE][OUT_CH]
) {
    for (int o = 0; o < OUT_CH; o++) { // first boucle sur outputs
        for (int k = 0; k < IMG_SIZE; k++) { 
// boucle sur k et l sur les lignes et colonnes
            for (int l = 0; l < IMG_SIZE; l++) {
                
                acc_t mac = biases[o];
                
		//boucle m,n dans le bloc 3x3
                for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                        
                        // decalage -1 car m=1 n=1 c'est le centre de carré 3x3
                        int in_r = k + m - 1;
                        int in_c = l + n - 1;
                        //les canaux normalement 3 r g b 
                        CONV_IN_CH: for (int i = 0; i < IN_CH; i++) {
                            // Zéro pour les pixels situés hors des bords.
                            data_t pixel_val = (data_t)0;
                            if (in_r >= 0 && in_r < IMG_SIZE && in_c >= 0 && in_c < IMG_SIZE) {
                                pixel_val = input[in_r][in_c][i];
                            }
                            
                            // Accès au poids aplati correspondant à weights[m][n][i][o]
                            int w_idx = ((m * 3 + n) * IN_CH + i) * OUT_CH + o;
                            mac += pixel_val * weights[w_idx];
                        }
                    }
                }
                // Activation ReLU : max(0, mac)
                output[k][l][o] = (mac > (acc_t)0) ? (data_t)mac : (data_t)0;
            }
        }
    }
}



template <int IMG_SIZE, int CHANNELS>
void maxpool(
    const data_t input[IMG_SIZE][IMG_SIZE][CHANNELS],
    data_t output[IMG_SIZE / 2][IMG_SIZE / 2][CHANNELS]
) {
    const int OUT_SIZE = IMG_SIZE / 2;

    for (int j = 0; j < OUT_SIZE; j++) { //j et k les cases de l'img
        for (int k = 0; k < OUT_SIZE; k++) {
            for (int i = 0; i < CHANNELS; i++) { // les canaux
                
                // vu Relu le min est 0
                data_t max_val = (data_t)0;
                
                for (int m = 0; m < 3; m++) {
                    for (int n = 0; n < 3; n++) {
                        int in_r = j * 2 + m;
                        int in_c = k * 2 + n;
                        
                        // verif des bordeer
                        data_t val = (data_t)0;
                        if (in_r < IMG_SIZE && in_c < IMG_SIZE) {
                            val = input[in_r][in_c][i];
                        }
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                output[j][k][i] = max_val;
            }
        }
    }
}

//reshape pour passer du cube à la sortie du 3eme maxpool 3x3x20 à 1D
void reshape(
    const data_t input[3][3][20],
    data_t output[180]
) {
    for (int i = 0; i < 3; i++) { //lignr
        for (int j = 0; j < 3; j++) { //colonne
            for (int c = 0; c < 20; c++) {
                output[c + j * 20 + i * 20 * 3] = input[i][j][c];
            }
        }
    }
}


void fully_connected(
    const data_t input[180],
    const weight_t weights[180 * 10],
    const weight_t biases[10],
    data_t output[10]
) {
    for (int j = 0; j < 10; j++) { //out
        acc_t mac = biases[j];
        for (int i = 0; i < 180; i++) {
            // Accès à weights[i][j] aplati
            mac += input[i] * weights[i * 10 + j];
        }
        output[j] = (data_t)mac;
    }
}

///////////////////////////////////////
void cnn_top(
    const data_t input_img[24][24][3],
    data_t output_logits[10]
) {
    
    data_t c1_out[24][24][64];
    data_t p1_out[12][12][64];
    
    data_t c2_out[12][12][32];
    data_t p2_out[6][6][32];
    
    data_t c3_out[6][6][20];
    data_t p3_out[3][3][20];
    
    data_t flat_out[180];

    
    conv2d_relu<24, 3, 64>(input_img, conv1_weights, conv1_biases, c1_out);
    maxpool<24, 64>(c1_out, p1_out);

    conv2d_relu<12, 64, 32>(p1_out, conv2_weights, conv2_biases, c2_out);
    maxpool<12, 32>(c2_out, p2_out);

    conv2d_relu<6, 32, 20>(p2_out, conv3_weights, conv3_biases, c3_out);
    maxpool<6, 20>(c3_out, p3_out);

    reshape(p3_out, flat_out);
    fully_connected(flat_out, fc_weights, fc_biases, output_logits);
}