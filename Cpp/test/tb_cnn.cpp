#include <cstdint>
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>

#include "cnn_ip.h"
#include "cnn_coeffs.h"


bool read_and_normalize_cifar(std::ifstream &file, int index, data_t img_norm[24][24][3], int &true_label) {
    const int RECORD_SIZE = 3073;
    uint8_t buffer[RECORD_SIZE];

    file.seekg(index * RECORD_SIZE, std::ios::beg);
    if (!file.read(reinterpret_cast<char*>(buffer), RECORD_SIZE)) {
        return false;
    }

    true_label = buffer[0];

    //normalisation et crop
    const int N = 24 * 24 * 3; // 1728
    float sum = 0.0f;
    float cropped[24][24][3];

    for (int r = 0; r < 24; r++) {
        for (int c = 0; c < 24; c++) {
            for (int ch = 0; ch < 3; ch++) {
                // Dans test_batch.bin : 1024 R, puis 1024 G, puis 1024 B (32x32)
                int src_r = r + 4;
                int src_c = c + 4;
                int byte_idx = 1 + (ch * 1024) + (src_r * 32) + src_c;
                
                float val = static_cast<float>(buffer[byte_idx]);
                cropped[r][c][ch] = val;
                sum += val;
            }
        }
    }
    float mu = sum / N;

    
    float var_sum = 0.0f;
    for (int r = 0; r < 24; r++) {
        for (int c = 0; c < 24; c++) {
            for (int ch = 0; ch < 3; ch++) {
                float diff = cropped[r][c][ch] - mu;
                var_sum += diff * diff;
            }
        }
    }
    float sigma = std::sqrt(var_sum / N);
    float min_sigma = 1.0f / std::sqrt(static_cast<float>(N));
    float denom = (sigma > min_sigma) ? sigma : min_sigma;

    
    for (int r = 0; r < 24; r++) {
        for (int c = 0; c < 24; c++) {
            for (int ch = 0; ch < 3; ch++) {
                img_norm[r][c][ch] = (data_t)((cropped[r][c][ch] - mu) / denom);
            }
        }
    }
    return true;
}

int main() {
    std::ifstream bin_file("../../cifar-10-batches-bin/test_batch.bin", std::ios::binary);
    if (!bin_file.is_open()) {
        std::cerr << "Erreur : impossible d'ouvrir test_batch.bin" << std::endl;
        return 1;
    }

    int nb_images = 1000; 
    int correct = 0;

    auto debut = std::chrono::steady_clock::now();


    for (int idx = 0; idx < nb_images; idx++) {
        data_t img_in[24][24][3];
        data_t logits_out[10];
        int true_label = 0;

        read_and_normalize_cifar(bin_file, idx, img_in, true_label);

        
        cnn_top(img_in, logits_out);

        
        int pred_label = 0;
        data_t max_score = logits_out[0];
        for (int c = 1; c < 10; c++) {
            if (logits_out[c] > max_score) {
                max_score = logits_out[c];
                pred_label = c;
            }
        }

        if (pred_label == true_label) {
            correct++;
        }

        std::cout << "Image #" << idx << " | Vrai: " << true_label 
                  << " | Predit: " << pred_label 
                  << " | Taux: " << (100.0f * correct / (idx + 1)) << "%" << std::endl;
    }

    auto fin = std::chrono::steady_clock::now();

    auto duree_ms = std::chrono::duration_cast<std::chrono::milliseconds>(fin - debut);
    auto duree_us = std::chrono::duration_cast<std::chrono::microseconds>(fin - debut);

    std::cout << "Temps : " << duree_ms.count() << " ms (" << duree_us.count() << " µs)\n";

    std::cout << "\n= TAUX DE DETECTION FINAL : " << (100.0f * correct / nb_images) << "% ===" << std::endl;
    return 0;
}