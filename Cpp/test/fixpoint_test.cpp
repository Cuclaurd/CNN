#include <cmath>
#include <iostream>

#include "../include/fixpoint.h"

using namespace std;
using namespace cnn;

int main() {
    Fixed16 a(1.5);
    Fixed16 b(0.25);

    Fixed16 sum = a + b;
    Fixed16 difference = a - b;
    Fixed16 product = a * b;
    Fixed16 quotient = a / b;

    std::cout << "sum=" << sum.to_double() << '\n';
    std::cout << "difference=" << difference.to_double() << '\n';
    std::cout << "product=" << product.to_double() << '\n';
    std::cout << "quotient=" << quotient.to_double() << '\n';

    cnn::Fixed16 large(30000.0);
    cnn::Fixed16 overflow = large + large;
    std::cout << "overflow=" << overflow.to_double() << '\n';

    return
        (std::abs(sum.to_double() - 1.75) < 0.001 &&
         std::abs(difference.to_double() - 1.25) < 0.001 &&
         std::abs(product.to_double() - 0.375) < 0.001 &&
         std::abs(quotient.to_double() - 6.0) < 0.001)
        ? 0
        : 1;
}
