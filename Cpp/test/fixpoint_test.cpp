#include <cmath>
#include <iostream>

#include "../include/fixpoint.h"
#include "../include/cnn_ip.h"

using namespace std;
using namespace cnn;

int main() {
    Fixed16 a(1.5);
    Fixed16 b(0.25);

    Fixed16 sum = a + b;
    Fixed16 difference = a - b;
    Fixed16 product = a * b;
    Fixed16 quotient = a / b;

    cout << "sum=" << sum.to_double() << '\n';
    cout << "difference=" << difference.to_double() << '\n';
    cout << "product=" << product.to_double() << '\n';
    cout << "quotient=" << quotient.to_double() << '\n';
    
    Fixed16 large(30000.0);
    Fixed16 overflow = large + large;
    cout << "overflow=" << overflow.to_double() << '\n';

    return
        (abs(sum.to_double() - 1.75) < 0.001 &&
         abs(difference.to_double() - 1.25) < 0.001 &&
         abs(product.to_double() - 0.375) < 0.001 &&
         abs(overflow.to_double() - 60000.0) < 0.001 &&
         abs(quotient.to_double() - 6.0) < 0.001)
        ? 0
        : 1;
}
