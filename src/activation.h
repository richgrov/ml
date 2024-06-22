#ifndef ACTIVATION_H_
#define ACTIVATION_H_

#include <math.h>

static inline double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

static inline double sigmoid_derivative(double x) {
    double sig_x = sigmoid(x);
    return sig_x * (1 - sig_x);
}

#endif // !ACTIVATION_H_
