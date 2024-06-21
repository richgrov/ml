#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "activation.h"
#include "linear.h"
#include "rand.h"

#define ARRAY_LEN(x) (sizeof(x) / sizeof(x[0]))

typedef struct {
    double input[2];
    double output;
} Datapoint;

typedef struct {
    Linear linear;
} Model;

void model_init(Model *model) {
    linear_init(&model->linear, 2, 1);
}

void model_randinit(Model *model) {
    linear_randinit(&model->linear, 0.0, 1.0);
    model->linear.biases[0] = 0.0;
}

double model_forward(Model *model, Datapoint *data) {
    linear_forward(&model->linear, data->input);
    return model->linear.outputs[0];
}

void model_backward(Model *model, double cost_gradient, double *inputs, double lr) {
    linear_backward(&model->linear, inputs, &cost_gradient, lr);
}

void model_deinit(Model *model) {
    linear_deinit(&model->linear);
}

int main(int argc, char **argv) {
    srand(time(NULL));

    Datapoint dataset[100];
    for (int i = 0; i < ARRAY_LEN(dataset); ++i) {
        Datapoint e;
        e.input[0] = rand_double(0.0, 1.0);
        e.input[1] = rand_double(0.0, 1.0);
        e.output = (e.input[0] + e.input[1]) / 2;
        dataset[i] = e;
    }

    Model model;
    model_init(&model);
    model_randinit(&model);

    for (int i = 0; i < ARRAY_LEN(dataset); ++i) {
        Datapoint *entry = &dataset[i];
        double out = model_forward(&model, entry);

        double cost = mse(out, entry->output);
        double cost_gradient = mse_derivative(out, entry->output);

        model_backward(&model, cost_gradient, entry->input, 0.1);
        printf("avg(%f, %f) is about %f (cost %f)\n", entry->input[0], entry->input[1], out, cost);
    }

    model_deinit(&model);
}
