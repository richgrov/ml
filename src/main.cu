#include <stdio.h>

__global__ void hello() {
    printf("Hello cuda\n");
}

int main() {
    hello<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 1;
}
