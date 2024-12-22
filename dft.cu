#include "fft.h"

int dft(const cuDoubleComplex* x, cuDoubleComplex* Y, uint32_t N) {
    for (size_t k = 0; k < N; k++) {
        cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
        double c = -2.0 * M_PI * k;

        for (size_t n = 0; n < N; n++) {
            double a = (c * n) / N;
            double sin_a, cos_a;

            sincos(a, &sin_a, &cos_a);
            cuDoubleComplex twiddle = make_cuDoubleComplex(cos_a, -sin_a);
            sum = cuCadd(sum, cuCmul(x[n], twiddle));
        }
        Y[k] = sum;
    }
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <N>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* input_file = argv[1];
    uint32_t N = (uint32_t)atoi(argv[2]);

    cuDoubleComplex* input = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    cuDoubleComplex output[N];

    if(read_file_input(input_file, N, input) != EXIT_SUCCESS){
        free(input);
        return EXIT_FAILURE;
    }

    if (dft(input, output, N) != EXIT_SUCCESS) {
        free(input);
        return EXIT_FAILURE;
    }

    print_complex_array("Standard dft output", output, N);

    free(input);
    return EXIT_SUCCESS;
}
