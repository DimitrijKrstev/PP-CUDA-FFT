
#include "fft.h"

uint32_t reverse_bits(uint32_t x) {
    x = ((x & 0xAAAAAAAA) >> 1) | ((x & 0x55555555) << 1);
    x = ((x & 0xCCCCCCCC) >> 2) | ((x & 0x33333333) << 2);
    x = ((x & 0xF0F0F0F0) >> 4) | ((x & 0x0F0F0F0F) << 4);
    x = ((x & 0xFF00FF00) >> 8) | ((x & 0x00FF00FF) << 8);
    return (x >> 16) | (x << 16);
}

int fft(const cuDoubleComplex* x, cuDoubleComplex* Y, uint32_t N) {
    int logN = (int)log2f((float)N);

    for (uint32_t i = 0; i < N; i++) {
        uint32_t rev = reverse_bits(i) >> (32 - logN);
        Y[i] = x[rev];
    }

    for (int s = 1; s <= logN; s++) {
        int m = 1 << s;      // Current transform size
        int mh = m >> 1;     // Half of the current transform size

        // Precompute twiddle factor for this stage
        float angle = -2.0f * M_PI / m;
        cuDoubleComplex twiddle = make_cuDoubleComplex(cosf(angle), sinf(angle));

        for (uint32_t k = 0; k < N; k += m) {
            cuDoubleComplex twiddle_factor = make_cuDoubleComplex(1.0f, 0.0f);

            for (int j = 0; j < mh; j++) {
                cuDoubleComplex a = Y[k + j];
                cuDoubleComplex b = cuCmul(twiddle_factor, Y[k + j + mh]);

                // Update Y[k + j] and Y[k + j + mh]
                Y[k + j] = cuCadd(a, b);
                Y[k + j + mh] = cuCsub(a, b);

                // Update twiddle factor
                twiddle_factor = cuCmul(twiddle_factor, twiddle);
            }
        }
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
    cuDoubleComplex* output = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));

    if(read_file_input(input_file, N, input) != EXIT_SUCCESS){
        free(input);
        return EXIT_FAILURE;
    }

    if (fft(input, output, N) != EXIT_SUCCESS) {
        free(input);
        return EXIT_FAILURE;
    }

    print_complex_array("FFT-iterative output", output, N);

    free(input);
    return EXIT_SUCCESS;
}

