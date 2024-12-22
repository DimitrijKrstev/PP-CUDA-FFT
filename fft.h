#ifndef FFT_H
#define FFT_H

#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <cuComplex.h>

void print_complex_array(const char* name, cuDoubleComplex* arr, int N) {
    printf("%s:\n", name);
    for (int i = 0; i < N; i++) {
        printf("[%d]: %.3f + %.3fi\n", i, cuCreal(arr[i]), cuCimag(arr[i]));
    }
    printf("\n");
}

int read_file_input(const char* input_file, uint32_t N, cuDoubleComplex* input){
    if (N & (N - 1)) {
        fprintf(stderr, "N=%u must be a power of 2.\n", N);
        return EXIT_FAILURE;
    }

    FILE* fp = fopen(input_file, "r");
    if (!fp) {
        perror("Error opening input file");
        return EXIT_FAILURE;
    }

    for (uint32_t i = 0; i < N; i++) {
        double real, imag;
        if (fscanf(fp, "%lf %lf", &real, &imag) != 2) {
            fprintf(stderr, "Error reading input data at index %u\n", i);
            fclose(fp);
            free(input);
            return EXIT_FAILURE;
        }
        input[i] = make_cuDoubleComplex(real, imag);
    }
    fclose(fp);

    return EXIT_SUCCESS;
}

cuDoubleComplex testInput[] = {
  make_cuDoubleComplex(3.6f, 2.6f),
  make_cuDoubleComplex(2.9f, 6.3f),
  make_cuDoubleComplex(5.6f, 4.0f),
  make_cuDoubleComplex(4.8f, 9.1f),
  make_cuDoubleComplex(3.3f, 0.4f),
  make_cuDoubleComplex(5.9f, 4.8f),
  make_cuDoubleComplex(5.0f, 2.6f),
  make_cuDoubleComplex(4.3f, 4.1f)
};

const int testN = 8;

#endif
