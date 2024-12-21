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
