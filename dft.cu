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

int main() {
  cuDoubleComplex output[testN];
  dft(testInput, output, testN);

  print_complex_array("Standard dft output", output, testN);
}
