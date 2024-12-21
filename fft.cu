#include "fft.h"

__device__ uint32_t reverse_bits_gpu(uint32_t x, int logN) {
    return __brev(x) >> (32 - logN);
}

__host__ void precompute_twiddle_factors(cuDoubleComplex* twiddle, uint32_t N) {
    for (uint32_t j = 0; j < N / 2; j++) {
        double angle = -2.0 * M_PI * j / N;
        twiddle[j] = make_cuDoubleComplex(cos(angle), sin(angle));
    }
}

// FFT kernel
__global__ void fft_kernel(cuDoubleComplex* Y, cuDoubleComplex* twiddle, uint32_t N, int logN) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N / 2) return;

    uint32_t rev = reverse_bits_gpu(i, logN);
    cuDoubleComplex temp = Y[i];
    Y[i] = Y[rev];
    Y[rev] = temp;

    __syncthreads();

    for (int s = 1; s <= logN; s++) {
        int mh = 1 << (s - 1);
        int m = mh << 1;

        int group = i / mh;
        int group_start = group * m;
        int j = i % mh;

        cuDoubleComplex a = Y[group_start + j];
        cuDoubleComplex b = cuCmul(twiddle[j * (N / m)], Y[group_start + j + mh]);

        Y[group_start + j] = cuCadd(a, b);
        Y[group_start + j + mh] = cuCsub(a, b);

        __syncthreads();
    }
}

int fft_gpu(const cuDoubleComplex* x, cuDoubleComplex* Y, uint32_t N) {
    if (N & (N - 1)) {
        fprintf(stderr, "N=%u must be a power of 2.\n", N);
        return -1;
    }

    int logN = (int)log2f((float)N);

    cuDoubleComplex* x_dev;
    cuDoubleComplex* Y_dev;
    cuDoubleComplex* twiddle_dev;

    cudaMalloc(&x_dev, sizeof(cuDoubleComplex) * N);
    cudaMalloc(&Y_dev, sizeof(cuDoubleComplex) * N);
    cudaMalloc(&twiddle_dev, sizeof(cuDoubleComplex) * (N / 2));

    cudaMemcpy(x_dev, x, sizeof(cuDoubleComplex) * N, cudaMemcpyHostToDevice);

    cuDoubleComplex* twiddle = (cuDoubleComplex*)malloc(sizeof(cuDoubleComplex) * (N / 2));
    precompute_twiddle_factors(twiddle, N);
    cudaMemcpy(twiddle_dev, twiddle, sizeof(cuDoubleComplex) * (N / 2), cudaMemcpyHostToDevice);
    free(twiddle);

    int block_size = 256;
    int grid_size = (N / 2 + block_size - 1) / block_size;

    cudaMemcpy(Y_dev, x_dev, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToDevice);
    fft_kernel<<<grid_size, block_size>>>(Y_dev, twiddle_dev, N, logN);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    cudaMemcpy(Y, Y_dev, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost);

    cudaFree(x_dev);
    cudaFree(Y_dev);
    cudaFree(twiddle_dev);

    return EXIT_SUCCESS;
}

int main() {
    cuDoubleComplex* gpu_output = (cuDoubleComplex*)malloc(testN * sizeof(cuDoubleComplex));

    printf("Running Parallel FFT...\n");
    fft_gpu(testInput, gpu_output, testN);
    print_complex_array("Parallel FFT Output", gpu_output, testN);

    free(gpu_output);

    return 0;
}
