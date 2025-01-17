#include "fft.h"

// Reduced constant memory size for small FFTs
__constant__ cuDoubleComplex d_twiddle_small[2048];  // 16KB constant memory

__device__ cuDoubleComplex compute_twiddle(int j, int m, uint32_t N) {
    double angle = -2.0 * M_PI * j * (N / m) / N;
    return make_cuDoubleComplex(cos(angle), sin(angle));
}

template<int BLOCK_SIZE>
__global__ void fft_kernel_optimized(cuDoubleComplex* Y, uint32_t N, int logN) {
    __shared__ cuDoubleComplex shared_mem[BLOCK_SIZE];
    
    // Coalesced global memory access using block-wise bit reversal
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int i = bid * BLOCK_SIZE + tid;
    
    // Load data with stride-1 access pattern
    if (i < N) {
        uint32_t rev = __brev(i) >> (32 - logN);
        shared_mem[tid] = Y[rev];
    }
    __syncthreads();
    
    // Perform butterfly operations within shared memory
    #pragma unroll
    for (int s = 1; s <= min(logN, (int)log2f(BLOCK_SIZE)); s++) {
        int mh = 1 << (s - 1);
        int m = mh << 1;
        
        if (tid < BLOCK_SIZE/2) {
            int pos = (tid >> (s-1)) * m + (tid & (mh-1));
            cuDoubleComplex twiddle = compute_twiddle(tid & (mh-1), m, N);
            
            cuDoubleComplex a = shared_mem[pos];
            cuDoubleComplex b = cuCmul(twiddle, shared_mem[pos + mh]);
            
            shared_mem[pos] = cuCadd(a, b);
            shared_mem[pos + mh] = cuCsub(a, b);
        }
        __syncthreads();
    }
    
    // Store results with coalesced writes
    if (i < N) {
        Y[i] = shared_mem[tid];
    }
}

template<int BLOCK_SIZE>
__global__ void fft_kernel_large_stage(cuDoubleComplex* Y, uint32_t N, int stage) {
    const int group_size = 1 << stage;
    const int half_group = group_size >> 1;
    
    int gid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // Process multiple elements per thread
    #pragma unroll 4
    for (int k = gid; k < (N/2); k += gridDim.x * BLOCK_SIZE) {
        int group = k / half_group;
        int offset = k % half_group;
        
        int i1 = group * group_size + offset;
        int i2 = i1 + half_group;
        
        cuDoubleComplex twiddle = compute_twiddle(offset, group_size, N);
        cuDoubleComplex a = Y[i1];
        cuDoubleComplex b = cuCmul(twiddle, Y[i2]);
        
        Y[i1] = cuCadd(a, b);
        Y[i2] = cuCsub(a, b);
    }
}

int fft_gpu(const cuDoubleComplex* x, cuDoubleComplex* Y, uint32_t N) {
    if (N & (N - 1)) return -1;
    
    int logN = (int)log2f((float)N);
    cuDoubleComplex *Y_dev;
    cudaMalloc(&Y_dev, sizeof(cuDoubleComplex) * N);
    cudaMemcpy(Y_dev, x, sizeof(cuDoubleComplex) * N, cudaMemcpyHostToDevice);
    
    const int BLOCK_SIZE = 256;
    
    if (N <= 1024) {
        // Small FFT: single kernel approach
        dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
        fft_kernel_optimized<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(Y_dev, N, logN);
    } else {
        // Large FFT: multi-stage approach
        dim3 grid((N/2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
        
        // Initial butterfly stages within blocks
        fft_kernel_optimized<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(Y_dev, N, min(10, logN));
        
        // Remaining stages with global memory
        for (int stage = min(10, logN) + 1; stage <= logN; stage++) {
            fft_kernel_large_stage<BLOCK_SIZE><<<grid, BLOCK_SIZE>>>(Y_dev, N, stage);
        }
    }
    
    cudaMemcpy(Y, Y_dev, sizeof(cuDoubleComplex) * N, cudaMemcpyDeviceToHost);
    cudaFree(Y_dev);
    return EXIT_SUCCESS;
}

int main(int argc, char* argv[]){
  if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_file> <N>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char* input_file = argv[1];
    uint32_t N = (uint32_t)atoi(argv[2]);

    cuDoubleComplex* input = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));
    cuDoubleComplex* gpu_output = (cuDoubleComplex*)malloc(N * sizeof(cuDoubleComplex));

    if(read_file_input(input_file, N, input) != EXIT_SUCCESS){
        free(input);
        return EXIT_FAILURE;
    }

    if (fft_gpu(input, gpu_output, N) != EXIT_SUCCESS) {
        free(input);
        return EXIT_FAILURE;
    }

    print_complex_array("Parallelised FFT output", gpu_output, N);

    free(input);
    return EXIT_SUCCESS;
}