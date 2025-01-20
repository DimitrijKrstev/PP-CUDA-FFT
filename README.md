# CUDA FFT and DFT Implementation

## About
This project provides a CUDA-parallelized implementation of the Fast Fourier Transform (FFT) Radix-2 algorithm and compares its execution time with the Discrete Fourier Transform (DFT) and a non-parallelized Radix-2 implementation. It is designed for high-performance computation of Fourier transforms, utilizing GPU acceleration.

The paper related to this project can be found [here](https://drive.google.com/file/d/1JXMN3On8vcQ2VDLQPCDVSC3c-wX8gjDS/view?usp=sharing)
## Files
- **`compile_cuda.sh`**: Shell script to compile the CUDA source files.
- **`dft.cu`**: Implementation of the Discrete Fourier Transform (DFT) on the GPU.
- **`fft-iterative.cu`**: An iterative CPU implementation of the FFT Radix-2 algorithm.
- **`fft.cu`**: CUDA-parallelized implementation of the FFT Radix-2 algorithm with optimizations.
- **`fft.h`**: Header file containing shared definitions and utility functions for the FFT implementation.
- **`fft_benchmark.ipynb`**: Jupyter notebook to benchmark the DFT and FFT implementations. It can be run on Google Colab for easy access and clones the entire repository for execution.

## Algorithms Explained

### Discrete Fourier Transform (DFT)
The DFT computes the frequency spectrum of a signal by directly applying the Fourier Transform formula:

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j\frac{2\pi kn}{N}}$$

- **Complexity**: $O(N^2)$, due to the nested loops for computing each frequency component.

### Fast Fourier Transform (FFT)
The FFT is an optimized algorithm for computing the DFT by leveraging the symmetry and periodicity properties of the Fourier Transform. The Radix-2 algorithm divides the input data into even and odd indices, recursively applying the FFT:

$$X[k] = X_\text{even}[k] + W_N^k \cdot X_\text{odd}[k]$$

- **Complexity**: $O(N \log N)$, making it significantly faster than the DFT for large datasets.

### CUDA FFT Implementation (`fft.cu`)
- **Features**:
  - **Optimized Memory Usage**: Uses shared memory and CUDA constant memory for efficient data access.
  - **Twiddle Factors**: Precomputes and reuses complex exponential values stored in constant memory.
  - **Hybrid Stages**: Combines shared memory operations for small FFT sizes and global memory for larger sizes.
  - **Block-wise Parallelism**: Divides the computation across CUDA threads for maximum parallel efficiency.
- **Key Functions**:
  - **`fft_kernel_optimized`**: Performs FFT using shared memory for small datasets.
  - **`fft_kernel_large_stage`**: Handles larger datasets using global memory and multi-stage processing.
  - **`fft_gpu`**: Orchestrates the entire FFT process, including data transfers and kernel launches.

## Getting Started

### Compilation

Run the following command to compile the CUDA files:
```bash
./compile_cuda.sh
```
### Usage

Run the compiled binary with:

```bash
./fft <input_file> <N>
```

- `<input_file>`: Path to the file containing the input signal.
- `<N>`: The size of the input, which must be a power of 2.

### Benchmarking

To benchmark the implementations, open the Jupyter notebook `fft_benchmark.ipynb` in Google Colab:

1. Upload the notebook to Colab.
2. Run the first cell to clone the repository.
3. Follow the steps to execute the benchmarks.

## License

This project is released under the MIT License. See `LICENSE` for details.
