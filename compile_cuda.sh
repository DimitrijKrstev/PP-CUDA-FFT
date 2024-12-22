#!/bin/bash

# Find all CUDA files in the current directory
cuda_files=$(find . -name "*.cu")

for file in $cuda_files; do
  nvcc -c -o "${file%.*}.o" "$file"
  chmod 755 "${file%.*}.o"
done
