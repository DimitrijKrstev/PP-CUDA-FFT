#!/bin/bash

# Find all CUDA files in the current directory
cuda_files=$(find . -name "*.cu")

for file in $cuda_files; do
  # Extract the base filename without extension
  base_name="${file%.*}"
  
  # Compile the CUDA file into an executable
  nvcc -o "$base_name" "$file"

  # Set executable permissions
  chmod 755 "$base_name"
done
