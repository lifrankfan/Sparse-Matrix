#include <iostream>
#include <vector>

// CUDA headers
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// CUDA kernel to convert a dense matrix to COO format
__global__ void denseToCOO(const float* denseMatrix, int numRows, int numCols, int* cooRowInd, int* cooColInd, float* cooValues, int* nnz) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    for (int i = tid; i < numRows * numCols; i += totalThreads) {
        int row = i / numCols;
        int col = i % numCols;
        float val = denseMatrix[i];

        if (val != 0.0) {
            /* 
            atomicAdd ensures that multiple threads won't update the same shared variable simultaneously
            in this case, it ensures nnz is syncronized
            nnz + 1
            https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions 
            */
            int index = atomicAdd(nnz, 1);
            cooValues[index] = val;
            cooRowInd[index] = row;
            cooColInd[index] = col;
        }
    }
}

int main() {
    // Define matrix dimensions
    int numRows = 4;
    int numCols = 3;

    // Create a dense matrix as a flat array
    // CUDA doesn't seem to support 2D arrays
    vector<float> denseMatrix = {
        1.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 3.5,
        0.0, 0.0, 4.0
    };

    // non-zero values and device non-zero values
    int nnz = 0;
    int* d_nnz;

    // device dense memory pointer
    float* d_denseMatrix;

    // device sparse memory pointers
    int* d_cooRowInd;
    int* d_cooColInd;
    float* d_cooValues;

    // Allocate GPU memory
    cudaMalloc((void**)&d_denseMatrix, numRows * numCols * sizeof(float));
    cudaMalloc((void**)&d_cooRowInd, numRows * numCols * sizeof(int));
    cudaMalloc((void**)&d_cooColInd, numRows * numCols * sizeof(int));
    cudaMalloc((void**)&d_cooValues, numRows * numCols * sizeof(float));
    cudaMalloc((void**)&d_nnz, sizeof(int));

    // Copy data from CPU to GPU
    cudaMemcpy(d_denseMatrix, denseMatrix.data(), numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nnz, &nnz, sizeof(int), cudaMemcpyHostToDevice);

    // Define CUDA grid and block sizes
    int blockSize = 256;
    int gridSize = (numRows * numCols + blockSize - 1) / blockSize;

    // Call the CUDA kernel to convert dense to COO format
    denseToCOO << <gridSize, blockSize >> > (d_denseMatrix, numRows, numCols, d_cooRowInd, d_cooColInd, d_cooValues, d_nnz);

    // Copy the result back from GPU to CPU
    cudaMemcpy(&nnz, d_nnz, sizeof(int), cudaMemcpyDeviceToHost);

    vector<int> cooRowInd(nnz);
    vector<int> cooColInd(nnz);
    vector<float> cooValues(nnz);

    cudaMemcpy(cooRowInd.data(), d_cooRowInd, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cooColInd.data(), d_cooColInd, nnz * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cooValues.data(), d_cooValues, nnz * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_denseMatrix);
    cudaFree(d_cooRowInd);
    cudaFree(d_cooColInd);
    cudaFree(d_cooValues);
    cudaFree(d_nnz);

    // Print COO format
    cout << "COO Format:" << endl;
    for (int i = 0; i < nnz; i++) {
        cout << "Row: " << cooRowInd[i] << ", Col: " << cooColInd[i] << ", Value: " << cooValues[i] << endl;
    }

    return 0;
}
