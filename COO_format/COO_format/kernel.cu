#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>

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
            int index = atomicAdd(nnz, 1);
            cooValues[index] = val;
            cooRowInd[index] = row;
            cooColInd[index] = col;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " input_csv output_csv" << endl;
        return 1;
    }

    string inputCsvFile = argv[1];
    string outputCsvFile = argv[2];

    // Read the dense matrix from the input CSV file
    vector<float> denseMatrix;
    int numRows = 0;
    int numCols = 0;

    ifstream inputFile(inputCsvFile);
    string line;
    while (getline(inputFile, line)) {
        stringstream ss(line);
        float value;
        while (ss >> value) {
            denseMatrix.push_back(value);
        }
        numRows++;
        if (numCols == 0) {
            numCols = denseMatrix.size();
        }
    }
    inputFile.close();

    for (float i : denseMatrix) {
        cout << denseMatrix[i];
    }

    // Create CUDA device memory pointers
    int nnz = 0;
    int* d_nnz;
    float* d_denseMatrix;
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

    // Write COO format to the output CSV file
    ofstream outputFile(outputCsvFile);
    for (int i = 0; i < nnz; i++) {
        outputFile << cooRowInd[i] << "," << cooColInd[i] << "," << cooValues[i] << endl;
    }
    outputFile.close();

    return 0;
}
