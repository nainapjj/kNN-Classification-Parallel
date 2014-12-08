#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <ctime>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define MAX 10
#define MIN 0

#define FILE_NAME "small12345.txt"
#define K 2

using namespace std;

__global__ void normalize(double * d_input, double *d_max, double *d_min, unsigned int numAttributes, 
    unsigned int numElems) {
    
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int attributeIdx = tid % numAttributes;
    
    if(tid < numElems*numAttributes) {
        d_input[tid] = (d_input[tid] - d_min[attributeIdx]) / (d_max[attributeIdx] - d_min[attributeIdx]);
    }
}

__global__ void findDistance(double *d_inputNormal, double *d_inputSample,  double *d_output, unsigned int numElems) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid < numElems) {
        d_output[tid] = (d_inputNormal[tid] - d_inputSample[tid])*(d_inputNormal[tid] - d_inputSample[tid]);
    }

}

/*__global__ void block_sum(double *input, double *results, size_t n)
{
    extern __shared__ float sdata[];
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int tx = threadIdx.x;
    // load input into __shared__ memory
    float x = 0;
    if(i < n) {
        x = input[i];
    }
    sdata[tx] = x; 
    
    for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
    {
      if(tx < offset)
      {
        // add a partial sum upstream to our own
        sdata[tx] += sdata[tx + offset];
      }
      __syncthreads();
    } 
    
    if(tx == 0) {
        results[blockIdx.x] = 
    }
}*/

void parse(int* numAttributes, int* numKnownSamples, int* numClass, int *numUnknowns,
    double ** min, double ** max, double ** knowns, int ** classifications, 
    double ** unknowns, string** unknownNames)
{
    ifstream myfile(FILE_NAME, ios::in);  // declare and open
    
    int numAttrib, numKnownSamp, numCla, numUn;
    myfile >> numKnownSamp >> numAttrib >> numCla >> numUn;
    
    *numAttributes = numAttrib;
    *numKnownSamples = numKnownSamp;
    *numClass = numCla;
    *numUnknowns = numUn;
    
    // Populate all of the mins and maxes
    *min = (double*) malloc(sizeof(double) * numAttrib);
    *max = (double*) malloc(sizeof(double) * numAttrib);
    for (int i = 0; i < numAttrib; i++) {
        int currentMax, currentMin;
        myfile >> currentMin >> currentMax;
        (*min)[i] = currentMin;
        (*max)[i] = currentMax;
    }
    
    
    // Populate the known object types
    *classifications =(int*) malloc(sizeof(int) * numKnownSamp);
    *knowns = (double*) malloc(sizeof(double) * numKnownSamp * numAttrib);
    
    for (int i = 0; i < numKnownSamp; i++) {
        int currentClass;
        myfile >> currentClass;
        (*classifications)[i] = currentClass;
        
        for (int j = 0; j < numAttrib; j++) {
            double currentAttrib;
            myfile >> currentAttrib;
            (*knowns)[i*numAttrib + j] = currentAttrib;
        }
    }
    
    // Populate the unknown object types
    *unknownNames = new string[numUn];
    *unknowns = (double*) malloc(sizeof(double) * numUn * numAttrib);
    
    for (int i = 0; i < numUn; i++) {
        string currentName;
        myfile >> currentName;
        (*unknownNames)[i] = currentName;
        
        for (int j = 0; j < numAttrib; j++) {
            double currentAttrib;
            myfile >> currentAttrib;
            (*unknowns)[i*numAttrib + j] = currentAttrib;
        }
    }
    
    myfile.close();
}



//this is pseudocode
int main() {
    unsigned int numBlocks = 512;
    unsigned int threadsPerBlock = 256;
    
    // Metadata about our learning algorithm data
    int numAttributes, numKnownSamples, numClass, numUnknowns;
    
    // Data that needs to be sent to the device.
    double *h_min, *h_max;
    double *h_knowns;
    int *h_classifications;
    double *h_unknowns;
    
    // Device data
    double *d_min, *d_max;
    double *d_knowns;
    int *d_classifications;
    double *d_unknowns;
    
    string *unknownNames;
    
    // Needed for the profiling
    std::clock_t start;
    double duration;
    
    parse(&numAttributes, &numKnownSamples, &numClass, &numUnknowns, 
        &h_min, &h_max, &h_knowns, &h_classifications, &h_unknowns, &unknownNames);
    
    // Start mallocing the data to the kernel
    cudaMalloc(&d_min, sizeof(double) * numAttributes);
    cudaMalloc(&d_max, sizeof(double) * numAttributes);
    cudaMalloc(&d_knowns, sizeof(double) * numKnownSamples * numAttributes);
    cudaMalloc(&d_unknowns, sizeof(double) * numUnknowns * numAttributes);
    cudaMalloc(&d_classifications, sizeof(int) * numKnownSamples);
    
    // Copy the data from the host to the kernel
    cudaMemcpy(d_min, h_min, sizeof(double) * numAttributes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, h_max, sizeof(double) * numAttributes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_knowns, h_knowns, sizeof(double) * numKnownSamples * numAttributes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_unknowns, h_unknowns, sizeof(double) * numUnknowns * numAttributes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_classifications, h_classifications, sizeof(int) * numKnownSamples, cudaMemcpyHostToDevice);

    
    // Normalize the known values
    threadsPerBlock = 256;
    numBlocks = numAttributes * numKnownSamples / threadsPerBlock;
    normalize<<<numBlocks, threadsPerBlock>>>(d_knowns, d_max, d_min, 
        numAttributes, numKnownSamples);
    
    // Normalize the unknown values
    threadsPerBlock = 256;
    numBlocks = numAttributes * numKnownSamples / threadsPerBlock;
    normalize<<<numBlocks, threadsPerBlock>>>(d_unknowns, d_max, d_min, 
        numAttributes, numUnknowns);
    
    cudaMemcpy(h_unknowns, d_unknowns, sizeof(double) * numUnknowns * numAttributes, cudaMemcpyDeviceToDevice);
    
    for (int i = 0; i < 5; i++) {
        printf("%d ", h_unknowns[i]); 
    }
    printf("\n");



    //findDistance<<<numBlocks, threadsPerBlock>>>(d_unknowns, 
}
