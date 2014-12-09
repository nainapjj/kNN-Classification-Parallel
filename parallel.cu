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

#define THREADS_PER_BLOCK 256
#define BITS_IN_BYTE 8

#define FILE_NAME "input.txt"
#define K 2

using namespace std;

__global__ void normalize(float * d_input, float *d_max, float *d_min, unsigned int numAttributes, 
    unsigned int numElems) {
    
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int attributeIdx = tid % numAttributes;
    
    if(tid < numElems*numAttributes) {
        d_input[tid] = (d_input[tid] - d_min[attributeIdx]) / (d_max[attributeIdx] - d_min[attributeIdx]);
    }
}

__global__ void findDistanceV2(float *d_inputAttributes, float **d_inputSample,  float *d_output, unsigned int numAttributes, 
    unsigned int numSamples) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < numSamples && col < numAttributes) {
        d_output[col+row*numAttributes] = (d_inputAttributes[col] - d_inputSample[row][col])*(d_inputAttributes[col] - d_inputSample[row][col]);
    }


}

__global__ void findDistance(float *d_inputAttributes, float *d_inputSample,  float *d_output, unsigned int numAttributes, 
    unsigned int numElems) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    float distance = 0;
    
    if (tid < numElems) {
        for (int i = 0; i < numAttributes; i++) {
            distance += pow(d_inputAttributes[numAttributes*tid + i] - d_inputSample[i], 2);
        }
        
        // OPTIMIZATION: We don't have to square root, because if 
        // there's no point in wasting all of the distance values are squares
        d_output[tid] = distance;
    }
}

// RADIX Sort helper function
// Map Ones and Zeros
__global__
void mapOnesZeros(unsigned int* const d_ones, unsigned int* const d_zeros, const unsigned int* const d_inputVals, 
			 const unsigned int mask, const size_t numElems) {
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	
	// Check if we're outside the bounds of the array
	if (myId < numElems) {
		if ((d_inputVals[myId] & mask) == 0) {
			d_zeros[myId] = 1;
			d_ones[myId] = 0;
		} else {
			d_zeros[myId] = 0;
			d_ones[myId] = 1;
		}
	}
}

// Reorder elements based on their generated positions
__global__
void reorderElements(unsigned int* const d_outputVals, unsigned int* const d_outputClassification, 
	const unsigned int* const d_inputVals, const unsigned int* const d_inputClassification, const unsigned int* const d_positions_zeros, 
	const unsigned int* const d_positions_ones, const unsigned int mask, const size_t numElems) {
	int myId = threadIdx.x + blockDim.x * blockIdx.x;
	
	// Check if we're outside the bounds of the array
	if (myId < numElems) {
		// Based on if the digit is zero or one depends on which position values
		if ((d_inputVals[myId] & mask) == 0) {
			d_outputVals[d_positions_zeros[myId]] = d_inputVals[myId];
			d_outputClassification[d_positions_zeros[myId]] = d_inputClassification[myId];
		} else {
			d_outputVals[d_positions_ones[myId]] = d_inputVals[myId];
			d_outputClassification[d_positions_ones[myId]] = d_inputClassification[myId];
		}
	}
}
 
void radixSort(unsigned int* const d_inputVals,
               unsigned int* const d_inputClassification,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputClassification,
               const size_t numElems)
{ 
  // Set the proper grid size and block size for this problem.
  int blockSize = THREADS_PER_BLOCK;
  int gridSize = numElems / blockSize + 1;

  // Iterate over the number of bits in the unsigned int.
  for (unsigned int i = 0; i < (sizeof(unsigned int) * BITS_IN_BYTE); i++) {
    unsigned int *d_zeros;
	unsigned int *d_ones;
    cudaMalloc(&d_zeros, sizeof(unsigned int) * numElems);
	cudaMalloc(&d_ones, sizeof(unsigned int) * numElems);
	
	// Choose which digit to check currently for our radix
	unsigned int mask = 1U << i;
	
	// Find out which digits end in 0, and which digits end in 1
	mapOnesZeros<<<gridSize, blockSize>>>(d_ones, d_zeros, d_inputVals, mask, numElems);

	// Thrust requires us to copy the memory from Cuda to the host for
	// processing.
	unsigned int *h_zeros = (unsigned int *) malloc(sizeof(unsigned int) * numElems);
	unsigned int *h_ones = (unsigned int *) malloc(sizeof(unsigned int) * numElems);
	unsigned int *h_positions_zeros = (unsigned int *) malloc(sizeof(unsigned int) * numElems);
	unsigned int *h_positions_ones = (unsigned int *) malloc(sizeof(unsigned int) * numElems);

	cudaMemcpy(h_zeros, d_zeros, sizeof(unsigned int) * numElems, 
		cudaMemcpyDeviceToHost);
	cudaMemcpy(h_ones, d_ones, sizeof(unsigned int) * numElems, 
		cudaMemcpyDeviceToHost);

	// Perform an exclusive scan on zeros to determine the position of elements with zero
	thrust::exclusive_scan(h_zeros, h_zeros + numElems, h_positions_zeros, 0);
	
	// Determine the position offset to shift the ones positions by
	// If the last element's digit is a zero, then it's the last element of d_positions_zeros
	// Otherwise, it's just the (last element of the d_positions_zeros array + 1)
	unsigned int offset;
	if (h_zeros[numElems - 1] == 1) { 
		offset = h_positions_zeros[numElems - 1] + 1;
	} else {
		offset = h_positions_zeros[numElems - 1];
	}
	
	// Perform an exclusive scan on the ones (with offset) to position elements with one
	thrust::exclusive_scan(h_ones, h_ones + numElems, h_positions_ones, offset);

	// Copy position elements to the device memory
	unsigned int *d_positions_ones;
	unsigned int *d_positions_zeros;
	cudaMalloc(&d_positions_ones, sizeof(unsigned int) * numElems);
	cudaMalloc(&d_positions_zeros, sizeof(unsigned int) * numElems);

	cudaMemcpy(d_positions_zeros, h_positions_zeros, sizeof(unsigned int) * numElems, 
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_positions_ones, h_positions_ones, sizeof(unsigned int) * numElems, 
		cudaMemcpyHostToDevice);
	
	// Now reorder the elements in cuda, based on our position items
	reorderElements<<<gridSize, blockSize>>>(d_outputVals, d_outputClassification, d_inputVals, d_inputClassification, 
		d_positions_zeros, d_positions_ones, mask, numElems);
	
	cudaMemcpy(d_inputVals, d_outputVals, sizeof(unsigned int) * numElems, 
		cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_inputClassification, d_outputClassification, sizeof(unsigned int) * numElems, 
		cudaMemcpyDeviceToDevice);
	
	// Clear all of our allocated memory
	cudaFree(d_positions_ones);
	cudaFree(d_positions_zeros);
	cudaFree(d_ones);
	cudaFree(d_zeros);

	free(h_zeros);
	free(h_ones);
	free(h_positions_ones);
	free(h_positions_zeros);
  }
}

/*__global__ void block_sum(float *input, float *results, size_t n)
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
    float ** min, float ** max, float ** knowns, int ** classifications, 
    float ** unknowns, string** unknownNames)
{
    ifstream myfile(FILE_NAME, ios::in);  // declare and open
    
    int numAttrib, numKnownSamp, numCla, numUn;
    myfile >> numKnownSamp >> numAttrib >> numCla >> numUn;
    
    *numAttributes = numAttrib;
    *numKnownSamples = numKnownSamp;
    *numClass = numCla;
    *numUnknowns = numUn;
    
    // Populate all of the mins and maxes
    *min = (float*) malloc(sizeof(float) * numAttrib);
    *max = (float*) malloc(sizeof(float) * numAttrib);
    for (int i = 0; i < numAttrib; i++) {
        int currentMax, currentMin;
        myfile >> currentMin >> currentMax;
        (*min)[i] = currentMin;
        (*max)[i] = currentMax;
    }
    
    
    // Populate the known object types
    *classifications =(int*) malloc(sizeof(int) * numKnownSamp);
    *knowns = (float*) malloc(sizeof(float) * numKnownSamp * numAttrib);
    
    for (int i = 0; i < numKnownSamp; i++) {
        int currentClass;
        myfile >> currentClass;
        (*classifications)[i] = currentClass;
        
        for (int j = 0; j < numAttrib; j++) {
            float currentAttrib;
            myfile >> currentAttrib;
            (*knowns)[i*numAttrib + j] = currentAttrib;
        }
    }
    
    // Populate the unknown object types
    *unknownNames = new string[numUn];
    *unknowns = (float*) malloc(sizeof(float) * numUn * numAttrib);
    
    for (int i = 0; i < numUn; i++) {
        string currentName;
        myfile >> currentName;
        (*unknownNames)[i] = currentName;
        
        for (int j = 0; j < numAttrib; j++) {
            float currentAttrib;
            myfile >> currentAttrib;
            (*unknowns)[i*numAttrib + j] = currentAttrib;
        }
    }
    
    myfile.close();
}



int main() {
    unsigned int numBlocks = 512;
    unsigned int threadsPerBlock = 256;
    
    // Metadata about our learning algorithm data
    int numAttributes, numKnownSamples, numClass, numUnknowns;
    
    // Data that needs to be sent to the device.
    float *h_min, *h_max;
    float *h_knowns;
    int *h_classifications;
    float *h_unknowns;
    
    // Device data
    float *d_min, *d_max;
    float *d_knowns;
    int *d_classifications;
    float *d_unknowns;
    
    string *unknownNames;
    
    // Needed for the profiling
    std::clock_t start;
    float duration;
    
    parse(&numAttributes, &numKnownSamples, &numClass, &numUnknowns, 
        &h_min, &h_max, &h_knowns, &h_classifications, &h_unknowns, &unknownNames);
    
    // Start mallocing the data to the kernel
    cudaMalloc(&d_min, sizeof(float) * numAttributes);
    cudaMalloc(&d_max, sizeof(float) * numAttributes);
    cudaMalloc(&d_knowns, sizeof(float) * numKnownSamples * numAttributes);
    cudaMalloc(&d_unknowns, sizeof(float) * numUnknowns * numAttributes);
    cudaMalloc(&d_classifications, sizeof(int) * numKnownSamples);
    
    // Copy the data from the host to the kernel
    cudaMemcpy(d_min, h_min, sizeof(float) * numAttributes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_max, h_max, sizeof(float) * numAttributes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_knowns, h_knowns, sizeof(float) * numKnownSamples * numAttributes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_unknowns, h_unknowns, sizeof(float) * numUnknowns * numAttributes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_classifications, h_classifications, sizeof(int) * numKnownSamples, cudaMemcpyHostToDevice);
    
    // Normalize the known values
    threadsPerBlock = 256;
    numBlocks = numAttributes * numKnownSamples / threadsPerBlock + 1;
    normalize<<<numBlocks, threadsPerBlock>>>(d_knowns, d_max, d_min, 
        numAttributes, numKnownSamples);
    
    // Normalize the unknown values
    threadsPerBlock = 256;
    numBlocks = numAttributes * numKnownSamples / threadsPerBlock + 1;
    normalize<<<numBlocks, threadsPerBlock>>>(d_unknowns, d_max, d_min, 
        numAttributes, numUnknowns);
        
    
    // Find the distances between the  
    float *d_distance;
    cudaMalloc(&d_distance, sizeof(float) * numKnownSamples);
    threadsPerBlock = 256;
    numBlocks = numAttributes / threadsPerBlock + 1;
    
    findDistance<<<numBlocks, threadsPerBlock>>>(d_knowns, d_unknowns+0,  d_distance, 
        numAttributes, numKnownSamples);
    
    /*float *h_distance = (float*) malloc(sizeof(float) * numKnownSamples);
    cudaMemcpy(h_distance, d_distance, sizeof(float) * numKnownSamples, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < numKnownSamples; i++) {
        printf("%f ", h_distance[i]); 
    }
    printf("\n");*/
    
    float *d_outputClassification;
    float *d_outputDistances;
    
    // Perform the sort
    cudaMalloc(&d_outputClassification, sizeof(float) * numKnownSamples);
    cudaMalloc(&d_outputDistances, sizeof(float) * numKnownSamples);
    
    radixSort((unsigned int*) d_distance,
               (unsigned int*) d_classifications,
               (unsigned int*) d_outputDistances,
               (unsigned int*) d_outputClassification,
               numKnownSamples);
               
    // Check to see if the sort worked
    float *h_outputDistances = (float*) malloc(sizeof(float) * numKnownSamples);
    float *h_outputClassifications = (float*) malloc(sizeof(float) * numKnownSamples);
    cudaMemcpy(h_outputDistances, d_outputDistances, sizeof(float) * numKnownSamples, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_outputClassifications, d_outputClassification, sizeof(float) * numKnownSamples, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < numKnownSamples; i++) {
        cout << h_outputClassifications[i] << " " << h_outputDistances[i] << endl;
    }
    
    /*cudaMemcpy(h_unknowns, d_unknowns, sizeof(float) * numUnknowns * numAttributes, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_unknowns[i]); 
    }
    printf("\n");*/

    //float * distance;
    //cudaMalloc(&distance, sizeof(float) * numSomething
    //findDistance<<<numBlocks, threadsPerBlock>>>(d_knowns, d_unknowns, d_distance,  
    
    cudaMemcpy(h_unknowns, d_unknowns, sizeof(float) * numUnknowns * numAttributes, cudaMemcpyDeviceToHost);
}
