/*******************************************************************
*			Sparse Auto-Encoder 
*					by
*		David Klaus and Alex Welles
*			EC527 Final Project	
*
*	CUDA Implementation With Timing Code
*	
*	Compile with: 
*
*	nvcc -o cudaNAIVE sparseAutoencoder_CUDA_NAIVE.cu
*
*	Contains Naive implementation for CUDA that is limited by shared
*	memory size. Also contains the alternative atomic add version of
*	the forward prop kernel. see sparseAutoencoder_CUDA_CHUNK.cu for
* 	optimized code.
*
*	If the kernel kernel_rho_forwardProp2 is commented in use:
*	nvcc -arch=sm_20 -o sparseAutoencoder_cuda sparseAutoencoder_cuda.cu
*
*******************************************************************/

#include <cstdio>//<stdio.h>
#include <cstdlib>//<stdio.h>
#include <time.h>
#include <math.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>


#define GIG 1000000000
#define CPG 2.527
#define OPTIONS 1
#define TOL 0.000001
#define PRINT_TIME 1

//Parameters necessary to set up network
#define PATCHES_PATH	"c_patches.csv"//For DEBUG
#define W1_PATH			"W1.csv"//For DEBUG
#define W2_PATH			"W2.csv"//For DEBUG
#define IMAGE_DIM		512		//pixels in 1 dimension (assumes square)
#define SAMPLE_SIZE		10000  //number of input patches
#define	VISIBLE_SIZE 	64      //number of input/output nodes 
#define	HIDDEN_SIZE 	128	   	//number hidden nodes
#define	HIDDEN_LAYERS 	1		//number hidden layers (NON-FUNCTIONAL)
#define NUM_SAMPLE_ELEMENTS SAMPLE_SIZE * VISIBLE_SIZE

//desired average activation of hidden nodes
#define	SPARSITY_PARAM 	0.01
#define SPARSITY_COMPLEMENT 1-SPARSITY_PARAM
//weight decay paramater
#define	LAMBDA			0.0001
//weight of sparsity penalty term
#define	BETA			3.0




// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

using namespace std;



	/***********************************
	 		KERNEL VECTOR OPS
    ***********************************/


//http://stackoverflow.com/questions/14291233/confusion-about-cuda-partial-sum-codes-threadidx-x-blockdim-x-and-a-2
//https://code.google.com/p/stanford-cs193g-sp2010/source/browse/trunk/tutorials/sum_reduction.cu
__global__ void kernel_block_sum(const float *input, float *per_block_results, const size_t n)
{
	extern __shared__ float sdata[];

	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// load input into __shared__ memory
	float x = 0;
	
	if(tid < n)
	{
		x = input[tid];
}	

	sdata[threadIdx.x] = x;

	__syncthreads();

	// contiguous range pattern --> div by 2 (may not be best)
	for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if(threadIdx.x < offset)
		{
			// add a partial sum upstream to our own
			sdata[threadIdx.x] += sdata[threadIdx.x + offset];
		}

		// wait until all threads in the block have
		// updated their partial sums
		__syncthreads();
	}

	// thread 0 writes the final result
	if(threadIdx.x == 0)
	{
		per_block_results[blockIdx.x] = sdata[0];
	}
}


//dim3 gridDim(HIDDEN_SIZE,1,1);
//dim3 blockDim(VISIBLE_SIZE,1,1);
__global__ void kernel_rho_forwardProp(float *input, float* W1, float* b1, float* rhoHat)
{
	//load shared memory with what you need
	__shared__ float inputS[VISIBLE_SIZE];
	__shared__ float W1S[VISIBLE_SIZE];
	__shared__ float b1S;

	inputS[threadIdx.x] = input[threadIdx.x];
	W1S[threadIdx.x] = W1[blockIdx.x * blockDim.x + threadIdx.x]; \
	b1S = b1[blockIdx.x];
	__syncthreads();

	//store the calculated value back in the shared weights array
	W1S[threadIdx.x] = W1S[threadIdx.x] * inputS[threadIdx.x];
	__syncthreads();

	for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if(threadIdx.x < offset)
		{
			// add a partial sum upstream to our own
			W1S[threadIdx.x] += W1S[threadIdx.x + offset];
		}

		// wait until all threads in the block have
		// updated their partial sums
		__syncthreads();
	}

	//thread 0 writes the final results.
	if( threadIdx.x == 0 )
	{
		//add the bias vector value
		W1S[0] += b1S;

		//apply sigma function
		W1S[0] = float(1/(1+exp(-W1S[0])));

		//accumulate rhoHat
		rhoHat[blockIdx.x] += W1S[0];
	}
}


//dim3 gridDim(HIDDEN_SIZE,1,1);
//dim3 blockDim(VISIBLE_SIZE,1,1);
__global__ void kernel_forwardProp1(float *input, float* W1, 
									float* b1, float* a2)
{
	__shared__ float inputS[VISIBLE_SIZE];
	__shared__ float W1S[VISIBLE_SIZE];
	__shared__ float b1S;

	inputS[threadIdx.x] = input[threadIdx.x];
	W1S[threadIdx.x] = W1[blockIdx.x * blockDim.x + threadIdx.x];
	b1S = b1[blockIdx.x];

	W1S[threadIdx.x] = W1S[threadIdx.x] * inputS[threadIdx.x];
	__syncthreads();
	
	for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if(threadIdx.x < offset)
		{
			// add a partial sum upstream to our own
			W1S[threadIdx.x] += W1S[threadIdx.x + offset];
		}

		// wait until all threads in the block have
		// updated their partial sums
		__syncthreads();
	}

	//only the 0 thread of each block does this.
	if( threadIdx.x == 0 )
	{
		//add the bias vector value
		W1S[0] += b1S;

		//apply sigma function and transfer back to global mem
		a2[blockIdx.x] = float(1/(1+exp(-W1S[0])));

		//perform other operations here
	}
}

//dim3 gridDim(VISIBLE_SIZE,1,1);
//dim3 blockDim(HIDDEN_SIZE,1,1);
__global__ void kernel_forwardProp2(float *input, float* W2, float* b2, float* a3)
{
	__shared__ float inputS[HIDDEN_SIZE];
	__shared__ float W2S[HIDDEN_SIZE];
	__shared__ float b2S;

	//inputS[threadIdx.x] = input[blockIdx.x * blockDim.x + threadIdx.x];
	inputS[threadIdx.x] = input[threadIdx.x];
	W2S[threadIdx.x] = W2[blockIdx.x * blockDim.x + threadIdx.x];
	b2S = b2[blockIdx.x];
	__syncthreads();
	
	W2S[threadIdx.x] = W2S[threadIdx.x] * inputS[threadIdx.x];
	__syncthreads();
	
	for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if(threadIdx.x < offset)
		{
			// add a partial sum upstream to our own
			W2S[threadIdx.x] += W2S[threadIdx.x + offset];
		}

		// wait until all threads in the block have
		// updated their partial sums
		__syncthreads();
	}

	//only the 0 thread of each block does this.
	if( threadIdx.x == 0 )
	{
		//add the bias vector value
		if(blockDim.x % 2 > 0)
		{
			W2S[0] += W2S[HIDDEN_SIZE-1];
		}
		W2S[0] += b2S;

		//apply sigma function and transfer back to global mem
		a3[blockIdx.x] = float(1/(1+exp(-W2S[0])));
	}
}


//dim3 gridDim(1,1,1);
//dim3 blockDim(VISIBLE_SIZE,1,1);
__global__ void kernel_backProp1(float* input, float* a3, float* d3, 
								 float* b2grad, float* cost)
{
	__shared__ float a3S[VISIBLE_SIZE];
	__shared__ float b2gradS[VISIBLE_SIZE];
	__shared__ float d3S[VISIBLE_SIZE];
	__shared__ float inputS[VISIBLE_SIZE];

	b2gradS[threadIdx.x] = b2grad[threadIdx.x];
	a3S[threadIdx.x] = a3[threadIdx.x];
	inputS[threadIdx.x] = input[threadIdx.x];
	__syncthreads();

	//d3 = -(xM - a3) .* (a3 .* (1 - a3));
	d3S[threadIdx.x] = -(inputS[threadIdx.x] - a3S[threadIdx.x]) 
						* (a3S[threadIdx.x] * (1-a3S[threadIdx.x]));

	//update the gradient
	//b2grad = b2grad + d3;
	b2gradS[threadIdx.x] += d3S[threadIdx.x];
	b2grad[threadIdx.x] = b2gradS[threadIdx.x];
	d3[threadIdx.x] = d3S[threadIdx.x];
	__syncthreads();

	//cost = cost + norm(a3 - xM)^2; 
	a3S[threadIdx.x] -= inputS[threadIdx.x];
	a3S[threadIdx.x] *= a3S[threadIdx.x];

	for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if(threadIdx.x < offset)
		{
			// add a partial sum upstream to our own
			a3S[threadIdx.x] += a3S[threadIdx.x + offset];
		}

		// wait until all threads in the block have
		// updated their partial sums
		__syncthreads();
	}

	if( threadIdx.x == 0 )
	{
		cost[0] = a3S[0];
	}
}



//dim3 gridDim(HIDDEN_SIZE,1,1);
//dim3 blockDim(VISIBLE_SIZE,1,1);
__global__ void kernel_backProp2(float* W2, float* d2, float* d3)
{
	__shared__ float d3S[VISIBLE_SIZE];
	__shared__ float W2S[VISIBLE_SIZE];

	//here we are loading the transpose into memory hence the somewhat odd looking indexing
	W2S[threadIdx.x] = W2[threadIdx.x * gridDim.x + blockIdx.x];
	d3S[threadIdx.x] = d3[threadIdx.x];
	__syncthreads();

	W2S[threadIdx.x] *= d3S[threadIdx.x];
	__syncthreads();

	for(int offset = blockDim.x / 2; offset > 0; offset >>= 1)
	{
		if(threadIdx.x < offset)
		{
			W2S[threadIdx.x] += W2S[threadIdx.x + offset];
		}
		__syncthreads();
	}

	if(threadIdx.x == 0)
	{
		//storing the interim value of d2 in d_d2 for next kernel call.
		//d2 = (W2' * d3)
		d2[blockIdx.x] = W2S[0];
	}
}


//dim3 gridDim(1,1,1);
//dim3 blockDim(HIDDEN_SIZE,1,1);
__global__ void kernel_backProp3(float* a2, float* d2, float* rhoHat, float* b1grad)
{
	//d2 = ((W2Trans * d3) + beta .* (-(sparsityParam./rhoHat) + (1-sparsityParam)./(1-rhoHat)))
			//.* (a2 .* (1 - a2));
	__shared__ float d2S[HIDDEN_SIZE];
	__shared__ float rhoHatS[HIDDEN_SIZE];
	__shared__ float a2S[HIDDEN_SIZE];
	__shared__ float b1gradS[HIDDEN_SIZE];

	b1gradS[threadIdx.x] = b1grad[threadIdx.x];
	d2S[threadIdx.x] = d2[threadIdx.x];
	rhoHatS[threadIdx.x] = rhoHat[threadIdx.x];
	a2S[threadIdx.x] = a2[threadIdx.x];
	__syncthreads();

	//calculate d2
	d2S[threadIdx.x] = (d2S[threadIdx.x] + BETA * (-(SPARSITY_PARAM/rhoHatS[threadIdx.x])) 
						+ (1-SPARSITY_PARAM)/(1-rhoHatS[threadIdx.x])) * (a2S[threadIdx.x] 
						* (1 - a2S[threadIdx.x]));

	d2[threadIdx.x] = d2S[threadIdx.x];

	//update b1 gradient
	b1gradS[threadIdx.x] += d2S[threadIdx.x];
	b1grad[threadIdx.x] = b1gradS[threadIdx.x];
}


//dim3 gridDim(HIDDEN_SIZE,1,1);
//dim3 blockDim(VISIBLE_SIZE,1,1);
__global__ void kernel_backProp4(float* input, float* a2, float* d2, float* d3, float* W1grad, float* W2grad)
{
	__shared__ float d3S[VISIBLE_SIZE];
	__shared__ float a2S;
	__shared__ float d2S;
	__shared__ float W2gradS[VISIBLE_SIZE];
	__shared__ float W1gradS[VISIBLE_SIZE];
	__shared__ float inputS[VISIBLE_SIZE];

	
	W1gradS[threadIdx.x] = W1grad[blockIdx.x * blockDim.x + threadIdx.x]; 
	//W2gradS is read in transposed
	W2gradS[threadIdx.x] = W2grad[threadIdx.x * gridDim.x + blockIdx.x];
	d3S[threadIdx.x] = d3[threadIdx.x];
	inputS[threadIdx.x] = input[threadIdx.x];

	if(threadIdx.x == 0)
	{
		a2S = a2[blockIdx.x];
		d2S = d2[blockIdx.x];
	}
	__syncthreads();

	//W2grad = W2grad + d3 * a2';
	W2gradS[threadIdx.x] += a2S * d3S[threadIdx.x];
	W2grad[threadIdx.x] = W2gradS[threadIdx.x];
	__syncthreads();//?

	//W1grad = W1grad + d2 * xM';
	W1gradS[threadIdx.x] +=  d2S * inputS[threadIdx.x];
	W1grad[threadIdx.x] = W1gradS[threadIdx.x];
}


//dim3 gridDim(HIDDEN_SIZE,1,1);
//dim3 blockDim(VISIBLE_SIZE,1,1);
__global__ void kernel_sparsityEnforcement(float* W1, float* W2, float* W1grad, float* W2grad)
{
	__shared__ float W1gradS[VISIBLE_SIZE];
	__shared__ float W2gradS[VISIBLE_SIZE];
	__shared__ float W1S[VISIBLE_SIZE];
	__shared__ float W2S[VISIBLE_SIZE];

	W1gradS[threadIdx.x] = W1grad[blockIdx.x * blockDim.x + threadIdx.x];
	W1S[threadIdx.x] = W1[blockIdx.x * blockDim.x + threadIdx.x];
	//read W2 in as transpose
	W2gradS[threadIdx.x] = W2grad[threadIdx.x * gridDim.x + blockIdx.x];
	W2S[threadIdx.x] = W2[threadIdx.x * gridDim.x + blockIdx.x];
	__syncthreads();

	//W1grad = W1grad ./ M + lambda .* W1;
	W1gradS[threadIdx.x] = W1gradS[threadIdx.x]/SAMPLE_SIZE + LAMBDA * W1S[threadIdx.x];
	W2gradS[threadIdx.x] = W2gradS[threadIdx.x]/SAMPLE_SIZE + LAMBDA * W2S[threadIdx.x];

	//W2grad = W2grad ./ M + lambda .* W2;
	W1grad[blockIdx.x * blockDim.x + threadIdx.x] = W1gradS[threadIdx.x];
	W2grad[threadIdx.x * gridDim.x + blockIdx.x] = W2gradS[threadIdx.x];
}

//an alternative way to perform forward propagation. Currently not optimized for blocks that require large amounts of shared memory.


//***************************************
//		 kernel_rho_forwardProp2
//***************************************

/*
dim3 gridDim1(HIDDEN_SIZE,1,1);
dim3 blockDim1(VISIBLE_SIZE,1,1);
for(int i = 0;i < NUM_SAMPLE_ELEMENTS; i+= VISIBLE_SIZE*HIDDEN_SIZE)
{
	cudaPrintfInit ();
	kernel_rho_forwardProp2<<<gridDim1, blockDim1>>>(&d_inputs[i], d_W1, d_b1, d_rhoHat);
	cudaPrintfDisplay (stdout, true);
	cudaPrintfEnd ();
	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());//may want to remove
}
*/


//dim3 blockDim(VISIBLE_SIZE,1,1);//each block is a row in the W1 matrix
//dim3 gridDim(HIDDEN_SIZE,1,1);//grid is the total number of rows
__global__ void kernel_rho_forwardProp2(float* input, float* W1, float* b1, float* global_rho_hat) 
{

	//load shared memory
	__shared__ float b1S[HIDDEN_SIZE];
	__shared__ float rho_hatS[HIDDEN_SIZE];
	__shared__ float W1S[HIDDEN_SIZE*VISIBLE_SIZE];
	__shared__ float inputS[HIDDEN_SIZE*VISIBLE_SIZE];
	__shared__ float tempS[VISIBLE_SIZE];

	//initialize variables
	if(threadIdx.x < HIDDEN_SIZE) {
		/* parallel conditional initialization for bias*/
		b1S[threadIdx.x] = b1[threadIdx.x];
	}

	for(int i = 0; i < HIDDEN_SIZE; i++) {
		/* load entire row of W1S on each iteration of the for loop. 
		 * ie load 0 1 2 .... 63 in parallel then 64 65 66 ... 127 in parallel 
		 */
		W1S[(VISIBLE_SIZE*i)+threadIdx.x] = W1[(VISIBLE_SIZE*i)+threadIdx.x];
		//W1[(VISIBLE_SIZE*i)+threadIdx.x] = (VISIBLE_SIZE*i)+threadIdx.x;
	}

	for(int i = 0; i < HIDDEN_SIZE; i++) {
		/* load entire row of inputS on each iteration of the for loop. 
		 * ie load 0 1 2 .... 63 in parallel then 64 65 66 ... 127 in parallel 
		 */
		inputS[(VISIBLE_SIZE*i)+threadIdx.x] = input[(VISIBLE_SIZE*i)+threadIdx.x];
		//input[(VISIBLE_SIZE*i)+threadIdx.x] = (VISIBLE_SIZE*i) + threadIdx.x;
	}
	__syncthreads(); //make sure all threads finish reading in data
	

	int tileOffset = blockIdx.x*VISIBLE_SIZE;	//begin tile offset based on rowblock distance from 0
	int ii = tileOffset+threadIdx.x;			//offset by tile amount used to loop 
	int jj = threadIdx.x;						//do not offset start at beginning
	
	
	for(;ii<VISIBLE_SIZE*HIDDEN_SIZE; ii+=VISIBLE_SIZE, jj+=VISIBLE_SIZE ) { //tile from tile offset of W1 to end of array	
		
		//multiply tile
		tempS[threadIdx.x] = W1S[jj]*inputS[ii];
		//tempS[threadIdx.x] = 1;
		__syncthreads();

		for(int k = VISIBLE_SIZE/2; k>0; k>>=1) { //sum tile using divide and conqure summation
			if(threadIdx.x < k) {
				tempS[threadIdx.x] += tempS[threadIdx.x+k];
			}
			__syncthreads();
		}

		if(threadIdx.x == 0) { //place resultant values in appropriate position in rho_hat

			tempS[0] += b1S[jj/VISIBLE_SIZE];

			//apply sigmoid function
			tempS[0] = float(1/(1+exp(-tempS[0])));

			rho_hatS[jj/VISIBLE_SIZE] = tempS[0];
		}
		
		__syncthreads();		
	}

	//tile from beginning of W1 to tile offset
	for(ii=0; ii < tileOffset; ii+=VISIBLE_SIZE, jj+=VISIBLE_SIZE) { 
		
		//multiply tile in parallel
		tempS[threadIdx.x] = W1S[jj]*inputS[ii];
		//tempS[threadIdx.x] = 1;
		__syncthreads();

		for(int k = VISIBLE_SIZE/2; k>0; k>>=1) { //sum tile using divide and conqure summation
			if(threadIdx.x < k) {
				tempS[threadIdx.x] += tempS[threadIdx.x+k];
			}
			__syncthreads();
		}

		if(threadIdx.x == 0) { //place resultant values in appropriate position in rho_hat

			//add component of dot product
			tempS[0] += b1S[jj/VISIBLE_SIZE];

			//apply sigmoid function
			tempS[0] = float(1/(1+exp(-tempS[0])));

			rho_hatS[jj/VISIBLE_SIZE] = tempS[0];
		}
		
		__syncthreads();	
	}

	//parallel sum rho_hat in each thread in each block to produce output
	if(threadIdx.x < HIDDEN_SIZE) {
		
		atomicAdd(&global_rho_hat[threadIdx.x],rho_hatS[threadIdx.x]);
		//global_rho_hat[threadIdx.x] += rho_hatS[threadIdx.x];
	}
}

	/**********************************
	 		SERIAL VECTOR OPS
    ***********************************/

void initializeMatrixWeightsRand(float *arr, int rows, int cols, int seed);
void initializeMatrixWeightsZero(float *arr, int rows, int cols);
void initializeVectorWeightsZero(float *arr, int numElements);
void mmm_kij(float* src1, float* src2, float* dest, int row1, int col1, int row2,int col2);
void mmm_ijk(float* src1, float* src2, float* dest, int row1, int col1, int row2,int col2);
void dotPdt(float* src1,float* src2, float* dest, int length);
void readCSV(float* array, int numElements, string filename);
void addVectors(float* src1, float* src2, float* dest, int length);
void subVectors(float* src1, float* src2, float* dest, int length);
void vectElemSigmoid(float* src,float* dest,int length);
void vectElemIntDiv(float* src, float* dest,int length,int divisor);
void vectElemFloatDiv(float* src, float* dest,int length,float divisor);
void vectElemVectDiv(float* src1,float* src2,float* dest,int length);
void initializeVector(float *array, int length, float val);
void vectElemVectMult(float* src1, float* src2, float* dest, int length);
void vectElemFloatMult(float* src, float* dest, int length,float multiplicand);
void matrixTranspose(float* src,float* dest,int rows, int cols);
float normVector(float* src,int length);
void vectElemLog(float* src,float* dest,int length);
float sumVector(float* src,int length);
/* PRINTOUT, DEBUG, AND TIMING FUNCTIONS */
void printVector(float* A, int length);
void printMatrix(float* A, int rows, int cols);
void printTiming(struct timespec* time_stamp,int numTimings);

int main(int argc, char *argv[])
{
	/***********************************
	 		  TIMING STUFF
    ***********************************/

	//CPU
	struct timespec diff(struct timespec start, struct timespec end);
    struct timespec time1, time2;
    struct timespec time_stamp[OPTIONS];//Can be increased if necessary.

    // GPU Timing variables
	cudaEvent_t start, stop;
	float elapsed_gpu;
	
	/***********************************
	 		ALLOCATE HOST MEMORY
    ***********************************/

	//Arrays on host memory (CPU)
	//input patches to train the autoencoder
	float *h_inputs;// 64 x 10000 [visible x sample]
	//sparsity vector
	float *h_rhoHat;//hidden x 1 [25 x 1]
	//weight matrices
	float *h_W1;//hidden X visible [25 x 64]
	float *h_W2;//visible X hidden [64 x 25]
	//weight vectors
	float *h_b1;//hidden X 1 [25 x 1]
	float *h_b2;//visible X 1 [64 x 1]
	//weight gradient matrices
	float *h_W1grad;//hidden x visible [25 x 64]
	float *h_W2grad;//visible x hidden [64 x 25]
	//weight gradient vectors
	float *h_b1grad;//hidden x 1 [25 x 1]
	float *h_b2grad;//visible x 1 [64 x 1]
	//z product vectors
	float *h_z2;//hidden x 1 [25 x 1]
	float *h_z3;//visible x 1 [64 x 1]
	//a product vectors
	float *h_a2;//hidden x 1 [25 x 1]
	float *h_a3;//visible x 1 [64 x 1]
	//partial derivatives for back prop
	float *h_d2;//hidden x 1 [25 x 1]
	float *h_d3;//visible x 1 [64 x 1]
	//temp vectors: both are 64 elements but will not always be used
	float *h_temp1;//64 x 1
	float *h_temp2;//64 x1
	//temp matrix
	float *h_Wtemp1;//64 x 25 or 25 x 64
	float *h_Wtemp2;//25 x 64 or 64 x 25
	//sparsity penalty
	float *h_sparsePen;//25x1
	float *h_cost;

	//Allocate input patches on host memory (CPU)
	size_t allocSize = VISIBLE_SIZE * SAMPLE_SIZE * sizeof(float);
	h_inputs = (float *) malloc(allocSize);

	//Allocate sparsity vector on host memory (CPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	h_rhoHat = (float *) malloc(allocSize);
	
	//Alocate weight arrays on host memory (CPU)
	allocSize = VISIBLE_SIZE * HIDDEN_SIZE * sizeof(float);
	h_W1 = (float *) malloc(allocSize);
	h_W2 = (float *) malloc(allocSize);

	//Alocate gradient arrays on host memory (CPU)
	allocSize = VISIBLE_SIZE * HIDDEN_SIZE * sizeof(float);
	h_W1grad = (float *) malloc(allocSize);
	h_W2grad = (float *) malloc(allocSize);
	
	//Allocate weight vectors on host memory (CPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	h_b1 = (float *) malloc(allocSize);
	allocSize = VISIBLE_SIZE * sizeof(float);
	h_b2 = (float *) malloc(allocSize);	

	//Allocate weight vectors on host memory (CPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	h_b1grad = (float *) malloc(allocSize);
	allocSize = VISIBLE_SIZE * sizeof(float);
	h_b2grad = (float *) malloc(allocSize);	

	//Allocate z product vectors (CPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	h_z2 = (float *) malloc(allocSize);
	allocSize = VISIBLE_SIZE * sizeof(float);
	h_z3 = (float *) malloc(allocSize);

	//Allocate a product vectors (CPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	h_a2 = (float *) malloc(allocSize);
	allocSize = VISIBLE_SIZE * sizeof(float);
	h_a3 = (float *) malloc(allocSize);

	//Allocate partial vectors (CPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	h_d2 = (float *) malloc(allocSize);
	allocSize = VISIBLE_SIZE * sizeof(float);
	h_d3 = (float *) malloc(allocSize);

	//Allocate temp vectors (CPU)
	allocSize = VISIBLE_SIZE * sizeof(float);
	h_temp1 = (float *) malloc(allocSize);
	h_temp2 = (float *) malloc(allocSize);

	//Allocate temp matrix (CPU)
	allocSize = VISIBLE_SIZE * HIDDEN_SIZE * sizeof(float);
	h_Wtemp1 = (float *) malloc(allocSize);
	h_Wtemp2 = (float *) malloc(allocSize);

	//Allocate sparsity penalty vector (CPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	h_sparsePen = (float *) malloc(allocSize);

	allocSize = sizeof(float);
	h_cost = (float *) malloc(allocSize);

	/***********************************
	 		ALLOCATE DEVICE MEMORY
    ***********************************/

	//input patches to train the autoencoder
	float *d_inputs;// 64 x 10000 [visible x sample]
	//sparsity vector
	float *d_rhoHat;//hidden x 1 [25 x 1]
	//weight matrices
	float *d_W1;//hidden X visible [25 x 64]
	float *d_W2;//visible X hidden [64 x 25]
	//weight vectors
	float *d_b1;//hidden X 1 [25 x 1]
	float *d_b2;//visible X 1 [64 x 1]
	//weight gradient matrices
	float *d_W1grad;//hidden x visible [25 x 64]
	float *d_W2grad;//visible x hidden [64 x 25]
	//weight gradient vectors
	float *d_b1grad;//hidden x 1 [25 x 1]
	float *d_b2grad;//visible x 1 [64 x 1]
	//a product vectors
	float *d_a2;//hidden x 1 [25 x 1]
	float *d_a3;//visible x 1 [64 x 1]
	//partial derivatives for back prop
	float *d_d2;//hidden x 1 [25 x 1]
	float *d_d3;//visible x 1 [64 x 1]
	//sparsity penalty
	float *d_sparsePen;//25x1
	float *d_cost;//1 lonely float

	//Allocate input patches on device memory (GPU)
	allocSize = VISIBLE_SIZE * SAMPLE_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_inputs,allocSize));

	//Allocate sparsity vector on device memory (GPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_rhoHat,allocSize));
	
	//Alocate weight arrays on device memory (GPU)
	allocSize = VISIBLE_SIZE * HIDDEN_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_W1,allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_W2,allocSize));

	//Alocate gradient arrays on device memory (GPU)
	allocSize = VISIBLE_SIZE * HIDDEN_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_W1grad,allocSize));
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_W2grad,allocSize));
	
	//Allocate weight vectors on device memory (GPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b1,allocSize));
	allocSize = VISIBLE_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b2,allocSize));	

	//Allocate weight vectors on device memory (GPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b1grad,allocSize));
	allocSize = VISIBLE_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_b2grad,allocSize));	

	//Allocate a product vectors (GPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_a2,allocSize));
	allocSize = VISIBLE_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_a3,allocSize));

	//Allocate partial vectors (GPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_d2,allocSize));
	allocSize = VISIBLE_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_d3,allocSize));

	//Allocate sparsity penalty vector (GPU)
	allocSize = HIDDEN_SIZE * sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_sparsePen,allocSize));

	//Allocate cost (GPU)
	allocSize = sizeof(float);
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_cost,allocSize));

	/***********************************
	 	INITIALIZE NETWORK WEIGHTS
    ***********************************/

	//Initialize the weight matrices to random values
	//initializeMatrixWeightsRand(h_inputs, VISIBLE_SIZE, SAMPLE_SIZE, 2254); 	
	 	
	initializeMatrixWeightsRand(h_W1, HIDDEN_SIZE, VISIBLE_SIZE, 2254);
	initializeMatrixWeightsRand(h_W2, VISIBLE_SIZE, HIDDEN_SIZE, 1345);

	initializeMatrixWeightsZero(h_W2grad,VISIBLE_SIZE,HIDDEN_SIZE);
	initializeMatrixWeightsZero(h_W1grad,HIDDEN_SIZE,VISIBLE_SIZE);

	initializeVectorWeightsZero(h_b1, HIDDEN_SIZE);
	initializeVectorWeightsZero(h_b2, VISIBLE_SIZE);
	initializeVectorWeightsZero(h_rhoHat, HIDDEN_SIZE);
	initializeVectorWeightsZero(h_z2, HIDDEN_SIZE);
	initializeVectorWeightsZero(h_a2, HIDDEN_SIZE);
	initializeVectorWeightsZero(h_z3, VISIBLE_SIZE);
	initializeVectorWeightsZero(h_a3, VISIBLE_SIZE);

	/***********************************
	 	   READ IN SAMPLE PATCHES
    ***********************************/

	readCSV(h_inputs, NUM_SAMPLE_ELEMENTS, PATCHES_PATH);
	//the following are for debug only
	readCSV(h_W1, HIDDEN_SIZE*VISIBLE_SIZE, W1_PATH);
	readCSV(h_W2, HIDDEN_SIZE*VISIBLE_SIZE, W2_PATH);

	/***************************************
			   BEGIN CUDA TIMING
	****************************************/

#if PRINT_TIME
	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Record event on the default stream
	cudaEventRecord(start, 0);
#endif

	// Transfer the arrays to the GPU memory
	allocSize = VISIBLE_SIZE * SAMPLE_SIZE * sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpy(d_inputs,h_inputs, allocSize, cudaMemcpyHostToDevice));
	
	allocSize = VISIBLE_SIZE * HIDDEN_SIZE * sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpy(d_W1,h_W1, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_W2,h_W2, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_W1grad,h_W1grad, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_W2grad,h_W2grad, allocSize, cudaMemcpyHostToDevice));

    allocSize = HIDDEN_SIZE * sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpy(d_rhoHat,h_rhoHat, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b1,h_b1, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b1grad,h_b1grad, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_a2,h_a2, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_d2,h_d2, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_sparsePen,h_sparsePen, allocSize, cudaMemcpyHostToDevice));


    allocSize = VISIBLE_SIZE * sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpy(d_b2,h_b2, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_b2grad,h_b2grad, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_a3,h_a3, allocSize, cudaMemcpyHostToDevice));  
    CUDA_SAFE_CALL(cudaMemcpy(d_d3,h_d3, allocSize, cudaMemcpyHostToDevice));

	dim3 gridDim1(HIDDEN_SIZE,1,1);
	dim3 blockDim1(VISIBLE_SIZE,1,1);

	//cout << "CPU Inputs" <<endl;//DEBUG
	//printVector(h_inputs, HIDDEN_SIZE);//DEBUG

	//cout <<"CPU W1" << endl;//DEBUG
    //printVector(h_W1,HIDDEN_SIZE);//DEBUG

    for(int i = 0;i < NUM_SAMPLE_ELEMENTS; i+= VISIBLE_SIZE)
    {
    	kernel_rho_forwardProp<<<gridDim1, blockDim1>>>(&d_inputs[i], d_W1, d_b1, d_rhoHat);
    	CUDA_SAFE_CALL(cudaPeekAtLastError());
    }

	//average rhoHat
	allocSize = HIDDEN_SIZE * sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpy(h_rhoHat,d_rhoHat, allocSize, cudaMemcpyDeviceToHost));
    vectElemFloatDiv(h_rhoHat, h_rhoHat, HIDDEN_SIZE, SAMPLE_SIZE);
    CUDA_SAFE_CALL(cudaMemcpy(d_rhoHat,h_rhoHat, allocSize, cudaMemcpyHostToDevice));

    dim3 gridDim2(VISIBLE_SIZE,1,1);
    dim3 blockDim2(HIDDEN_SIZE,1,1);

	dim3 gridDim3(1,1,1);
    dim3 blockDim3(VISIBLE_SIZE,1,1);

    dim3 gridDim4(HIDDEN_SIZE,1,1);
	dim3 blockDim4(VISIBLE_SIZE,1,1);

	dim3 gridDim5(1,1,1);
	dim3 blockDim5(HIDDEN_SIZE,1,1);

	dim3 gridDim6(HIDDEN_SIZE,1,1);
	dim3 blockDim6(VISIBLE_SIZE,1,1);

	dim3 gridDim7(HIDDEN_SIZE,1,1);
	dim3 blockDim7(VISIBLE_SIZE,1,1);

	
    for(int i = 0;i < NUM_SAMPLE_ELEMENTS; i+= VISIBLE_SIZE)
    {

    	//***************************************
      	//	FORWARD PROPAGATION a(1) --> a(2)				
    	//***************************************

    	kernel_forwardProp1<<<gridDim1, blockDim1>>>(&d_inputs[i], d_W1, d_b1, d_a2);
    	CUDA_SAFE_CALL(cudaPeekAtLastError());

    	//***************************************
      	//	FORWARD PROPAGATION a(2) --> a(3)			
    	//***************************************

    	kernel_forwardProp2<<<gridDim2, blockDim2>>>(&d_inputs[i], d_W2, d_b2, d_a3);
    	CUDA_SAFE_CALL(cudaPeekAtLastError());

		//***************************************
      	//	   BACK PROPAGATION d(3) --> d(2) 
    	//***************************************

      	kernel_backProp1<<<gridDim3, blockDim3>>>(&d_inputs[i], d_a3, d_d3, d_b2grad, d_cost);
      	CUDA_SAFE_CALL(cudaPeekAtLastError());

    	//***************************************
      	//	   BACK PROPAGATION d(2) --> input 
    	//***************************************

      	kernel_backProp2<<<gridDim4, blockDim4>>>(d_W2, d_d2, d_d3);
      	CUDA_SAFE_CALL(cudaPeekAtLastError());

		kernel_backProp3<<<gridDim5, blockDim5>>>(d_a2, d_d2, d_rhoHat, d_b1grad);
		CUDA_SAFE_CALL(cudaPeekAtLastError());

		kernel_backProp4<<<gridDim6, blockDim6>>>(&d_inputs[i], d_a2, d_d2, d_d3, d_W1grad, d_W2grad);
		CUDA_SAFE_CALL(cudaPeekAtLastError());
    }
    

    kernel_sparsityEnforcement<<<gridDim7, blockDim7>>>(d_W1, d_W2, d_W1grad, d_W2grad);
	CUDA_SAFE_CALL(cudaPeekAtLastError());

    allocSize = HIDDEN_SIZE * sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpy(h_a2,d_a2, allocSize, cudaMemcpyDeviceToHost));

    cout <<"GPU a2" << endl;//DEBUG
    printVector(h_a2,1);

    allocSize = VISIBLE_SIZE * sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpy(h_a3,d_a3, allocSize, cudaMemcpyDeviceToHost));

    cout <<"GPU a3" << endl;//DEBUG
    printVector(h_a3,1);

    allocSize = VISIBLE_SIZE * sizeof(float);
    CUDA_SAFE_CALL(cudaMemcpy(h_d3,d_d3, allocSize, cudaMemcpyDeviceToHost));
    cout << "GPU d3" << endl;//DEBUG
    printVector(h_d3,1);//DEBUG


#if PRINT_TIME
	// Stop and destroy the timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
	printf("\nGPU time: %f (sec)\n", elapsed_gpu*1000000/GIG);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
#endif

	/***************************************
			   FREEING HOST MEMORY
	****************************************/

    free(h_inputs);
    free(h_rhoHat);
    free(h_W1);
    free(h_W2);
    free(h_b1);
    free(h_b2);
    free(h_W1grad);
    free(h_W2grad);
    free(h_b1grad);
    free(h_b2grad);
    free(h_z2);
    free(h_z3);
    free(h_a2);
    free(h_a3);
    free(h_d2);
    free(h_d3);
    free(h_temp1);
    free(h_temp2);
    free(h_Wtemp1);
    free(h_Wtemp2);
    free(h_sparsePen);
    free(h_cost);

    /***************************************
			   FREEING DEVICE MEMORY
	****************************************/

   	CUDA_SAFE_CALL(cudaFree(d_inputs));
    CUDA_SAFE_CALL(cudaFree(d_rhoHat));
    CUDA_SAFE_CALL(cudaFree(d_W1));
    CUDA_SAFE_CALL(cudaFree(d_W2));
    CUDA_SAFE_CALL(cudaFree(d_b1));
    CUDA_SAFE_CALL(cudaFree(d_b2));
    CUDA_SAFE_CALL(cudaFree(d_W1grad));
    CUDA_SAFE_CALL(cudaFree(d_W2grad));
    CUDA_SAFE_CALL(cudaFree(d_b1grad));
    CUDA_SAFE_CALL(cudaFree(d_b2grad));
    CUDA_SAFE_CALL(cudaFree(d_a2));
    CUDA_SAFE_CALL(cudaFree(d_a3));
    CUDA_SAFE_CALL(cudaFree(d_d2));
    CUDA_SAFE_CALL(cudaFree(d_d3));
    CUDA_SAFE_CALL(cudaFree(d_sparsePen));
    CUDA_SAFE_CALL(cudaFree(d_cost));

    return 0;
}

/***********************************************
		TIMING FUNCTIONS AND STRUCTS
***********************************************/

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) 
  {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } 
  else 
  {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

void printTiming(struct timespec* time_stamp,int numTimings)
{
	
	for (int j = 0; j < numTimings; j++) 
	{
        if (j != 0) printf(", ");
        printf("\nCPU time: %f (sec)", ((double)(CPG)*(double)
            (GIG * time_stamp[j].tv_sec + time_stamp[j].tv_nsec)/GIG));
    }
    printf("\n");
}


/***********************************************
		NAIVE VECTOR OPERATIONS
***********************************************/

float sumVector(float* src,int length)
{
	float sum = 0;
	for(int i = 0;i < length;i++)
	{
		sum += src[i];
	}
	return sum;
}

void vectElemLog(float* src,float* dest,int length)
{
	for(int i = 0;i < length;i++)
	{
		dest[i] = log(src[i]);
	}
}

float normVector(float* src,int length)
{
	float x = 0;
	float sum = 0;
	for(int i = 0;i < length;i++)
	{
		x = src[i];
		sum += x*x;
	}
	sum = sqrt(sum);
	return sum;
}

void vectElemSigmoid(float* src,float* dest,int length)
{
	for(int i = 0; i < length;i++)
	{
		dest[i] = float(1/(1+exp(-src[i])));
	}
}

void vectElemVectMult(float* src1, float* src2, float* dest, int length)
{
	for(int i = 0; i < length;i++)
	{
		dest[i] = src1[i] * src2[i];
	}
}

//faster if float is used instead?
void vectElemIntDiv(float* src, float* dest,int length,int divisor)
{
	for(int i = 0;i < length;i++)
	{
		dest[i] = float(src[i]/divisor);
	}
}

void vectElemFloatDiv(float* src, float* dest,int length,float divisor)
{
	for(int i = 0;i < length;i++)
	{
		dest[i] = float(src[i]/divisor);
	}
}

void vectElemFloatMult(float* src, float* dest, int length,float multiplicand)
{
	for(int i = 0;i < length;i++)
	{
		dest[i] = src[i] * multiplicand;
	}
}

void vectElemVectDiv(float* src1,float* src2,float* dest,int length)
{
	for(int i = 0;i < length;i++)
	{
		dest[i] = (src1[i]/src2[i]);
	}
}

//Just for debugging eh?
void printVector(float* A, int length)
{
	for(int i = 0;i < length; i++)
	{
		cout << A[i] << endl;
	}
}

void initializeVector(float *array, int length, float val)
{
	for(int i = 0; i < length; i++)
	{
		array[i] = val;
	}
}

//Just for debugging eh?
void printMatrix(float* A, int rows, int cols)
{
	for(int i = 0;i < rows; i++)
	{
		for(int j = 0;j < cols;j++)
		{
			cout << A[i*rows+j] << "\t";
		}
		cout << endl;
	}
}

void addVectors(float* src1, float* src2, float* dest, int length)
{
	for(int i = 0;i < length; i++)
	{
		dest[i] = src1[i] + src2[i];
	}
}

void subVectors(float* src1, float* src2, float* dest, int length)
{
	for(int i = 0;i < length;i++)
	{
		dest[i] = src1[i] - src2[i];
	}
}

void dotPdt(float* src1,float* src2, float *dest, int length)
{
	float accum = 0;
	for(int i = 0; i< length;i++)
	{
		accum += src1[i] * src2[i];
	}
	*dest = accum;
	//cout << accum << endl;//DEBUG
}

void matrixTranspose(float* src,float* dest,int rows,int cols)
{
	for(int i = 0;i < rows;i++)
	{
		for(int j = 0;j < cols;j++)
		{
			//cout << src[i*rows+j] << "I: " << i << "J: " << j << endl;//DEBUG
			dest[j*rows+i] = src[i*cols+j];
		}
	} 	
}

void initializeMatrixWeightsRand(float *arr, int rows, int cols, int seed) 
{
    int i;
    float randNum, r;
    srand(seed);

    //rows and cols depend on hidden and visible sizes
    int numElements = rows*cols;

    for (i = 0; i < numElements; i++) 
    {
    	//Choose weights uniformly from the interval [-r, r]
        r = sqrt(6) / sqrt(rows+cols+1); 
        randNum = float(rand()%10000)/10000;
        randNum = randNum * 2 * r - r;
        arr[i] = randNum;
    }
}

void initializeMatrixWeightsZero(float *arr, int rows, int cols) 
{
    //rows and cols depend on hidden and visible sizes
    int numElements = rows*cols;

    for (int i = 0; i < numElements; i++) 
    {
        arr[i] = 0.0;
    }
}

//initialize the vector weights to 0
void initializeVectorWeightsZero(float *arr, int numElements)
{
	int i;
	for (i = 0; i < numElements; i++)
	{
		arr[i] = 0;
	}
}

/* mmm kij */ //BROKEN CURRENTLY
void mmm_kij(float* src1, float* src2, float* dest, int row1, int col1, int row2,int col2)
{

    float r = 0;

    for (int k = 0; k < row2; k++)
    {
        for (int i = 0; i < row1; i++) 
        {
            r = src1[i*col1+k];
            for (int j = 0; j < col2; j++)
            {
                dest[i*row1+j] += r*src2[k*row2+j];
            }
        }
    }
}

void mmm_ijk(float* src1, float* src2, float* dest, int row1, int col1, int row2, int col2)
{
	for(int i = 0;i < row1;i++)
	{
		for(int j = 0;j < col1;j++)
		{//or row2
			for(int k = 0;k < col2;k++)
			{
				dest[i*col2+k] += src1[i*col1+j] * src2[j*col2+k]; 
				//cout << "src1: " << src1[i*col1+j] << " src2: " << src2[j*col2+k] << endl;//DEBUG
				//cout << "I: " << i << " J: " << j << " K: " << k << endl;//DEBUG
			}
		}
	}
}


//http://www.cplusplus.com/forum/general/13087/
//http://www.cplusplus.com/forum/general/17771/
//http://www.cplusplus.com/forum/beginner/24906/
void readCSV(float* array, int numElements, string filename)
{
	
	ifstream infile(filename.c_str());
	int index = 0;

	if(infile){
		string line;
		while(getline(infile,line))
		{
			istringstream sep(line);
			string result;
			while(getline(sep, result,','))
			{
				array[index] = atof(result.c_str());
				if(array[index] == 0)
				{
					printf("\n %d", index);//DEBUG
				}
				index++;
			}
		}
	}
}

