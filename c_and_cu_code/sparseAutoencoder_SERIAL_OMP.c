/*******************************************************************
*			Sparse Auto-Encoder 
*					by
*		David Klaus and Alex Welles
*			EC527 Final Project	
*
*	Serial Implementation With Timing Code
*	
*	Compile with: 
*
*	gcc -fopenmp -o serialOMP  sparseAutoencoder_SERIAL_OMP.c -lm -lrt
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <omp.h>

#define GIG 1000000000
#define CPG 2.527
#define OPTIONS 1

//Parameters necessary to set up network
#define PATCHES_PATH	"c_patches.csv"
#define W1_PATH			"W1.csv"
#define W2_PATH			"W2.csv"
#define IMAGE_DIM		512		//pixels in 1 dimension (assumes square)
#define SAMPLE_SIZE		10000  //number of input patches
#define	HIDDEN_LAYERS 	1		//number hidden layers 
#define NUM_SAMPLE_ELEMENTS SAMPLE_SIZE * VISIBLE_SIZE

//desired average activation of hidden nodes
#define	SPARSITY_PARAM 	0.01
#define SPARSITY_COMPLEMENT 1-SPARSITY_PARAM
//weight decay paramater
#define	LAMBDA			0.0001
//weight of sparsity penalty term
#define	BETA			3.0

/* VECTOR FUNCTIONS */
void initializeMatrixWeightsRand(float *arr, int rows, int cols, int seed);
void initializeMatrixWeightsZero(float *arr, int rows, int cols);
void initializeVectorWeightsZero(float *arr, int numElements);
void initializeVector(float *array, int length, float val);
void initializeVectorRand(float *arr, int length, int seed, int HIDDEN_SIZE, int VISIBLE_SIZE);
void readCSV(float* array, char* filename);
/* PRINTOUT, DEBUG, AND TIMING FUNCTIONS */
void printVector(float* A, int length);
void printMatrix(float* A, int rows, int cols);
void printTiming(struct timespec* time_stamp,int numTimings);


int main(int argc, char *argv[]){

	int VISIBLE_SIZE;
	int HIDDEN_SIZE;
 	sscanf (argv[1],"%d",&VISIBLE_SIZE);
 	sscanf (argv[2],"%d",&HIDDEN_SIZE);

	/***********************************
	 		  TIMING STUFF
    ***********************************/

	struct timespec diff(struct timespec start, struct timespec end);
    struct timespec time1, time2;
    struct timespec time_stamp[OPTIONS];//Can be increased if necessary.
    int OPTION = 0;//default is zero. Placeholder in case.

	/***********************************
	 		ALLOCATE MEMORY
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
	//a product vectors
	float *h_a2;//hidden x 1 [25 x 1]
	float *h_a3;//visible x 1 [64 x 1]
	//partial derivatives for back prop
	float *h_d2;//hidden x 1 [25 x 1]
	float *h_d3;//visible x 1 [64 x 1]

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

	/***********************************
	 	INITIALIZE NETWORK WEIGHTS
    ***********************************/

	//Initialize the weight matrices to random values
	initializeMatrixWeightsRand(h_W1, HIDDEN_SIZE, VISIBLE_SIZE, 2254);
	initializeMatrixWeightsRand(h_W2, VISIBLE_SIZE, HIDDEN_SIZE, 1345);
	initializeVectorRand(h_inputs, NUM_SAMPLE_ELEMENTS, 9999, HIDDEN_SIZE, VISIBLE_SIZE);
	initializeMatrixWeightsZero(h_W2grad,VISIBLE_SIZE,HIDDEN_SIZE);
	initializeMatrixWeightsZero(h_W1grad,HIDDEN_SIZE,VISIBLE_SIZE);

	initializeVectorWeightsZero(h_b1, HIDDEN_SIZE);
	initializeVectorWeightsZero(h_b2, VISIBLE_SIZE);
	initializeVectorWeightsZero(h_rhoHat, HIDDEN_SIZE);
	initializeVectorWeightsZero(h_a2, HIDDEN_SIZE);
	initializeVectorWeightsZero(h_a3, VISIBLE_SIZE);

	/***********************************
	 	   READ IN SAMPLE PATCHES
    ***********************************/

	readCSV(h_inputs, PATCHES_PATH);
	//the following are for debug only
	readCSV(h_W1, W1_PATH);
	readCSV(h_W2, W2_PATH);

	/***************************************
			   BEGIN SERIAL TIMING
	****************************************/

	printf("\nStarting Serial OMP Timing");
    clock_gettime(CLOCK_REALTIME, &time1);

	/***************************************
      CALCULATE SPARSITY CONSTRAINT RHO-HAT
    ****************************************/

    int i, j, k;
    float accum1, accum2, accum3, cost;
    
    omp_set_num_threads(2);

    #pragma omp parallel shared(h_inputs, h_W1, h_b1, h_rhoHat) private(i, j, k, accum1)
    {
    	#pragma omp for
		for( i = 0; i < NUM_SAMPLE_ELEMENTS; i += VISIBLE_SIZE )
		{
			
			for( j = 0; j < HIDDEN_SIZE; j++ )
			{
				accum1 = 0;
				for( k = 0; k < VISIBLE_SIZE ; k++ )
				{
					accum1 += h_inputs[ i + k ] * h_W1[ j * VISIBLE_SIZE + k ];
				}
				#pragma omp critical
				h_rhoHat[ j ] += (float)(1/(1+exp(-accum1 - h_b1[ j ])));
			}
		}
	}
	
	for( i = 0; i < HIDDEN_SIZE; i++ )
	{
		h_rhoHat[ i ] = h_rhoHat[ i ] / SAMPLE_SIZE;
	}

	//printVector(h_rhoHat, HIDDEN_SIZE);//DEBUG

	//***************************************
    // 	FORWARD AND BACKWARD PROPAGATION
    //****************************************

    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);//DEBUG TIMING



    for( i = 0; i < NUM_SAMPLE_ELEMENTS; i += VISIBLE_SIZE)
    {
    	#pragma omp parallel shared(h_inputs, h_W1, h_b1, h_a2) private(i, j, k, accum1)
    	{	
	    	#pragma omp for
	    	for( j = 0;j < HIDDEN_SIZE; j++ )
	    	{
				accum1 = 0;
				for( k = 0; k < VISIBLE_SIZE ; k++ )
				{
					accum1 += h_inputs[ i + k ] * h_W1[ j * VISIBLE_SIZE + k ];
				}
				h_a2[j] = (float)(1/(1+exp(-accum1 - h_b1[j])));
			}
		}

		//***************************************
      	//	FORWARD PROPAGATION a(2) --> a(3)				
    	//****************************************

    	//clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);//DEBUG TIMING
    	#pragma omp parallel shared(h_inputs, h_W2, h_a2, h_a3, h_b2) private(i, j, k, accum1)
    	{	
    		#pragma omp for
			for( j = 0; j < VISIBLE_SIZE; j++ )
			{
				accum1 = 0;
				for( k = 0; k < HIDDEN_SIZE; k++ )
				{
					accum1 += h_a2[ k ] * h_W2[ j * HIDDEN_SIZE + k ];
				}
				//#pragma omp critical
				h_a3[j] = (float)(1/(1+exp(-accum1 - h_b2[j])));
			}
		}
		
	//}//DEBUG TEMP BRACKET
	//printVector(h_a3, 1);//DEBUG

		//***************************************
      	//	   BACK PROPAGATION d(3) --> d(2) 
    	//****************************************

		//omp_set_num_threads(8);
		#pragma omp parallel shared(h_inputs, h_d3, h_a3) private(i, j, k, accum1)
    	{	
    		#pragma omp for
	    	for( j = 0; j < VISIBLE_SIZE; j++ )
	    	{
	    		accum1 = h_a3[ j ];
	    		//#pragma omp critical
	    		h_d3[ j ] = ( accum1 - h_inputs[ i + j ] ) * ( accum1 * ( 1 - accum1 ) );
	    	}
	    }

	    //***************************************
      	//	   BACK PROPAGATION d(2) --> input 
    	//****************************************

	    #pragma omp parallel shared( h_W2, h_rhoHat, h_a2, h_b1, h_d3) private(i, j, k, accum1, accum2, accum3)
    	{	
    		#pragma omp for
	    	for( j = 0; j < HIDDEN_SIZE; j++ )
	    	{
	    		accum1 = 0;
	    		for( k = 0; k < VISIBLE_SIZE; k++ )
				{
					accum1 += h_d3[ k ] * h_W2[ j * VISIBLE_SIZE + k ];
				}

				//d2 = ((W2t * d3) + beta .* (-(sparsityParam./rhoHat) + (1-sparsityParam)./(1-rhoHat)))
							//	 .* (a2 .* (1 - a2));	
				accum2 = h_rhoHat[ j ];
				accum3 = h_a2[ j ];
				h_d3[ j ] = (accum1 + BETA * ( -SPARSITY_PARAM / accum2 ) + ( 1 - SPARSITY_PARAM ) / (1 - accum2 )) 
								* ( accum3 * ( 1 - accum3 ));
	    	}
    	}

	//}//DEBUG TEMP BRACKET	
	//printVector(h_d3, 1);//DEBUG

	    //***************************************
      	//	   AGGREGATE PARTIAL DERIVATIVES 
    	//****************************************

		#pragma omp parallel shared( h_d3, h_W2grad, h_a2, h_b2grad) private(i, j, k, accum1, accum2)
    	{	
	    	#pragma omp for
	    	for( j = 0; j < VISIBLE_SIZE; j++ )
	    	{
	    		accum1 = j * HIDDEN_SIZE;
	    		accum2 = h_d3[ j ];
	    		for( k = 0; k < HIDDEN_SIZE; k++ )
				{
					h_W2grad[ (int)accum1 + k ] +=  accum2 * h_a2[ k ]; 
				}
				h_b2grad[ j ] += accum2;
	    	}
    	}

    	#pragma omp parallel shared( h_d2, h_W1grad, h_inputs, h_b1grad) private(i, j, k, accum1, accum2)
    	{	
	    	#pragma omp for
	    	for( j = 0; j < HIDDEN_SIZE; j++ )
	    	{
	    		accum1 = j * VISIBLE_SIZE;
	    		accum2 = h_d2[ j ];
	    		for( k = 0; k < VISIBLE_SIZE; k++ )
	    		{
	    			h_W1grad[ (int)accum1 + k ] += accum2 * h_inputs[ i + k ];
	    		}
	    		h_b1grad[ j ] += accum2;
	    	}
    	}

	    //cost = cost + norm(a3 - xM)^2; 
	    for( j = 0; j < VISIBLE_SIZE; j++ )
	    {
	    	accum1 = h_a3[ j ] - h_inputs[ i + j ];
	    	cost += accum1 * accum1;
	    }   
    }

    accum1 = 0;
    accum2 = 0;

    for( i = 0; i < VISIBLE_SIZE * HIDDEN_SIZE; i++ )
    {
    	h_W2grad[ i ] = h_W2grad[ i ] / SAMPLE_SIZE + LAMBDA * h_W2[ i ];
    	h_W1grad[ i ] = h_W1grad[ i ] / SAMPLE_SIZE + LAMBDA * h_W1[ i ];
    	//accum1 will hold sum of W1's weights
    	//accum2 will hold sum of W2's weights
    	accum1 += h_W1[ i ];
    	accum2 += h_W2[ i ];
    }

    for( i = 0; i < VISIBLE_SIZE; i++ )
    {
    	h_b2grad[ i ] = h_b2grad[ i ] / SAMPLE_SIZE;
    }

    for( i = 0; i < HIDDEN_SIZE; i++ )
    {
    	h_b1grad[ i ] = h_b1grad[ i ] / SAMPLE_SIZE;
    }

	//sparsePen = sparsityParam .* log(sparsityParam./rhoHat) + (1-sparsityParam).*log((1-sparsityParam)./(1-rhoHat));
    accum3 = 0; //WILL HOLD SPARSITY PENALTY
    for( i = 0; i < HIDDEN_SIZE; i++ ){
    	accum1 += SPARSITY_PARAM * log( SPARSITY_PARAM / h_rhoHat[ i ] ) + SPARSITY_COMPLEMENT 
    							* log(( SPARSITY_COMPLEMENT / ( 1 - h_rhoHat[ i ])));
    }	

	//cost = (cost / (2 * M)) + (lambda / 2 ) * (sum(sum(W1.^2)) + (sum(sum(W2.^2)))) + beta * sum(sparsePen); 
	cost = ( cost / (2 * SAMPLE_SIZE ))  + ( LAMBDA / 2 ) * ( accum1 + accum2 ) + BETA * accum3;

	//***************************************
	//     END AND PRINT SERIAL TIMING
	//****************************************/

    clock_gettime(CLOCK_REALTIME, &time2);
    time_stamp[OPTION] = diff(time1,time2);
	printTiming(time_stamp,OPTIONS);

    /***************************************
			   DEBUG OUTPUTS
	****************************************/
	
   printf("\nrhoHat");
   printVector(h_rhoHat, 1);
   printf("\na2");
   printVector(h_a2, 1);
   printf("\na3");
   printVector(h_a3, 1);
   printf("\nd3");
   printVector(h_d3, 1);


	/***************************************
			   FREEING MEMORY
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
    free(h_a2);
    free(h_a3);
    free(h_d2);
    free(h_d3);

    return 0;
}

/***********************************************
		TIMING FUNCTIONS AND STRUCTS
***********************************************/

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}

void printTiming(struct timespec* time_stamp,int numTimings){
	int j = 0;
	for ( j = 0; j < numTimings; j++) {
        if (j != 0) printf(", ");
        printf("\nCPU time: %f (sec)", ((double)(CPG)*(double)
            (GIG * time_stamp[j].tv_sec + time_stamp[j].tv_nsec)/GIG));
    }
    printf("\n");
}


/***********************************************
		NAIVE VECTOR OPERATIONS
***********************************************/

//Just for debugging eh?
void printVector(float* A, int length){
	int i = 0;
	printf("\n");
	for( i = 0;i < length; i++){
		printf("%f\n",A[i]);
	}
}

void initializeVector(float *array, int length, float val){
	int i = 0;
	for( i = 0; i < length; i++){
		array[i] = val;
	}
}

//Just for debugging eh?
void printMatrix(float* A, int rows, int cols){
	int i = 0;
	int j = 0;
	printf("\n");
	for( i = 0;i < rows; i++){
		for( j = 0;j < cols;j++){
			printf("%f\t",A[i*rows+j]);
		}
		printf("\n");
	}
}

void initializeMatrixWeightsRand(float *arr, int rows, int cols, int seed) {
    int i;
    float randNum, r;
    srand(seed);

    //rows and cols depend on hidden and visible sizes
    int numElements = rows*cols;

    for (i = 0; i < numElements; i++) {
    	//Choose weights uniformly from the interval [-r, r]
        r = sqrt(6) / sqrt(rows+cols+1); 
        randNum = (float)(rand()%10000)/10000;
        randNum = randNum * 2 * r - r;
        arr[i] = randNum;
    }
}

void initializeVectorRand(float *arr, int length, int seed, int HIDDEN_SIZE, int VISIBLE_SIZE)
{
	int i;
	float r;
	float randNum;
	srand(seed);
	for (i = 0; i < length; i++) {
    	//Choose weights uniformly from the interval [-r, r]
        r = sqrt(6) / sqrt(HIDDEN_SIZE+VISIBLE_SIZE+1); 
        randNum = (float)(rand()%10000)/10000;
        randNum = randNum * 2 * r - r;
        arr[i] = randNum;
    }
}

void initializeMatrixWeightsZero(float *arr, int rows, int cols) {
    //rows and cols depend on hidden and visible sizes
    int numElements = rows*cols;
    int i = 0;
    for ( i = 0; i < numElements; i++) {
        arr[i] = 0.0;
    }
}

//initialize the vector weights to 0
void initializeVectorWeightsZero(float *arr, int numElements){
	int i;
	for (i = 0; i < numElements; i++){
		arr[i] = 0;
	}
}

//http://stackoverflow.com/questions/20013693/read-csv-file-to-a-2d-array-on-c
//http://www.cplusplus.com/reference/cstdlib/strtof/
void readCSV(float* array, char* filename)
{
	char buffer[100];
	
	FILE *fstream = fopen(filename,"r");
	
	int index = 0;

	while(fgets(buffer,sizeof(buffer),fstream) !=NULL)
	{	
		array[index] = strtof(buffer,NULL);
		index++;
	}
}








