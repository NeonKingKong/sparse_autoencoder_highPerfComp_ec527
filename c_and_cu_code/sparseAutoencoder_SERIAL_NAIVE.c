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
*	gcc -o serialNAIVE sparseAutoencoder_SERIAL_NAIVE.c -lm -lrt
*
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

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
void mmm_kij(float* src1, float* src2, float* dest, int row1, int col1, int row2,int col2);
void mmm_ijk(float* src1, float* src2, float* dest, int row1, int col1, int row2,int col2);
void dotPdt(float* src1,float* src2, float* dest, int length);
void addVectors(float* src1, float* src2, float* dest, int length);
void subVectors(float* src1, float* src2, float* dest, int length);
void vectElemSigmoid(float* src,float* dest,int length);
void vectElemIntDiv(float* src, float* dest,int length,int divisor);
void vectElemFloatDiv(float* src, float* dest,int length,float divisor);
void vectElemVectDiv(float* src1,float* src2,float* dest,int length);
void initializeVector(float *array, int length, float val);
void initializeVectorRand(float *arr, int length, int seed, int HIDDEN_SIZE, int VISIBLE_SIZE);
void vectElemVectMult(float* src1, float* src2, float* dest, int length);
void vectElemFloatMult(float* src, float* dest, int length,float multiplicand);
void matrixTranspose(float* src,float* dest,int rows, int cols);
float normVector(float* src,int length);
void vectElemLog(float* src,float* dest,int length);
float sumVector(float* src,int length);
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

	printf("\nStarting Serial Timing");
	struct timespec diff(struct timespec start, struct timespec end);
    struct timespec time1, time2;
    struct timespec time_stamp[OPTIONS];//Can be increased if necessary.

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

	/***********************************
	 			VARIABLES
    ***********************************/
		
	float cost = 0;

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
	initializeVectorWeightsZero(h_z2, HIDDEN_SIZE);
	initializeVectorWeightsZero(h_a2, HIDDEN_SIZE);
	initializeVectorWeightsZero(h_z3, VISIBLE_SIZE);
	initializeVectorWeightsZero(h_a3, VISIBLE_SIZE);


	/***********************************
	 	   READ IN SAMPLE PATCHES
    ***********************************/

	//readCSV(h_inputs, PATCHES_PATH);
	//the following are for debug only
	//readCSV(h_W1, W1_PATH);
	//readCSV(h_W2, W2_PATH);


	/***************************************
			   BEGIN SERIAL TIMING
	****************************************/

    printf("\nTesting Baseline Sparse Autoencoder");
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
    int OPTION = 0;//default is zero. Placeholder in case.

	/***************************************
      CALCULATE SPARSITY CONSTRAINT RHO-HAT
    ****************************************/

    /* 
    This involves forward propagation through 10000 64 pixel values 
    (note values are stored as floats). These are stored as a 64000
    element vector. Another forward propagation is performed later.
	*/

    int i, j;
    
	for( i = 0; i < NUM_SAMPLE_ELEMENTS; i+=VISIBLE_SIZE)
	{//for number of input patches
		//we are performing the following matrix op
		//[hsize x vsize] * [vsize x 1] = [vsize x 1]
		//[25x64] * [64x1] = [64x1]
		//using 25 dot product calls
		for( j = 0;j < HIDDEN_SIZE;j++ )
		{

			dotPdt(&h_inputs[i],&h_W1[j*VISIBLE_SIZE],&h_z2[j],VISIBLE_SIZE);
			//h_z2[j] = altDotPdt(&h_inputs[i],&h_W1[j*VISIBLE_SIZE],VISIBLE_SIZE);
			// &h_inputs[i] 				--> pointer to current input patch
			// &h_W1[j*VISIBLE_SIZE] 		--> pointer to current row of the W1 weights
			// &h_z2[j]						--> pointer to current output element of z1
			// VISIBLE_SIZE					--> patch size, should be 64
		}
		//[25x1] + [25x1] = [25x1]
		//[hidden x 1] + [hidden x1] = [hidden x 1]
		addVectors(h_z2,h_b1,h_z2, HIDDEN_SIZE);
		//sigma([25x1]) = [25x1] 
		vectElemSigmoid(h_z2,h_a2,HIDDEN_SIZE);
		//calc the sparsity constraint
		addVectors(h_rhoHat,h_a2,h_rhoHat,HIDDEN_SIZE);
	}

	vectElemIntDiv(h_rhoHat,h_rhoHat,HIDDEN_SIZE,SAMPLE_SIZE);


	/***************************************
      	FORWARD AND BACKWARD PROPAGATION
    ****************************************/

    for( i = 0; i < NUM_SAMPLE_ELEMENTS; i+=VISIBLE_SIZE){
    	
    	/***************************************
      		FORWARD PROPAGATION a(1) --> a(2)				//DEBUG PASSED
    	****************************************/

      	/*
      	%forward propagate
	    %from a1 = input to a2
	    xM = data(:,i);%current input
	    z2 = W1 * xM + b1;
	    a2 = 1./ ( 1 + exp(-z2));
    	*/

    	//we are performing the following matrix op
		//[hsize x vsize] * [vsize x 1] = [vsize x 1]
		//[25x64] * [64x1] = [64x1]
		//using 25 dot product calls
    	for( j = 0;j < HIDDEN_SIZE;j++ )
    	{

			dotPdt(&h_inputs[i],&h_W1[j*VISIBLE_SIZE],&h_z2[j],VISIBLE_SIZE);
			//h_z2[j] = altDotPdt(&h_inputs[i],&h_W1[j*VISIBLE_SIZE],VISIBLE_SIZE);
			// &h_inputs[i] 				--> pointer to current input patch
			// &h_W1[j*VISIBLE_SIZE] 		--> pointer to current row of the W1 weights
			// &h_z2[j]						--> pointer to current output element of z1
			// VISIBLE_SIZE					--> patch size, should be 64
		}
		//[25x1] + [25x1] = [25x1]
		//[hidden x 1] + [hidden x1] = [hidden x 1]
		addVectors(h_z2,h_b1,h_z2, HIDDEN_SIZE);
		//sigma([25x1]) = [25x1] 
		vectElemSigmoid(h_z2,h_a2,HIDDEN_SIZE);


		/***************************************
      		FORWARD PROPAGATION a(2) --> a(3)				
    	****************************************/

      	/*
      	%from a2 to a3 = output
	    z3 = W2 * a2 + b2;
	    a3 = 1./ (1 + exp(-z3));
		*/

		//we are performing the following matrix op
		//[visible x hidden] * [hidden x 1] = [visible x 1]
		//[64x25] * [25x1] = [64x1]
		//using 64 dot product calls

		for( j = 0;j < VISIBLE_SIZE;j++ )
		{
			//the key difference here is that we are using h_a2 as the input
			dotPdt(h_a2,&h_W2[j*HIDDEN_SIZE],&h_z3[j],VISIBLE_SIZE);//  W2 * a2 --> h_z3
			//h_z2[j] = altDotPdt(&h_inputs[i],&h_W1[j*VISIBLE_SIZE],VISIBLE_SIZE);
			// h_a2 						--> pointer to current input vector
			// &h_W1[j*VISIBLE_SIZE] 		--> pointer to current row of the W1 weights
			// &h_z2[j]						--> pointer to current output element of z1
		}

		//[64x1] + [64x1] = [64x1]
		//[visible x 1] + [visible x1] = [visible x 1]
		addVectors(h_z3,h_b2,h_z3,VISIBLE_SIZE);
		//sigma([64x1]) = [64x1] 
		vectElemSigmoid(h_z3,h_a3,VISIBLE_SIZE);


		/***************************************
      		   BACK PROPAGATION d(3) --> d(2) 
    	****************************************/
      	/*
      	%back propagate
	    %a3 -> a2
	    d3 = -(xM - a3) .* (a3 .* (1 - a3));
      	*/

	    //d3 = -(xM - a3) .* (a3 .* (1 - a3));
	    initializeVector(h_temp1, VISIBLE_SIZE, 1.0);//vec(1) --> h_temp1	    
	    subVectors(h_temp1,h_a3,h_temp1,VISIBLE_SIZE);//(1 - a3) --> h_temp1
	    vectElemVectMult(h_a3,h_temp1,h_temp1,VISIBLE_SIZE);//(a3 .* (1 - a3)) --> h_temp2
	    subVectors(h_a3,&h_inputs[i],h_temp2,VISIBLE_SIZE);//-(xM - a3) = (a3 - xM) --> h_temp2
	    vectElemVectMult(h_temp1,h_temp2,h_d3,VISIBLE_SIZE);//d3 = -(xM - a3) .* (a3 .* (1 - a3)); --> h_d3
	    //initializeVector(h_temp1,VISIBLE_SIZE,-1);
	    //vectElemVectMult(h_temp1,h_d3,h_d3,VISIBLE_SIZE);
	    
	    /***************************************
      		   BACK PROPAGATION d(2) --> input 
    	****************************************/

	    //d2 = ((W2' * d3) + beta .* (-(sparsityParam./rhoHat) + (1-sparsityParam)./(1-rhoHat))) .* (a2 .* (1 - a2));
	   
 	    //Note the h_temp1 & 2 vectors are actually VISIBLE_SIZE long
	    //but we are only using HIDDEN_SIZE values
	    initializeVector(h_temp2, HIDDEN_SIZE, 1.0);//(1)
	    subVectors(h_temp2,h_rhoHat,h_temp2,HIDDEN_SIZE);//(1-rhoHat) --> h_temp2
	    initializeVector(h_temp1, HIDDEN_SIZE, SPARSITY_COMPLEMENT);//(1-sparsityParam) --> h_temp1
	    vectElemVectDiv(h_temp1,h_temp2,h_temp2,HIDDEN_SIZE);//(1-sparsityParam)./(1-rhoHat) --> h_temp2

	  
	    initializeVector(h_temp1,HIDDEN_SIZE,-1.0*SPARSITY_PARAM);//-(sparsityParam) --> h_temp1
	    vectElemVectDiv(h_temp1,h_rhoHat,h_temp1,HIDDEN_SIZE);//-(sparsityParam./rhoHat) --> h_temp1

	    addVectors(h_temp1,h_temp2,h_temp1,HIDDEN_SIZE);//(-(sparsityParam./rhoHat) + (1-sparsityParam)./(1-rhoHat)) --> h_temp1
	    vectElemFloatMult(h_temp1,h_temp1,HIDDEN_SIZE,BETA);//beta .* (-(sparsityParam./rhoHat) + (1-sparsityParam)./(1-rhoHat))) --> h_temp1

	    matrixTranspose(h_W2,h_Wtemp1,VISIBLE_SIZE,HIDDEN_SIZE);//W2' --> h_Wtemp1
	    for( j = 0;j < VISIBLE_SIZE;j++)
	    {//(W2' * d3) --> h_temp2
	    	dotPdt(h_d3,&h_Wtemp1[j*HIDDEN_SIZE],&h_temp2[j],VISIBLE_SIZE);
	    }

	    //((W2' * d3) + beta .* (-(sparsityParam./rhoHat) + (1-sparsityParam)./(1-rhoHat))) --> h_temp2
	    addVectors(h_temp2,h_temp1,h_temp2,HIDDEN_SIZE);
	    initializeVector(h_temp1, HIDDEN_SIZE, 1);//1-a2 --> h_temp1
	    subVectors(h_temp1,h_a2,h_temp1,HIDDEN_SIZE);//(1-a2) --> h_temp1
	    vectElemVectMult(h_a2,h_temp1,h_temp1,HIDDEN_SIZE);//(a2 .* (1 - a2)) --> h_temp1
	    //d2 = ((W2' * d3) + beta .* (-(sparsityParam./rhoHat) + (1-sparsityParam)./(1-rhoHat))) .* (a2 .* (1 - a2));--> h_d2
	    vectElemVectMult(h_temp1,h_temp2,h_d2,HIDDEN_SIZE);

	    /*
		%compute partial derivatives
	    W2grad = W2grad + d3 * a2';
	    b2grad = b2grad + d3;
	    W1grad = W1grad + d2 * xM';
	    b1grad = b1grad + d2; 
	    */

	    mmm_ijk(h_d3,h_a2,h_Wtemp1,VISIBLE_SIZE,1,1,HIDDEN_SIZE);//d3 * a2' --> h_Wtemp1
	    addVectors(h_W2grad,h_Wtemp1,h_W2grad,HIDDEN_SIZE*VISIBLE_SIZE);//W2grad + d3 * a2'; --> h_W2grad
	    addVectors(h_b2grad,h_d3,h_b2grad,VISIBLE_SIZE);//b2grad = b2grad + d3;
	    mmm_ijk(h_d2,&h_inputs[i],h_Wtemp1,HIDDEN_SIZE,1,1,VISIBLE_SIZE);//d2 * xM' --> h_Wtemp1
	    addVectors(h_W1grad,h_Wtemp1,h_W1grad,HIDDEN_SIZE*VISIBLE_SIZE);//W1grad = W1grad + d2 * xM'; --> h_W1grad
	    addVectors(h_b1grad,h_d2,h_b1grad,HIDDEN_SIZE);//b1grad = b1grad + d2; --> h_b1grad
 
	    /*
	    %for calculating cost
	    cost = cost + norm(a3 - xM)^2; 
		*/

	    subVectors(h_a3,&h_inputs[i],h_temp1,VISIBLE_SIZE);//(a3 - xM) --> h_temp1
		float temp = 0;
		temp = normVector(h_temp1,VISIBLE_SIZE);//norm(a3 - xM)^2; --> temp
		cost += temp*temp;//cost = cost + norm(a3 - xM)^2; --> cost
    }

	/*
    %W1grad = [(1/m) \Delta W^{(1)} + \lambda W^{(1)}]
	W2grad = W2grad ./ M + lambda .* W2;
	b2grad = b2grad ./ M;
	W1grad = W1grad ./ M + lambda .* W1;
	b1grad = b1grad ./ M;
	*/

	//NOTE: M = SAMPLE_SIZE = 10000 by default

	//W2grad = W2grad ./ M + lambda .* W2;
	vectElemFloatDiv(h_W2grad,h_W2grad,VISIBLE_SIZE*HIDDEN_SIZE,SAMPLE_SIZE);//W2grad ./ M  --> h_W2grad
	vectElemFloatMult(h_W2,h_Wtemp1,VISIBLE_SIZE*HIDDEN_SIZE,LAMBDA);// lambda .* W2; --> h_Wtemp1
	addVectors(h_W2grad,h_Wtemp1,h_W2grad,VISIBLE_SIZE*HIDDEN_SIZE);//W2grad = W2grad ./ M + lambda .* W2; --> h_W2grad
	//b2grad = b2grad ./ M;
	vectElemFloatDiv(h_b2grad,h_b2grad,VISIBLE_SIZE,SAMPLE_SIZE);//b2grad = b2grad ./ M; --> h_b2grad
	//W1grad = W1grad ./ M + lambda .* W1;
	vectElemFloatDiv(h_W2grad,h_W2grad,VISIBLE_SIZE*HIDDEN_SIZE,SAMPLE_SIZE);//W1grad ./ M  --> h_W2grad
	vectElemFloatMult(h_W2,h_Wtemp1,VISIBLE_SIZE*HIDDEN_SIZE,LAMBDA);// lambda .* W1; --> h_Wtemp1
	addVectors(h_W2grad,h_Wtemp1,h_W2grad,VISIBLE_SIZE*HIDDEN_SIZE);//W1grad = W1grad ./ M + lambda .* W1; --> h_W1grad
	//b1grad = b1grad ./ M;
	vectElemFloatDiv(h_b1grad,h_b1grad,HIDDEN_SIZE,SAMPLE_SIZE);//b1grad = b1grad ./ M; --> h_b1grad

	/*
	%rho
	sparsePen = sparsityParam .* log(sparsityParam./rhoHat) + (1-sparsityParam).*log((1-sparsityParam)./(1-rhoHat));
	*/

	initializeVector(h_temp1,HIDDEN_SIZE,1.0);//1--> h_temp1
	subVectors(h_temp1,h_rhoHat,h_temp2,HIDDEN_SIZE);//(1-rhoHat) --> h_temp2
	initializeVector(h_temp1, HIDDEN_SIZE, SPARSITY_COMPLEMENT);//(1-sparsityParam) --> h_temp1
	vectElemVectDiv(h_temp1,h_temp2,h_temp2,HIDDEN_SIZE);//(1-sparsityParam)./(1-rhoHat) --> h_temp2
	vectElemLog(h_temp2,h_temp2,HIDDEN_SIZE);//log((1-sparsityParam)./(1-rhoHat)) --> h_temp2
	vectElemVectMult(h_temp1,h_temp2,h_temp2,HIDDEN_SIZE);//(1-sparsityParam).*log((1-sparsityParam)./(1-rhoHat)) --> h_temp2
	initializeVector(h_temp1,HIDDEN_SIZE,SPARSITY_PARAM);//sparsityParam --> h_temp1
	vectElemVectDiv(h_temp1,h_rhoHat,h_temp1,HIDDEN_SIZE);//(sparsityParam./rhoHat) --> h_temp1
	vectElemLog(h_temp1,h_temp1,HIDDEN_SIZE);//log(sparsityParam./rhoHat) --> h_temp1
	vectElemFloatMult(h_temp1,h_temp1,HIDDEN_SIZE,SPARSITY_PARAM);//sparsityParam .* log(sparsityParam./rhoHat)--> h_temp1
	addVectors(h_temp1,h_temp2,h_sparsePen,HIDDEN_SIZE);//sparsePen = --> h_sparsePen

	/*
	cost = (cost / (2 * M)) + (lambda / 2 ) * (sum(sum(W1.^2)) + (sum(sum(W2.^2)))) + beta * sum(sparsePen); 
	*/

	vectElemVectMult(h_W1,h_W1,h_Wtemp1,VISIBLE_SIZE*HIDDEN_SIZE);//W1.^2 --> h_Wtemp1
	float sumW1 = sumVector(h_Wtemp1,VISIBLE_SIZE*HIDDEN_SIZE);//sum(sum(W1.^2)) --> sumW1
	vectElemVectMult(h_W2,h_W2,h_Wtemp2,VISIBLE_SIZE*HIDDEN_SIZE);//W2.^2 --> h_Wtemp2
	float sumW2 = sumVector(h_Wtemp2,VISIBLE_SIZE*HIDDEN_SIZE);// (sum(sum(W2.^2))) --> sumW2
	
	cost = (cost/(2*SAMPLE_SIZE)) + (LAMBDA/2)*(sumW1 + sumW2) + BETA*sumVector(h_sparsePen,HIDDEN_SIZE);

	/***************************************
	     END AND PRINT SERIAL TIMING
	****************************************/

    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
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

float sumVector(float* src,int length){
	float sum = 0;
	int i = 0;
	for( i = 0;i < length;i++){
		sum += src[i];
	}
	return sum;
}

void vectElemLog(float* src,float* dest,int length){
	int i = 0;
	for( i = 0;i < length;i++){
		dest[i] = log(src[i]);
	}
}

float normVector(float* src,int length){
	float x = 0;
	float sum = 0;
	int i = 0;
	for( i = 0;i < length;i++){
		x = src[i];
		sum += x*x;
	}
	sum = sqrt(sum);
	return sum;
}

void vectElemSigmoid(float* src,float* dest,int length){
	int i = 0;
	for( i = 0; i < length;i++){
		dest[i] = (float)(1/(1+exp(-src[i])));
	}
}

void vectElemVectMult(float* src1, float* src2, float* dest, int length){
	int i = 0;
	for( i = 0; i < length;i++){
		dest[i] = src1[i] * src2[i];
	}
}

//faster if float is used instead?
void vectElemIntDiv(float* src, float* dest,int length,int divisor){
	int i = 0;
	for( i = 0;i < length;i++){
		dest[i] = (float)(src[i]/divisor);
	}
}

void vectElemFloatDiv(float* src, float* dest,int length,float divisor){
	int i = 0;
	for( i = 0;i < length;i++){
		dest[i] = (float)(src[i]/divisor);
	}
}

void vectElemFloatMult(float* src, float* dest, int length,float multiplicand){
	int i = 0;
	for( i = 0;i < length;i++){
		dest[i] = src[i] * multiplicand;
	}
}

void vectElemVectDiv(float* src1,float* src2,float* dest,int length){
	int i = 0;
	for( i = 0;i < length;i++){
		dest[i] = (src1[i]/src2[i]);
	}
}

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

void addVectors(float* src1, float* src2, float* dest, int length){
	int i = 0;
	for( i = 0;i < length; i++){
		dest[i] = src1[i] + src2[i];
	}
}

void subVectors(float* src1, float* src2, float* dest, int length){
	int i = 0;
	for( i = 0;i < length;i++){
		dest[i] = src1[i] - src2[i];
	}
}

void dotPdt(float* src1,float* src2, float *dest, int length){
	float accum = 0;
	int i = 0;
	for( i = 0; i< length;i++){
		accum += src1[i] * src2[i];
	}
	*dest = accum;
}

void matrixTranspose(float* src,float* dest,int rows,int cols){
	int i = 0;
	int j = 0;
	for( i = 0;i < rows;i++){
		for( j = 0;j < cols;j++){
			dest[j*rows+i] = src[i*cols+j];
		}
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


void mmm_ijk(float* src1, float* src2, float* dest, int row1, int col1, int row2, int col2){
	
	int i = 0;
	int j = 0;
	int k = 0;
	for( i = 0;i < row1;i++){
		for( j = 0;j < col1;j++){//or row2
			for( k = 0;k < col2;k++){
				dest[i*col2+k] += src1[i*col1+j] * src2[j*col2+k]; 
			}
		}
	}
}

//http://stackoverflow.com/questions/20013693/read-csv-file-to-a-2d-array-on-c
void readCSV(float* array, char* filename)
{
	char buffer[1024];
	char *record, *line;
	FILE *fstream = fopen(filename,"r");
	if(fstream == NULL)
	{
		printf("\n file opening failed ");
	}

	int i = 0;
	int j = 0;
	int index = 0;

	while((line=fgets(buffer,sizeof(buffer),fstream))!=NULL)
	{	
		printf("\n%d",index);//DEBUG
		record = strtok(line,",");
		while(record != NULL)
		{
			array[index] = atof(record);
			record = strtok(line,",");
			index++;
		}
	}
}








