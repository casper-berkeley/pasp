#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
#include <cutil_inline.h>
#include <cufft.h>
#include "fft_gpu.h"

#define FILENAME_BUFSIZE 200
#define SUM_MAX_THREAD   256
#define SUB_MAX_THREAD   256
#define MAX_THREAD       256
#define MAX_GRID         32*1024
#define LOOP_NUM         1

#include "kernelExec.cu"
#include "fourStepFFT.cu"

#include "output_greg.c"
#include "random.c"



// default value of boxcar, this can be change by -boxcar argument
int  boxcar = 8192;

// default value of threshold, this can be changed by -threshold argument
int  threshold = 20;

// default filename of output file, this can be changed by -output_file argument
char outputFileName[FILENAME_BUFSIZE] = "report.txt";

// the memory size and pointer for generated signal data
unsigned int hostSignalDataMemSize = 0;
char         *hostSignalData = NULL;
float        *hostPowerData = NULL;
float        *hostcuFFTRData = NULL;
cufftComplex *hostcuFFTData = NULL;

// the memory size and pointer for output data
unsigned int hostPowerDataMemSize = 0;
unsigned int hostcuFFTDataMemSize = 0;
unsigned int hostOutputDataMemSize = 0;
outputStruct *hostOutputData = NULL;

// we limit the detected signal points in each boxcar.
// 'maximumDetectPointInBoxcar' specify its value, this can be changed by -max_detect_point argument
int maximumDetectPointInBoxcar = 16;

// output file 
int   outputCounter = 0;
int   outputFclosePeriod = 10;
FILE *outputFilePointer = NULL;

unsigned int timer;
unsigned int timerA;
unsigned int timerB;

cufftHandle plan;

// -- Device memory pointer
char         *devSignalData = NULL;
cufftComplex *devFFTData    = NULL;
cufftComplex *devCuFFTData  = NULL;
float        *devCuFFTRData  = NULL;
float        *devPowerData  = NULL;
float        *devAvgRe      = NULL;
float        *devAvgIm      = NULL;
float        *devPartSumRe  = NULL;
float        *devPartSumIm  = NULL;
outputStruct *devOutputData = NULL;

// Memory size for device
unsigned int devSignalDataMemSize = 0;
unsigned int devFFTDataMemSize    = 0;
unsigned int devPowerDataMemSize  = 0;
unsigned int devPartSumMemSize    = 0;
unsigned int devOutputDataMemSize = 0;

int matrixX, matrixY;

int iter;

int signalLength;

void initializeFFT()
{
	printf("Init Host memory\n");
	
    // -- Init host memory --  
    hostPowerDataMemSize = sizeof(float) * signalLength;
    cutilSafeCall( cudaMallocHost( (void**)&hostPowerData, hostPowerDataMemSize));
    if(hostPowerData==NULL){
            fprintf(stderr,"Error : cudaMallocHost failed\n");
            exit(-1);
    }

    cutilSafeCall( cudaMallocHost( (void**)&hostcuFFTRData, hostPowerDataMemSize));
    if(hostcuFFTRData==NULL){
            fprintf(stderr,"Error : cudaMallocHost failed\n");
            exit(-1);
    }

    hostcuFFTDataMemSize = sizeof(cufftComplex) * signalLength;
    cutilSafeCall( cudaMallocHost( (void**)&hostcuFFTData, hostcuFFTDataMemSize));
    if(hostcuFFTData==NULL){
            fprintf(stderr,"Error : cudaMallocHost failed\n");
            exit(-1);
    }
    hostSignalDataMemSize = sizeof(char) * signalLength * 2;
    hostOutputDataMemSize = sizeof(outputStruct) * maximumDetectPointInBoxcar * (signalLength / boxcar);

    cutilSafeCall( cudaMallocHost( (void**)&hostSignalData, hostSignalDataMemSize));
    if(hostSignalData==NULL){
            fprintf(stderr,"Error : cudaMallocHost failed\n");
            exit(-1);
    }

    cutilSafeCall( cudaMallocHost( (void**)&hostOutputData, hostOutputDataMemSize));
    if(hostOutputData==NULL){
            fprintf(stderr,"Error : cudaMallocHost failed\n");
            exit(-1);
    }

	



	// Calculate memory size
	devSignalDataMemSize = hostSignalDataMemSize;
	devFFTDataMemSize    = sizeof(cufftComplex) * signalLength;
	devPowerDataMemSize  = sizeof(float) * signalLength;
	devPartSumMemSize    = sizeof(float) * SUM_MAX_THREAD;
	devOutputDataMemSize    = hostOutputDataMemSize;

	// Allocate device memory
	printf("Allocate device memory\n");
	cutilSafeCall( cudaMalloc( (void**) &devSignalData, devSignalDataMemSize) );
	cutilSafeCall( cudaMalloc( (void**) &devFFTData,    devFFTDataMemSize) );
	cutilSafeCall( cudaMalloc( (void**) &devCuFFTData,    devFFTDataMemSize) );
	cutilSafeCall( cudaMalloc( (void**) &devCuFFTRData,  devPowerDataMemSize) );
	cutilSafeCall( cudaMalloc( (void**) &devPowerData,  devPowerDataMemSize) );
	cutilSafeCall( cudaMalloc( (void**) &devPartSumRe,  devPartSumMemSize) );
	cutilSafeCall( cudaMalloc( (void**) &devPartSumIm,  devPartSumMemSize) );
	cutilSafeCall( cudaMalloc( (void**) &devAvgRe, sizeof(float) * 1) );
	cutilSafeCall( cudaMalloc( (void**) &devAvgIm, sizeof(float) * 1) );
	cutilSafeCall( cudaMalloc( (void**) &devOutputData, devOutputDataMemSize) );

	// the row length and col length of matrix
	

	// the value of 'matrixY' must be fixed!!! because this program includes only 16-point fft kernel.
	matrixY = 16;
	matrixX = signalLength / matrixY;

	// Generate signal
	long seed1 = 27;
	long seed2 = 22;

	printf("Generating gaussian numbers\n");
	for(int i=0; i<signalLength; i++){

        	//hostSignalData[2*i] = 0.01 * cosf(i * 8 * 2*3.14159265/(float)signalLength) + gauss(&seed1, 0.0, 1.0);
        	//hostSignalData[2*i+1] = 0.01 * sinf(i * 8 * 2*3.14159265/(float)signalLength) + gauss(&seed2, 0.0, 1.0); 
		hostcuFFTData[i].x = 0.01 * cosf(i * 800 * 2*3.14159265/(float)signalLength) + gauss(&seed1, 0.0, 1.0);
		hostcuFFTData[i].y = 0.01 * sinf(i * 800 * 2*3.14159265/(float)signalLength) + gauss(&seed2, 0.0, 1.0);

		//printf("gauss = %f\n",gauss(&seed1, 0.0, 1.0));
	}

	// Initialize output file
	int  result;
	char buf[FILENAME_BUFSIZE];

        result = sprintf(buf,"%s",outputFileName);
	printf("Output File Name %s\n",outputFileName);
        if(result==EOF){
                fprintf(stderr,"Error : sprintf failed in init_output_file()\n");
                //return 0;
        }

        //outputFilePointer = fopen(buf,"wb");
        outputFilePointer = fopen(buf,"w");
        if(outputFilePointer==NULL){
                fprintf(stderr,"Error : fopen failed int init_output_file()\n");
                //return 0;
        }

	// timer
	
	cutCreateTimer(&timer);	
	cutCreateTimer(&timerA);	
	cutCreateTimer(&timerB);	

	// Cufft
	// CUFFT_SAFE_CALL(cufftPlan1d(&plan, signalLength, CUFFT_C2C, 1));
	CUFFT_SAFE_CALL(cufftPlan1d(&plan, signalLength, CUFFT_C2R, 1));



}

void callFFT(cufftComplex *data)
{

		// timer
		cutResetTimer(timer);
		cutResetTimer(timerA);
		cutResetTimer(timerB);

		cutStartTimer(timer);
#if 0
		// CPU -> GPU : move signal data from host to device
		cutilSafeCall( cudaMemcpy(devSignalData, hostSignalData, devSignalDataMemSize, cudaMemcpyHostToDevice));
		cudaThreadSynchronize();

		// GPU : convert char format signal data to float format
		convert_to_float_exec(devSignalData, devPartSumRe, devPartSumIm, devAvgRe, devAvgIm, devFFTData, signalLength);
#endif
		cutilSafeCall( cudaMemcpy( devFFTData, hostcuFFTData, devFFTDataMemSize, cudaMemcpyHostToDevice));

		// GPU : do fft
		if(1) {
			cutStartTimer(timerA);
			do_four_step_fft(devFFTData, devPowerData, matrixX, matrixY);
			cudaThreadSynchronize();
			cutStopTimer(timerA);
		}

		// Do Cuda FFT
		cutStartTimer(timerB);
		if(1) {
			CUFFT_SAFE_CALL(cufftExecC2C(plan, devFFTData, devCuFFTData, CUFFT_FORWARD));
			// CUFFT_SAFE_CALL(cufftExecC2R(plan, devFFTData, devCuFFTRData));
			//CUFFT_SAFE_CALL(cufftExecC2R(plan, devFFTData, (cufftReal*)devFFTData));
			cudaThreadSynchronize();
		}
		cutStopTimer(timerB);

		// GPU : detect strong power spectrum
		//calc_over_threshold_exec(devPowerData, devOutputData, signalLength);

		// GPU -> CPU : copy cuFFT data from device to host
		cutilSafeCall( cudaMemcpy( hostcuFFTRData, devFFTData, devFFTDataMemSize, cudaMemcpyDeviceToHost));
		// cutilSafeCall( cudaMemcpy( hostcuFFTRData, devCuFFTRData, devPowerDataMemSize, cudaMemcpyDeviceToHost));
		cudaThreadSynchronize();

		// GPU -> CPU : copy myfft data from device to host
		cutilSafeCall( cudaMemcpy( hostPowerData, devPowerData, devPowerDataMemSize, cudaMemcpyDeviceToHost));
		cudaThreadSynchronize();

		// CPU : output detect power spectrum to file
		//output_spectrum_to_file(outputFilePointer, signalLength, hostSignalData, hostPowerData,  (cufftReal*)hostcuFFTData);
		output_spectrum_to_file_float(outputFilePointer, signalLength, hostcuFFTData, hostPowerData, hostcuFFTRData);

		// timer
		cutStopTimer(timer);
		printf("time = %f  myfft = %f  CuFFT= %f, %d done...\n",cutGetTimerValue(timer), cutGetTimerValue(timerA), cutGetTimerValue(timerB), iter);



}
void destroyFFT()
{
	// Terminate output file
	terminate_output_file(outputFilePointer);


	// Free device memory
	cutilSafeCall( cudaFree( devSignalData ) );
	cutilSafeCall( cudaFree( devFFTData ) );
	cutilSafeCall( cudaFree( devPowerData ) );
	cutilSafeCall( cudaFree( devPartSumRe ) );
	cutilSafeCall( cudaFree( devPartSumIm ) );
	cutilSafeCall( cudaFree( devAvgRe ) );
	cutilSafeCall( cudaFree( devAvgIm ) );


    // Free signal data memory
    cudaFreeHost(hostSignalData);
    hostSignalData = NULL;

    // Free output data memory
    cudaFreeHost(hostOutputData);
    hostOutputData = NULL;


}

extern "C"
int do_analyze_on_gpu(int sl)  {
    signalLength=sl;
    initializeFFT();


	// Main loop
	printf("Main loop\n");
	for(iter=0; iter<LOOP_NUM; iter++){
        callFFT(NULL);
	}

    destroyFFT();

    return 0;
}
