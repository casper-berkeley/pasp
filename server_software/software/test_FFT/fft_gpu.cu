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


#include "kernelExec.cu"
#include "fourStepFFT.cu"

#include "output_greg.c"
#include "output.c"


// default value of boxcar, this can be change by -boxcar argument
int  boxcar = 8192;

// default value of threshold, this can be changed by -threshold argument
int  threshold = 20;

// default filename of output file, this can be changed by -output_file argument
char outputFileName[FILENAME_BUFSIZE] = "report.txt";

// the memory size and pointer for generated signal data
//char         *hostSignalData = NULL;
float        *hostPowerData = NULL;
float        *hostcuFFTRData = NULL;
//cufftComplex *hostcuFFTData = NULL;

// the memory size and pointer for output data
unsigned int hostPowerDataMemSize = 0;
//unsigned int hostcuFFTDataMemSize = 0;
unsigned int hostOutputDataMemSize = 0;
outputStruct *hostOutputData = NULL;

// we limit the detected signal points in each boxcar.
// 'maximumDetectPointInBoxcar' specify its value, this can be changed by -max_detect_point argument
int maximumDetectPointInBoxcar = 16;

// output file 
int   outputCounter = 0;
int   outputFclosePeriod = 10;
FILE *outputFilePointer = NULL;

//unsigned int timer;
//unsigned int timerA;
//unsigned int timerB;

//cufftHandle plan;

// -- Device memory pointer
static char         *devSignalData = NULL;
static cufftComplex *devFFTData    = NULL;
//cufftComplex *devCuFFTData  = NULL;
//float        *devCuFFTRData  = NULL;
static float        *devPowerData  = NULL;
static float        *devAvgRe      = NULL;
static float        *devAvgIm      = NULL;
static float        *devPartSumRe  = NULL;
static float        *devPartSumIm  = NULL;
static outputStruct *devOutputData = NULL;

// Memory size for device
static unsigned int devSignalDataMemSize = 0;
static unsigned int devFFTDataMemSize    = 0;
static unsigned int devPowerDataMemSize  = 0;
static unsigned int devOutputDataMemSize = 0;

int matrixX, matrixY;



int signalLength;

void initializeFFT(int initializedSignalLength)
{
    unsigned int devPartSumMemSize    = 0;
    
    signalLength=initializedSignalLength;
    // initialize signal data on the host (cpu)
	hostOutputDataMemSize = sizeof(outputStruct) * maximumDetectPointInBoxcar * (signalLength / boxcar);


	cutilSafeCall( cudaMallocHost( (void**)&hostOutputData, hostOutputDataMemSize));
	if(hostOutputData==NULL){
		fprintf(stderr,"Error : cudaMallocHost failed\n");
		exit(-1);
	}
    

    
	// Calculate memory size
	devSignalDataMemSize = sizeof(char) * signalLength * 2;
	devFFTDataMemSize    = sizeof(cufftComplex) * signalLength;
	devPowerDataMemSize  = sizeof(float) * signalLength;
	devPartSumMemSize    = sizeof(float) * SUM_MAX_THREAD;
	devOutputDataMemSize    = hostOutputDataMemSize;

	// Allocate device memory
	cutilSafeCall( cudaMalloc( (void**) &devSignalData, devSignalDataMemSize) );
	cutilSafeCall( cudaMalloc( (void**) &devFFTData,    devFFTDataMemSize) );
	cutilSafeCall( cudaMalloc( (void**) &devPowerData,  devPowerDataMemSize) );
	cutilSafeCall( cudaMalloc( (void**) &devPartSumRe,  devPartSumMemSize) );
	cutilSafeCall( cudaMalloc( (void**) &devPartSumIm,  devPartSumMemSize) );
	cutilSafeCall( cudaMalloc( (void**) &devAvgRe, sizeof(float) * 1) );
	cutilSafeCall( cudaMalloc( (void**) &devAvgIm, sizeof(float) * 1) );
	cutilSafeCall( cudaMalloc( (void**) &devOutputData, devOutputDataMemSize) );
    
    cutilSafeCall( cudaMallocHost( (void**)&hostPowerData, devPowerDataMemSize));

	// the row length and col length of matrix
	//int matrixX, matrixY;

	// the value of 'matrixY' must be fixed!!! because this program includes only 16-point fft kernel.
	matrixY = 16;
	matrixX = signalLength / matrixY;

	// Initialize output file
	char buf[FILENAME_BUFSIZE];
	int  result;

	result = sprintf(buf,"%d_%s",outputCounter,outputFileName);
	if(result==EOF){
		fprintf(stderr,"Error : sprintf failed in init_output_file()\n");
		//return 0;
	}

	outputFilePointer = fopen(buf,"wb");
	//outputFilePointer = fopen(buf,"w");
	if(outputFilePointer==NULL){
		fprintf(stderr,"Error : fopen failed int init_output_file()\n");
		//return 0;
	}

	outputCounter++;


	// timer
	//cutCreateTimer(&timer);	

}

float * callFFT(char *hostSignalData)
{
    // timer
//    cutResetTimer(timer);
//    cutStartTimer(timer);
    

    // CPU -> GPU : move signal data from host to device
    cutilSafeCall( cudaMemcpy(devSignalData, hostSignalData, devSignalDataMemSize, cudaMemcpyHostToDevice));
    cudaThreadSynchronize();

    // GPU : convert char format signal data to float format
    convert_to_float_exec(devSignalData, devPartSumRe, devPartSumIm, devAvgRe, devAvgIm, devFFTData, signalLength);
    
    //copy over data after convert_to_float_exec
    //cufftComplex * hostFFTData;
    //cutilSafeCall( cudaMallocHost( (void**) &hostFFTData,    devFFTDataMemSize) );
    //cutilSafeCall( cudaMemcpy( hostFFTData, devFFTData, devOutputDataMemSize, cudaMemcpyDeviceToHost));
    ////for(int i=0;i<signalLength;i++)
    //for(int i=0;i<20;i++)
    //{
    //    printf("%f %f\n", hostFFTData[i].x, hostFFTData[i].y);
    //}
    

    // GPU : do fft
    do_four_step_fft(devFFTData, devPowerData, matrixX, matrixY);

    // GPU : detect strong power spectrum
//    calc_over_threshold_exec(devPowerData, devOutputData, signalLength, boxcar, threshold, maximumDetectPointInBoxcar);

    // GPU -> CPU : copy detect spectrum data from device to host
//    cutilSafeCall( cudaMemcpy( hostOutputData, devOutputData, devOutputDataMemSize, cudaMemcpyDeviceToHost));
//    cudaThreadSynchronize();

    // CPU : output detect power spectrum to file
    //output_spectrum(hostOutputData, iter, 1);
    //output_spectrum_to_file_float(outputFilePointer, signalLength, hostcuFFTData, hostPowerData);

    cutilSafeCall( cudaMemcpy( hostPowerData, devPowerData, devPowerDataMemSize, cudaMemcpyDeviceToHost));
    
//    for(int i=0; i<signalLength; i++)
//    {
//        if(hostPowerData[i]!=0)
//        {
//            printf("%d: %f\n", i, hostPowerData[i]);
//        }
//    }

    // timer
    //cutStopTimer(timer);
    //printf("time = %f done...\n",cutGetTimerValue(timer));
    return hostPowerData;
}
void destroyFFT()
{
	// Terminate output file
	terminate_output_file();


	// Free device memory
	cutilSafeCall( cudaFree( devSignalData ) );
	cutilSafeCall( cudaFree( devFFTData ) );
	cutilSafeCall( cudaFree( devPowerData ) );
	cutilSafeCall( cudaFree( devPartSumRe ) );
	cutilSafeCall( cudaFree( devPartSumIm ) );
	cutilSafeCall( cudaFree( devAvgRe ) );
	cutilSafeCall( cudaFree( devAvgIm ) );
}
