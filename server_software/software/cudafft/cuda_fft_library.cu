#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>

//#include "fft_library.h"
//#include "pasp_config.h"

static cufftHandle plan;
cufftComplex *gpudata;
cufftComplex *fftgpudata;

/*******************
REMOVE
*******************/
#define CHANNEL_BUFFER_SIZE 11
#define NX  16
#define BATCH 10
#define SAMPLES_PER_CHANNEL 5

void initializeFFT()
{
    // allocate device memory for the fft
    cudaMalloc((void**)&gpudata,CHANNEL_BUFFER_SIZE*NX*BATCH);
    cudaMalloc((void**)&fftgpudata,CHANNEL_BUFFER_SIZE*NX*BATCH);
    
    cufftPlan1d(&plan,SAMPLES_PER_CHANNEL*NX,CUFFT_C2C, BATCH);
}


void callFFT(cufftComplex *data)
{
    //int i;
    // allocate device memory and copy over data
    cudaMemcpy(gpudata, data, CHANNEL_BUFFER_SIZE*NX*BATCH, cudaMemcpyHostToDevice);
    
    // run the fft
    
    cufftExecC2C(plan,gpudata,fftgpudata,CUFFT_FORWARD);
    
    // copy the result back
    cudaMemcpy(data, fftgpudata, CHANNEL_BUFFER_SIZE*NX*BATCH, cudaMemcpyDeviceToHost);
    
//    for(i=0; i<SAMPLES_PER_CHANNEL*NX*BATCH; i++)
//    {
//        fprintf(stderr,"%d %f %f\n", i, data[i].x, data[i].y);
//    }
    

}

void destroyFFT()
{
    cufftDestroy(plan);
    cudaFree(gpudata);
}
