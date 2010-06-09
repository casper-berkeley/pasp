#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>
#include <cutil_inline.h>


static cufftHandle plan;
cufftComplex *hostSignalData;
float *hostPowerData;
cufftComplex *gpudata;
cufftComplex *fftgpudata;

int signalLength;

/*******************
REMOVE
*******************/
#define BATCH 1

void initializeFFT(int initializedSignalLength)
{
    signalLength = initializedSignalLength;

    cutilSafeCall( cudaMallocHost( (void**)&hostSignalData, sizeof(cufftComplex)*signalLength) );
    cutilSafeCall( cudaMallocHost((void**)&hostPowerData, sizeof(float)*signalLength) );
    
    // allocate device memory for the fft
    cudaMalloc((void**)&gpudata,sizeof(cufftComplex)*signalLength);
    cudaMalloc((void**)&fftgpudata,sizeof(cufftComplex)*signalLength);
    
    
    
    cufftPlan1d(&plan,signalLength,CUFFT_C2C, BATCH);
}


float * callFFT(char *data)
{
    //convert the data to float
    for(int i=0; i<signalLength; i++)
    {
        hostSignalData[i].x = data[2*i];
        hostSignalData[i].y = data[2*i+1];
    }
    
    // copy data to the gpu
    cudaMemcpy(gpudata, hostSignalData, sizeof(cufftComplex)*signalLength, cudaMemcpyHostToDevice);
    
    // run the fft
    
    cufftExecC2C(plan,gpudata,fftgpudata,CUFFT_FORWARD);
    cudaThreadSynchronize();
    
    // copy the result back
    cudaMemcpy(hostSignalData, fftgpudata, sizeof(cufftComplex)*signalLength, cudaMemcpyDeviceToHost);
    
    for(int i=0;i<signalLength;i++)
    {
        hostPowerData[i] = hostSignalData[i].x*hostSignalData[i].x + hostSignalData[i].y*hostSignalData[i].y;
    }
    
//    for(int i=0; i<signalLength; i++)
//    {
//        fprintf(stderr,"%d %f %f\n", i, hostSignalData[i].x, hostSignalData[i].y);
//    }
    
    return hostPowerData;
}

void destroyFFT()
{
    cufftDestroy(plan);
    cudaFreeHost(hostSignalData);
    cudaFreeHost(hostPowerData);
    cudaFree(gpudata);
    cudaFree(fftgpudata);
}
