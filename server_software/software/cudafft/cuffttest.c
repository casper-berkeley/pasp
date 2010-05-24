#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>

#define NX      256
#define BATCH   4

int main ()
{
    int i;
    cufftHandle plan;
    cufftComplex data[NX*BATCH];
    cufftComplex *gpudata;
    
    // generate some random data
    for(i=0; i<NX*BATCH; i++)
    {
        data[i].x=1.0f;
        data[i].y=1.0f;
    }
    
    // allocate device memory and copy over data
    cudaMalloc((void**)&gpudata,sizeof(cufftComplex)*NX*BATCH);
    cudaMemcpy(gpudata, data, sizeof(cufftComplex)*NX*BATCH, cudaMemcpyHostToDevice);
    
    // run the fft
    cufftPlan1d(&plan,NX,CUFFT_C2C, BATCH);
    cufftExecC2C(plan,gpudata,gpudata,CUFFT_FORWARD);
    
    // copy the result back
    cudaMemcpy(data, gpudata, sizeof(cufftComplex)*NX*BATCH, cudaMemcpyDeviceToHost);
    
    for(i=0; i<NX*BATCH; i++)
    {
        fprintf(stderr,"%d %f %f\n", i, data[i].x, data[i].y);
    }
    
    cufftDestroy(plan);
    cudaFree(gpudata);
    
    return 0;
}
