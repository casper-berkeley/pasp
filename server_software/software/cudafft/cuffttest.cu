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
#include <cutil_inline.h>

//#define NX      256
//#define BATCH   4
#define MAX_NX      16777216
#define MAX_BATCH   4096
#define MAX_DIM     16777216


static cufftHandle plan;
cufftComplex *gpudata;
cufftComplex *fftgpudata;


int main ()
{
    long long i;
    long long nx;
    long long batch;
    unsigned int complete_fft_timer;
    unsigned int copy_to_gpu_timer;
    unsigned int fft_only_timer;
    unsigned int copy_from_gpu_timer;
    
    //cufftHandle plan;
    cufftComplex *data = (cufftComplex *) malloc(sizeof(cufftComplex)*MAX_DIM);
    cufftComplex *result = (cufftComplex *) malloc(sizeof(cufftComplex)*MAX_DIM);
    //cufftComplex *gpudata;
    
    //fprintf(stderr, "Initializing data... ");
    // generate some random data
    for(i=0; i<MAX_DIM; i++)
    {
        data[i].x=1.0f;
        data[i].y=1.0f;
    }
    //fprintf(stderr, "done\n");
    cutCreateTimer(&complete_fft_timer);
    cutCreateTimer(&copy_to_gpu_timer);
    cutCreateTimer(&fft_only_timer);
    cutCreateTimer(&copy_from_gpu_timer);
    
    fprintf(stderr, "nx, batch, time, copy_to_gpu, actual_fft, copy_from_gpu\n");
    for(nx=2; nx<=MAX_NX; nx=nx*2)
    {
        for(batch=1;batch<=MAX_BATCH;batch=batch*2)
        {
            if(nx*batch <= MAX_DIM)
            {  
                // allocate device memory for the fft
                CUDA_SAFE_CALL(cudaMalloc((void**)&gpudata,sizeof(cufftComplex)*nx*batch));
                CUDA_SAFE_CALL(cudaMalloc((void**)&fftgpudata,sizeof(cufftComplex)*nx*batch));

                cufftPlan1d(&plan,nx,CUFFT_C2C, batch);
                
                cutResetTimer(complete_fft_timer);
                cutStartTimer(complete_fft_timer);
                // run the fft
                // allocate device memory and copy over data
                cudaMemcpy(gpudata, data, sizeof(cufftComplex)*nx*batch, cudaMemcpyHostToDevice);
                // run the fft
                cufftExecC2C(plan,gpudata,fftgpudata,CUFFT_FORWARD);
                // copy the result back
                cudaMemcpy(result, fftgpudata, sizeof(cufftComplex)*nx*batch, cudaMemcpyDeviceToHost);
                cutStopTimer(complete_fft_timer);
                
                cutResetTimer(copy_to_gpu_timer);
                cutStartTimer(copy_to_gpu_timer);
                cudaMemcpy(gpudata, data, sizeof(cufftComplex)*nx*batch, cudaMemcpyHostToDevice);
                cutStopTimer(copy_to_gpu_timer);
                
                cutResetTimer(fft_only_timer);
                cutStartTimer(fft_only_timer);
                cufftExecC2C(plan,gpudata,fftgpudata,CUFFT_FORWARD);
                cutStopTimer(fft_only_timer);
                
                cutResetTimer(copy_from_gpu_timer);
                cutStartTimer(copy_from_gpu_timer);
                cudaMemcpy(result, fftgpudata, sizeof(cufftComplex)*nx*batch, cudaMemcpyDeviceToHost);
                cutStopTimer(copy_from_gpu_timer);
                
                cufftDestroy(plan);
                CUDA_SAFE_CALL(cudaFree(gpudata));
                CUDA_SAFE_CALL(cudaFree(fftgpudata));
            
            
                fprintf(stderr, "%d, %d, %f, %f, %f, %f\n",
                    nx, batch, 
                    cutGetTimerValue(complete_fft_timer), cutGetTimerValue(copy_to_gpu_timer), 
                    cutGetTimerValue(fft_only_timer), cutGetTimerValue(copy_from_gpu_timer));
            }
        }
    }
    
    //print fft data
//    for(i=0; i<NX*BATCH; i++)
//    {
//        fprintf(stderr,"%d %f %f\n", i, data[i].x, data[i].y);
//    }
    
    
    return 0;
}
