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
#define MIN_NX      32768
#define MAX_NX      32768
#define MIN_BATCH   1
#define MAX_BATCH   4096
#define MAX_DIM     32768*4096
//#define MAX_DIM     16777216


static cufftHandle plan;
cufftComplex *gpudata;
cufftComplex *fftgpudata;


int main ()
{
    //int deviceCount = 0;
    CUresult err = cuInit(0);
    //CU_SAFE_CALL_NO_SYNC(cuDeviceGetCount(&deviceCount));
    //printf("There are %d devices supporting CUDA\n", deviceCount);
    
    long long i;
    long long nx;
    long long batch;
    unsigned int complete_fft_timer;
    unsigned int piecewise_fft_timer;
    unsigned int copy_to_gpu_timer;
    unsigned int fft_only_timer;
    unsigned int copy_from_gpu_timer;
    
    cutCreateTimer(&complete_fft_timer);
    cutCreateTimer(&piecewise_fft_timer);
    cutCreateTimer(&copy_to_gpu_timer);
    cutCreateTimer(&fft_only_timer);
    cutCreateTimer(&copy_from_gpu_timer);
    
    //cufftHandle plan;
    cufftComplex *data;
    cufftComplex *result;
    //cufftComplex *gpudata;
    
    CUDA_SAFE_CALL(cudaMallocHost(&data, sizeof(cufftComplex)*MAX_DIM));
    CUDA_SAFE_CALL(cudaMallocHost(&result, sizeof(cufftComplex)*MAX_DIM));
    //cudaHostAlloc(&data, sizeof(cufftComplex)*MAX_DIM,cudaHostAllocWriteCombined);
    //cudaHostAlloc(&result, sizeof(cufftComplex)*MAX_DIM,cudaHostAllocWriteCombined);
    
    // get the total global memory of the device
    CUdevice dev=0;
    unsigned int totalGlobalMem;
	CU_SAFE_CALL_NO_SYNC( cuDeviceTotalMem(&totalGlobalMem, dev) );
    fprintf(stderr,"Total amount of global memory: %u bytes\n", totalGlobalMem);
    
    //fprintf(stderr, "Initializing data... ");
    // generate some random data
    for(i=0; i<MAX_DIM; i++)
    {
        data[i].x=1.0f;
        data[i].y=1.0f;
    }
    //fprintf(stderr, "done\n");

    fprintf(stderr, "Testing cufft C2C 1D fft with cudaMallocHost allocated memory\n");
    //fprintf(stderr, "nx\tbatch\ttime\tcopy_to_gpu\tactual_fft\tcopy_from_gpu\tavg\n");
    for(nx=MIN_NX; nx<=MAX_NX; nx=nx*2)
    //for(nx=4096; nx<=MAX_NX; nx+=4096)
    {
        fprintf(stderr, "Pts\tbatch\ttotal time\tsum piecewise\tcopy to gpu\tfft on gpu\tcopy from gpu\tperformance per point\ttimes reported in ms\n");
        for(batch=MIN_BATCH;batch<=MAX_BATCH;batch=batch*2)
        {
            if(sizeof(cufftComplex)*nx*batch*2 <= totalGlobalMem/2 && nx*batch<MAX_DIM)
            {  
                //fprintf(stderr, "Allocating %lld bytes\n", sizeof(cufftComplex)*nx*batch*2);
                // allocate device memory for the fft
                CUDA_SAFE_CALL(cudaMalloc((void**)&gpudata,sizeof(cufftComplex)*nx*batch));
                CUDA_SAFE_CALL(cudaMalloc((void**)&fftgpudata,sizeof(cufftComplex)*nx*batch));

                cufftPlan1d(&plan,nx,CUFFT_C2C, batch);
                cudaThreadSynchronize();
                
                cutResetTimer(complete_fft_timer);
                cutStartTimer(complete_fft_timer);
                // run the fft
                // allocate device memory and copy over data
                cudaMemcpy(gpudata, data, sizeof(cufftComplex)*nx*batch, cudaMemcpyHostToDevice);
                cudaThreadSynchronize();
                // run the fft
                cufftExecC2C(plan,gpudata,fftgpudata,CUFFT_FORWARD);
                cudaThreadSynchronize();
                // copy the result back
                cudaMemcpy(result, fftgpudata, sizeof(cufftComplex)*nx*batch, cudaMemcpyDeviceToHost);
                cudaThreadSynchronize();
                cutStopTimer(complete_fft_timer);
                
                cutResetTimer(piecewise_fft_timer);
                cutStartTimer(piecewise_fft_timer);
                
                cutResetTimer(copy_to_gpu_timer);
                cutStartTimer(copy_to_gpu_timer);
                cudaMemcpy(gpudata, data, sizeof(cufftComplex)*nx*batch, cudaMemcpyHostToDevice);
                cudaThreadSynchronize();
                cutStopTimer(copy_to_gpu_timer);
                
                cutResetTimer(fft_only_timer);
                cutStartTimer(fft_only_timer);
                cufftExecC2C(plan,gpudata,fftgpudata,CUFFT_FORWARD);
                cudaThreadSynchronize();
                cutStopTimer(fft_only_timer);
                
                cutResetTimer(copy_from_gpu_timer);
                cutStartTimer(copy_from_gpu_timer);
                cudaMemcpy(result, fftgpudata, sizeof(cufftComplex)*nx*batch, cudaMemcpyDeviceToHost);
                cudaThreadSynchronize();
                cutStopTimer(copy_from_gpu_timer);
                
                cutStopTimer(piecewise_fft_timer);
                
                cufftDestroy(plan);
                CUDA_SAFE_CALL(cudaFree(gpudata));
                CUDA_SAFE_CALL(cudaFree(fftgpudata));
                cudaThreadSynchronize();
            
                
                fprintf(stderr, "%lld\t%lld\t%f\t%f\t%f\t%f\t%f\t%f\n",
                    nx, batch, 
                    cutGetTimerValue(complete_fft_timer), cutGetTimerValue(piecewise_fft_timer), 
                    cutGetTimerValue(copy_to_gpu_timer), 
                    cutGetTimerValue(fft_only_timer), cutGetTimerValue(copy_from_gpu_timer),
                    cutGetTimerValue(complete_fft_timer)/(nx*batch));
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
