/*
 *  pasp_process.c
 *  
 *
 *  Created by Terry E. Filiba on 2/24/09.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
//#include <fftw3.h>

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>

#include "pasp_config.h"
#include "pasp_process.h"

#define NX      256
#define BATCH   1

void callGPUFFT(cufftComplex *data)
{
    int i;
    cufftHandle plan;
    // cufftComplex data[NX*BATCH];
    cufftComplex *gpudata;
    
    // generate some random data
//    for(i=0; i<NX*BATCH; i++)
//    {
//        data[i].x=1.0f;
//        data[i].y=1.0f;
//    }
    
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
    cudaFree(data);
}

void accumulate_packet(pasp_packet *newpacket,int numchannels)
{
    int i,j;
    // allocate space for accumulated data
    accumulated_channel *accumulated_channels = (accumulated_channel *) calloc(numchannels,sizeof(accumulated_channel));
    
    for(i=0;i<SAMPLES_PER_CHANNEL;i++)
    {
        for(j=0;j<numchannels;j++)
        {
            accumulated_channels[j].pol0_re += newpacket->samples[i][j].pol0_re;
            accumulated_channels[j].pol0_im += newpacket->samples[i][j].pol0_im;
            accumulated_channels[j].pol1_re += newpacket->samples[i][j].pol1_re;
            accumulated_channels[j].pol1_im += newpacket->samples[i][j].pol1_im;
        }
    }
    
    for(i=0;i<numchannels;i++)
    {
        fprintf(stderr,"%d %d %d %d\n", 
                accumulated_channels[i].pol0_re,
                accumulated_channels[i].pol0_im,
                accumulated_channels[i].pol1_re,
                accumulated_channels[i].pol1_im);
    }
}

void process_packet(pasp_packet *newpacket)
{
    int i;
    cufftComplex data[NX*BATCH];
    
    //copy a single channel into the buffer
    for(i=0;i<NX*BATCH;i++)
    {
        data[i].x=newpacket->samples[i][0].pol0_re;
        data[i].y=newpacket->samples[i][0].pol0_im;
    }
    
    accumulate_packet(newpacket,CHANNELS_PER_PACKET);
    
    //call fft on the buffer
    //callGPUFFT(data);
}


int main(int argc, char *argv[])
{
    int fifo;
    pasp_packet *nextpacket = malloc(PACKET_SIZE_BYTES);
    int numbytes=0;
    //int byteswritten=0;
    struct sigaction newact;
    int numpackets=0;
    long long totalbytes=0;
    
    //set up the signal handler
    newact.sa_handler = cleanup;
    sigemptyset(&newact.sa_mask);
    newact.sa_flags = 0;
    
    //start listening for Ctrl-C
	sigaction(SIGINT, &newact, NULL);
    
    debug_fprintf(stderr, "Opening fifo\n");
    fifo = open(FILE_NAME,O_RDONLY);
    
    
    debug_fprintf(stderr, "Waiting for data\n");
    while(run_fifo_read==1)
    {
        // read packet from fifo
        numbytes = read(fifo, (void *) nextpacket, PACKET_SIZE_BYTES);
        if(numbytes==-1 && run_fifo_read==1)
        {
            perror("Error reading from fifo");
            exit(0);
        }
        
        // process packet
        if(run_fifo_read==1 && numbytes!=0)
        {
            numpackets++;
            totalbytes+=numbytes;
            
            fprintf(stderr,"%ld %ld\n",ntohll(nextpacket->seq_no),ntohll(nextpacket->id_no));
            //fprintf(stderr,"%d\n",nextpacket->samples[0][0].pol0_re);
            process_packet(nextpacket);
        }
    }
    
    debug_fprintf(stderr, "Received %d packets, %lld bytes\n", numpackets, totalbytes);
    debug_fprintf(stderr, "Closing fifo\n");
    close(fifo);
    free(nextpacket);
    return 0;
}

static void cleanup(int signal)
{
    debug_fprintf(stderr, "Ctrl-C received... cleaning up\n");
	run_fifo_read = 0;
}

