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
#include <cutil_inline.h>

#include "pasp_config.h"
#include "pasp_process.h"
#include "fft_library.h"
#include "debug_macros.h"

static cufftComplex * generateData()
{
    int i;
    cufftComplex *data = (cufftComplex *) malloc(sizeof(cufftComplex)*NX*BATCH*SAMPLES_PER_CHANNEL);
    
    //generate some random data
    for(i=0; i<NX*BATCH*SAMPLES_PER_CHANNEL; i++)
    {
        data[i].x=1.0f;
        data[i].y=1.0f;
    }
    
    //    for(i=0; i<SAMPLES_PER_CHANNEL*BATCH; i++)
    //    {
    //        fprintf(stderr,"%d %f %f\n", i, data[i].x, data[i].y);
    //    }
    
    return data;
}

int main(int argc, char *argv[])
{
    // input fifo file info
    //int input_fifo;
    
    //int i=0;
    
    // buffer for the next packet
    cufftComplex *newdata=generateData();
    //int numbytes=0;
    struct sigaction newact;
    int numpackets=0;
    long long totalbytes=0;
    
    unsigned int timer;
    
    cutCreateTimer(&timer);
    
    // this should really be a command line opt
    //int channelid=8;
    //int polid=0;    
    
    
    //set up the signal handler
    newact.sa_handler = cleanup;
    sigemptyset(&newact.sa_mask);
    newact.sa_flags = 0;
    
    //start listening for Ctrl-C
	sigaction(SIGINT, &newact, NULL);
    
    initializeFFT();
    while(run_fifo_read==1)
    {

        numpackets++;
        totalbytes+=sizeof(cufftComplex)*NX*BATCH*SAMPLES_PER_CHANNEL;
		cutResetTimer(timer);
		cutStartTimer(timer);
        callFFT((cufftComplex *) newdata);
        cutStopTimer(timer);
        printf("time = %f done...\n",cutGetTimerValue(timer));
    }
    debug_fprintf(stderr, "Received %d packets, %lld bytes\n", numpackets, totalbytes);
    //debug_fprintf(stderr, "Closing fifo\n");
    //close(input_fifo);
    fprintf(stderr,"destroying fft\n");
    destroyFFT();
    //free(newdata);
    return 0;
}

static void cleanup(int signal)
{
    debug_fprintf(stderr, "Ctrl-C received... cleaning up\n");
	run_fifo_read = 0;
}

