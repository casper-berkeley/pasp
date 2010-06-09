/*
 *  pasp_process_test.cu
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
#include <cutil_inline.h>

#include "pasp_config.h"
#include "pasp_process.h"
#include "fft_library.h"
#include "debug_macros.h"

static const unsigned int signalLength = 1024*1024;

static char * generateData(int signalLength)
{
    int i;
    char *hostSignalData;
    unsigned int hostSignalDataMemSize = sizeof(char) * signalLength * 2;
    cutilSafeCall( cudaMallocHost( (void**)&hostSignalData, hostSignalDataMemSize));
	if(hostSignalData==NULL){
		fprintf(stderr,"Error : cudaMallocHost failed\n");
		exit(-1);
	}
    
    //generate some data
     for(i=0; i<signalLength; i++){
        hostSignalData[2*i] = 10;
        hostSignalData[2*i+1] = 10;
        //sine wave at 1/2 nyquist
//        if(i%4==0)
//        {
//            hostSignalData[2*i] = -10;
//            hostSignalData[2*i+1] = 10*(i%2);
//        }
//        else if(i%4==1)
//        {
//            hostSignalData[2*i] = 0;
//            hostSignalData[2*i+1] = 10*(i%2);
//        }
//        else if(i%4==2)
//        {
//            hostSignalData[2*i] = 10;
//            hostSignalData[2*i+1] = 10*(i%2);
//        }
//        else if(i%4==3)
//        {
//            hostSignalData[2*i] = 0;
//            hostSignalData[2*i+1] = 10*(i%2);
//        }
        //sine wave a nyquist
//        hostSignalData[2*i] = 10*(i%2);
//        hostSignalData[2*i+1] = 10*(i%2);
	}
    
    return hostSignalData;
}

int main(int argc, char *argv[])
{     
    // buffer for the next packet
    char *newdata=generateData(signalLength);
    //int numbytes=0;
    struct sigaction newact;
    int numpackets=0;
    long long totalbytes=0;
    unsigned int timer;
    float *hostPower;
    
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
    
    initializeFFT(signalLength);
//    while(run_fifo_read==1)
//    {

        numpackets++;
        totalbytes+=sizeof(char)*signalLength*2;
		cutResetTimer(timer);
		cutStartTimer(timer);
        hostPower = callFFT((char *) newdata);
        cutStopTimer(timer);
        printf("time = %f done...\n",cutGetTimerValue(timer));
        for(int i=0;i<signalLength;i++)
        {
            if(hostPower[i] != 0)
            {
                fprintf(stdout, "%d %f\n", i, hostPower[i]);
            }
        }
//    }
    debug_fprintf(stderr, "Processed %d ffts, %lld bytes\n", numpackets, totalbytes);

    fprintf(stderr,"destroying fft\n");
    destroyFFT();
    cudaFreeHost(newdata);
    return 0;
}

static void cleanup(int signal)
{
    debug_fprintf(stderr, "Ctrl-C received... cleaning up\n");
	run_fifo_read = 0;
}

