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
#include <cutil_inline.h>

#include "pasp_config.h"
#include "pasp_process.h"
#include "fft_library.h"
#include "debug_macros.h"

// size of the fft
static const unsigned int signalLength = 1024*1024;

int main(int argc, char *argv[])
{
    int i=0;
    
    // buffer for the next packet
    char *hostSignalData;
    
    unsigned int hostSignalDataMemSize = sizeof(char) * signalLength * 2;
    cutilSafeCall( cudaMallocHost( (void**)&hostSignalData, hostSignalDataMemSize));
	if(hostSignalData==NULL){
		fprintf(stderr,"Error : cudaMallocHost failed\n");
		exit(-1);
	}
    
    struct sigaction newact;
    
    int numbytes=0;
    int numpackets=0;
    long long totalbytes=0;
    
    // input fifo file info
    int input_fifo;
    char input_file_name[CHANNEL_FILE_NAME_SIZE];
    
    // this should really be a command line opt
    int channelid=8;
    int polid=0;  
    
    float *hostPower;
    
    unsigned int timer;
    cutCreateTimer(&timer);
    
    
    
    //set up the signal handler
    newact.sa_handler = cleanup;
    sigemptyset(&newact.sa_mask);
    newact.sa_flags = 0;
    
    //start listening for Ctrl-C
	sigaction(SIGINT, &newact, NULL);
    
    // open the fifo with complex data for a single channel/pol
    snprintf(input_file_name, CHANNEL_FILE_NAME_SIZE, CHANNEL_FILE_BASE, channelid, polid);
    debug_fprintf(stderr, "Opening fifo %s\n", input_file_name);
    input_fifo = open(input_file_name,O_RDONLY);
    
    initializeFFT(signalLength);
    
    debug_fprintf(stderr, "Waiting for data\n");
    while(run_fifo_read==1)
    {
        // read packet from fifo
        numbytes = read(input_fifo, (void *) &(hostSignalData[i]), hostSignalDataMemSize);
        //fprintf(stderr,"tried to read %d got %d at %x\n", hostSignalDataMemSize, numbytes, (void *) hostSignalData);
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
            i+=numbytes;
            if(i>=hostSignalDataMemSize)
            {
                cutResetTimer(timer);
		        cutStartTimer(timer);
                hostPower=callFFT(hostSignalData);
                cutStopTimer(timer);
                
                printf("time = %f done...\n",cutGetTimerValue(timer));
                for(int j=0;j<signalLength;j++)
                {
                    if(hostPower[j] != 0)
                    {
                        fprintf(stdout, "%d\t%f\n", j, hostPower[j]);
                    }
                }
                i=0;
            }

        }
    }
    
    debug_fprintf(stderr, "Received %d packets, %lld bytes\n", numpackets, totalbytes);
    debug_fprintf(stderr, "Closing fifo\n");
    close(input_fifo);
    destroyFFT();
    //free(newdata);
    return 0;
}

static void cleanup(int signal)
{
    debug_fprintf(stderr, "Ctrl-C received... cleaning up\n");
	run_fifo_read = 0;
}

