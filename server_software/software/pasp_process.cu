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

#include "pasp_config.h"
#include "pasp_process.h"
#include "fft_library.h"



int main(int argc, char *argv[])
{
    // input fifo file info
    int input_fifo;
    
    int i=0;
    
    // buffer for the next packet
    cufftComplex newdata[NX*BATCH][SAMPLES_PER_CHANNEL];
    int numbytes=0;
    struct sigaction newact;
    int numpackets=0;
    long long totalbytes=0;
    char input_file_name[CHANNEL_FILE_NAME_SIZE];
    
    // this should really be a command line opt
    int channelid=8;
    int polid=0;    
    
    
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
    
    initializeFFT();
    
    debug_fprintf(stderr, "Waiting for data\n");
    while(run_fifo_read==1)
    {
        // read packet from fifo
        numbytes = read(input_fifo, (void *) &(newdata[i][0]), CHANNEL_BUFFER_SIZE);
        //fprintf(stderr,"tried to read %d got %d at %x\n", CHANNEL_BUFFER_SIZE, numbytes, (void *) &(newdata[i][0]));
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
            if(i==NX*BATCH-1)
            {
                callFFT((cufftComplex *) newdata);
                i=0;
            }
            else 
            {
                i++;
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

