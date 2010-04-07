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

#define DEFAULT_RECORD_FILE "pasp_recording"


int main(int argc, char *argv[])
{
    int fifo;
    char *data = malloc(PACKET_SIZE_BYTES);
    int numbytes=0;
    //int byteswritten=0;
    struct sigaction newact;
    FILE *pasp_recording;
    int numpackets=0;
    long long totalbytes=0;
    
    if(argc >= 2)
    {
        pasp_recording = fopen(argv[1],"w");
    }
    else
    {
        pasp_recording = fopen(DEFAULT_RECORD_FILE,"w");
    }
    
    if(pasp_recording==NULL)
    {
        perror("Error opening file pasp_recording");
    }
    
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
        numbytes = read(fifo, (void *) data, PACKET_SIZE_BYTES);
        if(numbytes==-1 && run_fifo_read==1)
        {
            perror("Error reading from fifo");
            exit(0);
        }
        numpackets++;
        totalbytes+=numbytes;
        
        if(run_fifo_read==1 && numbytes!=0)
        {
            //debug_fprintf(stderr, "Fifo contains %s\n", (char *) data);
            
//            byteswritten = fwrite(data, numbytes, 1, pasp_recording);
//            if(byteswritten==0)
//            {
//                debug_fprintf(stderr, "Tried to write %d bytes, wrote 0\n", numbytes);
//            }
        }
    }
    
    debug_fprintf(stderr, "Received %d packets, %llx bytes\n", numpackets, totalbytes);
    debug_fprintf(stderr, "Closing fifo\n");
    close(fifo);
    fclose(pasp_recording);
    free(data);
    return 0;
}

static void cleanup(int signal)
{
    debug_fprintf(stderr, "Ctrl-C received... cleaning up\n");
	run_fifo_read = 0;
}

