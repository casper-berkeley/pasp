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


int main()
{
    int fifo;
    char *data = malloc(DATA_LEN);
    int numbytes=0;
    struct sigaction newact;
    
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
        numbytes = read(fifo, (void *) data, PACKET_SIZE_BITS);
        fprintf(stderr, "Tried to read %d bytes, got %d bytes from fifo\n", DATA_LEN, numbytes);
        fprintf(stderr, "Fifo contains %s\n", (char *) data);
    }
    
    debug_fprintf(stderr, "Closing fifo\n");
    close(fifo);
    free(data);
    return 0;
}

static void cleanup(int signal)
{
    debug_fprintf(stderr, "Ctrl-C received... cleaning up\n");
	run_fifo_read = 0;
}

