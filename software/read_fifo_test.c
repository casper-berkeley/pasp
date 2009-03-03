#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <fftw3.h>
#include "fifo_test.h"


int main()
{
    int fifo;
    char *data = malloc(DATA_LEN);
    int numbytes=0;
    fprintf(stderr, "Opening fifo\n");
    fifo = open(FILE_NAME,O_RDONLY);
    
    fprintf(stderr, "Sleeping\n");
    sleep(10);
    
    fprintf(stderr, "Attempting to read from fifo\n");
    while(numbytes==0)
    {
        numbytes = read(fifo, (void *) data, 17);
        fprintf(stderr, "Tried to read %d bytes, got %d bytes from fifo\n", DATA_LEN, numbytes);
    }
    fprintf(stderr, "Attempting to read from fifo again\n");
    numbytes=0;
    while(numbytes==0)
    {
        numbytes = read(fifo, (void *) data, 17);
        fprintf(stderr, "Tried to read %d bytes, got %d bytes from fifo\n", DATA_LEN, numbytes);
    }
    //close(fifo);
    free(data);
    return 0;
}
