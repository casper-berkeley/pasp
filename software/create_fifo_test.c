#include <stdlib.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include "fifo_test.h"


int main()
{
    int fifo;
    char *somedata = "CakeCakeCakeCake";
    int numbytes;
    fprintf(stderr, "Creating fifo\n");
    mkfifo(FILE_NAME,0666);
    fprintf(stderr, "Opening fifo\n");
    fifo = open(FILE_NAME, O_WRONLY);
    
//    fprintf(stderr, "Sleeping\n");
//    sleep(10);
    
    fprintf(stderr, "Writing %s to fifo\n", somedata);
    numbytes = write(fifo, (void *) somedata, strlen(somedata)+1);
    fprintf(stderr, "Wrote %d bytes to fifo\n", numbytes);
    numbytes = write(fifo, (void *) somedata, strlen(somedata)+1);
    fprintf(stderr, "Wrote %d bytes to fifo\n", numbytes);
    
    //close(fifo);
    return 0;
}
