/*
 ** talker.c -- a datagram "client" demo
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include "pasp_config.h"

#define SERVERPORT "33107"    // the port users will be connecting to

int main(int argc, char *argv[])
{
    int sockfd;
    struct addrinfo hints, *servinfo, *p;
    int rv;
    int numbytes;
    char * buffer;
    int counter;
    char strchar='a';
    FILE *replay;
    int replaymode;
    
    if (argc != 3 && argc != 2) {
        fprintf(stderr,"usage: talker hostname message\n");
        exit(1);
    }
    
    memset(&hints, 0, sizeof hints);
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_DGRAM;
    
    if ((rv = getaddrinfo(argv[1], SERVERPORT, &hints, &servinfo)) != 0) {
        fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
        return 1;
    }
    
    // loop through all the results and make a socket
    for(p = servinfo; p != NULL; p = p->ai_next) {
        if ((sockfd = socket(p->ai_family, p->ai_socktype,
                             p->ai_protocol)) == -1) {
            perror("talker: socket");
            continue;
        }
        
        break;
    }
    
    if (p == NULL) {
        fprintf(stderr, "talker: failed to bind socket\n");
        return 2;
    }
    
    //if we didn't pass a message generate one of length PACKET_SIZE_BYTES
    if (argc == 2)
    {
        buffer = malloc(PACKET_SIZE_BYTES);
        for(counter = 0; counter < PACKET_SIZE_BYTES-1; counter++)
        {
            buffer[counter] = strchar;
            strchar++;
            if(strchar > 'z') strchar = 'a';
        }
        buffer[PACKET_SIZE_BYTES-1] = '\0';
        replaymode = 0;
    }
    //try to open a file for replay
    else
    {
        replay = fopen(argv[2],"r");
        if(replay == NULL)
        {
            fprintf(stderr, "failed to open file: %s using filename as message\n", argv[2]);
            buffer = argv[2];
            replaymode = 0;
        }
        else
        {
            fprintf(stderr, "opened file %s for reading\n", argv[2]);
            buffer = malloc(PACKET_SIZE_BYTES);
            replaymode = 1;
        }
    }
    if (!replaymode)
    {
        if ((numbytes = sendto(sockfd, buffer, PACKET_SIZE_BYTES, 0,
                              p->ai_addr, p->ai_addrlen)) == -1) {
            perror("talker: sendto");
            exit(1);
        }
        printf("talker: sent %d bytes to %s\n", numbytes, argv[1]);
    }
    else
    {
        while(fread(buffer, PACKET_SIZE_BYTES, 1, replay) != 0)
        {
            if ((numbytes = sendto(sockfd, buffer, PACKET_SIZE_BYTES, 0,
                                                 p->ai_addr, p->ai_addrlen)) == -1) {
                perror("talker: sendto");
                exit(1);
            }
            printf("talker: sent %d bytes to %s\n", numbytes, argv[1]);
        }
    }
    
    freeaddrinfo(servinfo);
    close(sockfd);
    
    return 0;
}


