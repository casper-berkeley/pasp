/*
 *  pasp_recv.c
 *  
 *
 *  Created by Terry Filiba on 2/24/09.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <dirent.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <signal.h>

#define MYPORT "33107"	// the port users will be connecting to
#define PACKET_SIZE 1024
#define PACKET_SIZE_BYTES 8192


static void cleanup(int signal);
void receive_packets(int fifo);

static uint32_t run_net_thread = 1;
typedef int socket_t;

#define ntohll(x)   ((((uint64_t)ntohl(x)) << 32) + ntohl(x >> 32))

// get sockaddr, IPv4 or IPv6:
void *get_in_addr(struct sockaddr *sa)
{
	if (sa->sa_family == AF_INET) {
		return &(((struct sockaddr_in*)sa)->sin_addr);
	}
    
	return &(((struct sockaddr_in6*)sa)->sin6_addr);
}

int main()
{
    int fifo;
    int ret;
    struct sigaction newact;
    
    //create the fifo
    //    debug_fprintf(stderr, "Creating fifo %s\n", RAW_UDP_FILE_NAME);
    //    ret = mkfifo(RAW_UDP_FILE_NAME,0666);
    //    if(ret == -1)
    //    {
    //        if(errno == EEXIST)
    //        {
    //            debug_fprintf(stderr, "File already exists. Will attempt to open.\n");
    //        }
    //        else
    //        {
    //            perror("Error creating fifo");
    //            exit(1);
    //        }
    //    }
    //    
    //    //open the fifo
    //    debug_fprintf(stderr, "Opening fifo %s\n", RAW_UDP_FILE_NAME);
    //    fifo = open(RAW_UDP_FILE_NAME, O_WRONLY);
    //    if(fifo == -1)
    //    {
    //        perror("Error opening fifo");
    //        exit(1);
    //    }
    
    //set up the signal handler
    newact.sa_handler = cleanup;
    sigemptyset(&newact.sa_mask);
    newact.sa_flags = 0;
    
    //start listening for Ctrl-C
    sigaction(SIGINT, &newact, NULL);
    
    //receive packets and write into the fifos
    receive_packets(fifo);
    
    fprintf(stderr, "Closing fifo\n");
    close(fifo);
    return 0;
}

/*
 * Read data from the network.
 * Write data to ring buffer.
 */
void receive_packets(int fifo)
{
    
	socket_t sock = setup_network_listener();
	int numbytes, bytesinfifo;
	struct sockaddr_storage their_addr;
	char buf[PACKET_SIZE_BYTES];
	size_t addr_len;
	//char s[INET6_ADDRSTRLEN];
    int numpackets=0;
    long long totalbytes=0;
    long long currentpacket=0;
    long long newpacket=0;
    long long errors=0;
    int i;
    
    
	fprintf(stderr, "Entering network thread loop.\n");
    
	/*
	 * loop forever:
	 *   read data from network
	 */
    while(run_net_thread)
    {
        //printf("listener: waiting to recvfrom...\n");
        
        if ((numbytes = recvfrom(sock, buf, PACKET_SIZE_BYTES , 0,
                                 (struct sockaddr *)&their_addr, (socklen_t *) &addr_len)) == -1) {
            if(errno==EINTR && run_net_thread==0)
            {
                perror("Interrupt received and handled");
            }
            else
            {
                perror("Unable to receive packet");
                exit(1);
            }
        }
        else
        {
            /*
             debug_fprintf(stderr, "[net thread] Received %d bytes.\n", numbytes);
             debug_fprintf(stderr, "listener: got packet from %s\n",
             inet_ntop(their_addr.ss_family,
             get_in_addr((struct sockaddr *)&their_addr),
             s, sizeof s));
             debug_fprintf(stderr,"listener: packet is %d bytes long\n", numbytes);
             debug_fprintf(stderr,"listener: packet contains \"%s\"\n", buf);
             */
            
            //send packets over the fifo
            //bytesinfifo = write(fifo, buf, numbytes);
            //debug_fprintf(stderr, "wrote %d bytes to fifo\n", bytesinfifo);
            numpackets++;
            totalbytes+=numbytes;
            newpacket = ntohll(((uint64_t *) buf)[PACKET_SIZE_BYTES/8-1]);
            if(currentpacket+1 != newpacket)
            {
                errors++;
            }
            currentpacket = newpacket;
            
        }
    }
    for(i=0;i<PACKET_SIZE;i++)
    {
        fprintf(stderr, "%lld ", ntohll(((uint64_t *) buf)[i]));
    }
    
	fprintf(stderr, "Exiting network thread loop.\n");
    fprintf(stderr, "Received %d packets, %lld bytes, %lld errors\n", numpackets, totalbytes, errors);
    
	close(sock);
    
	return;
}

/*
 * Bind to a port and listen for incoming data.
 */
int setup_network_listener()
{
	int sockfd;
	struct addrinfo hints, *servinfo, *p;
	int rv;
    
	memset(&hints, 0, sizeof hints);
	hints.ai_family = AF_UNSPEC; // set to AF_INET to force IPv4
	hints.ai_socktype = SOCK_DGRAM;
	hints.ai_flags = AI_PASSIVE; // use my IP
    
	if ((rv = getaddrinfo(NULL, MYPORT, &hints, &servinfo)) != 0) {
		fprintf(stderr, "getaddrinfo: %s\n", gai_strerror(rv));
		return 1;
	}
    
	// loop through all the results and bind to the first we can
	for(p = servinfo; p != NULL; p = p->ai_next) {
		if ((sockfd = socket(p->ai_family, p->ai_socktype,
                             p->ai_protocol)) == -1) {
			perror("listener: socket");
			continue;
		}
        
		if (bind(sockfd, p->ai_addr, p->ai_addrlen) == -1) {
			close(sockfd);
			perror("listener: bind");
			continue;
		}
        
		break;
	}
    
	if (p == NULL) {
		fprintf(stderr, "listener: failed to bind socket\n");
		return 2;
	}
    
	freeaddrinfo(servinfo);    
    
	return sockfd;
}


static void cleanup(int signal)
{
    fprintf(stderr, "Ctrl-C received... cleaning up\n");
	run_net_thread = 0;
}
