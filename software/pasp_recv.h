/*
 *  pasp_recv.h
 *  
 *
 *  Created by Terry Filiba on 2/24/09.
 *
 */

#ifndef _PASP_RECV_H_
#define _PASP_RECV_H_

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


/*
 * Useful typedefs
 */

typedef int socket_t;
//typedef struct sockaddr_in SA_in;
//typedef struct sockaddr SA;

/*
 * Function Declarations
 */

void receive_packets(int fifo);
socket_t setup_network_listener();
static void cleanup(int signal);

static uint32_t run_net_thread = 1;

#endif /* _PASP_RECV_H_ */
