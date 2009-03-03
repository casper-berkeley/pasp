/*
 * file: net2hdd.h
 * auth: William Mallard
 * mail: wjm@berkeley.edu
 * date: 2008-12-22
 */

#ifndef _NET2HDD_H_
#define _NET2HDD_H_

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
#include <dirent.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <signal.h>

#include <pthread.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "ring_buffer.h"
#include "debug_macros.h"

/*
 * Useful typedefs
 */

typedef int socket_t;
typedef struct sockaddr_in SA_in;
typedef struct sockaddr SA;

/*
 * Structure Definitions
 */

typedef struct {
	RING_BUFFER *pkt_buffer;
} NET_THREAD_ARGS;

typedef struct {
	RING_BUFFER *pkt_buffer;
} HDD_THREAD_ARGS;

/*
 * Function Declarations
 */

void *net_thread_function(void *arg);
void *hdd_thread_function(void *arg);
socket_t setup_network_listener();
int open_output_file(const char *path);
void cleanup(int signal);

static uint32_t run_net_thread = 1;
static uint32_t run_hdd_thread = 1;

#endif /* _NET2HDD_H_ */
