/*
 *  pasp_process.h
 *  
 *
 *  Created by Terry E. Filiba on 2/25/09.
 *  Copyright 2009 __MyCompanyName__. All rights reserved.
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

static void cleanup(int signal);

static uint32_t run_fifo_read = 1;

