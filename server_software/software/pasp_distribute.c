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

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime_api.h>

#include "pasp_config.h"
#include "pasp_process.h"

#define NX      256
#define BATCH   1

int pasp_channel_enable[NUM_CHANNELS][2];
// output fifo file info
static int channeldata_fifo[CHANNELS_PER_PACKET][2];

static void accumulate_packet(pasp_packet *newpacket,int numchannels)
{
    int i,j;
    // allocate space for accumulated data
    accumulated_channel *accumulated_channels = (accumulated_channel *) calloc(numchannels,sizeof(accumulated_channel));
    
    for(i=0;i<SAMPLES_PER_CHANNEL;i++)
    {
        for(j=0;j<numchannels;j++)
        {
            accumulated_channels[j].pol0_re += newpacket->samples[i][j].pol0_re;
            accumulated_channels[j].pol0_im += newpacket->samples[i][j].pol0_im;
            accumulated_channels[j].pol1_re += newpacket->samples[i][j].pol1_re;
            accumulated_channels[j].pol1_im += newpacket->samples[i][j].pol1_im;
        }
    }
    
    for(i=0;i<numchannels;i++)
    {
        fprintf(stderr,"%d %d %d %d\n", 
                accumulated_channels[i].pol0_re,
                accumulated_channels[i].pol0_im,
                accumulated_channels[i].pol1_re,
                accumulated_channels[i].pol1_im);
    }
}

static void process_packet(pasp_packet *newpacket)
{
    int i;
    single_pol_sample data[CHANNELS_PER_PACKET][2][SAMPLES_PER_CHANNEL];
    
    //cufftComplex data[NX*BATCH];
    
    //copy a single channel into the buffer
    for(i=0;i<SAMPLES_PER_CHANNEL;i++)
    {
        data[0][0][i].re=newpacket->samples[i][0].pol0_re;
        data[0][0][i].im=newpacket->samples[i][0].pol0_im;
    }
    
    write(channeldata_fifo[0][0],data[0][0],SAMPLES_PER_CHANNEL*sizeof(single_pol_sample));
    
    //accumulate_packet(newpacket,CHANNELS_PER_PACKET);
    
    //call fft on the buffer
    //callGPUFFT(data);
}



static int create_output_fifos(int packet_id)
{
    int i,j;
    int ret;
    char output_file_name[CHANNEL_FILE_NAME_SIZE];
    
    //open a fifo for each channel and pol
    for(i=0;i<CHANNELS_PER_PACKET;i++)
    {
        for(j=0;j<2;j++)
        {
            snprintf(output_file_name,CHANNEL_FILE_NAME_SIZE, CHANNEL_FILE_BASE, i+packet_id*CHANNELS_PER_PACKET, j);
            debug_fprintf(stderr, "Creating fifo %s\n",output_file_name);
            ret = mkfifo(output_file_name,0666);
            if(ret == -1)
            {
                if(errno == EEXIST)
                {
                    debug_fprintf(stderr, "File already exists. Will attempt to open.\n");
                }
                else
                {
                    perror("Error creating fifo");
                    return -1;
                }
            }
            if(pasp_channel_enable[i+packet_id*CHANNELS_PER_PACKET][j])
            {
                //open the fifo
                debug_fprintf(stderr, "Opening fifo\n");
                channeldata_fifo[i][j] = open(output_file_name, O_WRONLY);
                if(channeldata_fifo[i][j] == -1)
                {
                    perror("Error opening fifo");
                    return -1;
                }
            }
        }
    }
    return 0;
}


int main(int argc, char *argv[])
{
    // input fifo file info
    int input_fifo;
    int ret;
    
    // buffer for the next packet
    pasp_packet *nextpacket = malloc(PACKET_SIZE_BYTES);
    int numbytes=0;
    struct sigaction newact;
    int numpackets=0;
    long long totalbytes=0;
    long long packet_index;
    long long packet_id;
    
    pasp_channel_enable[8][0]=1;
    
    //set up the signal handler
    newact.sa_handler = cleanup;
    sigemptyset(&newact.sa_mask);
    newact.sa_flags = 0;
    
    //start listening for Ctrl-C
	sigaction(SIGINT, &newact, NULL);
    
    // open the fifo with raw udp data
    debug_fprintf(stderr, "Opening fifo %s\n", RAW_UDP_FILE_NAME);
    input_fifo = open(RAW_UDP_FILE_NAME,O_RDONLY);
    
    //read a packet so we can open the appropriate channel buffers
    numbytes = read(input_fifo, (void *) nextpacket, PACKET_SIZE_BYTES);
    if(numbytes==-1 && run_fifo_read==1)
    {
        perror("Error reading from fifo");
        exit(0);
    }
    packet_index = ntohll(nextpacket->seq_no);
    packet_id = ntohll(nextpacket->id_no);
    
    ret = create_output_fifos(packet_id);
    // if something went wrong when we tried to create the fifos quit
    if(ret==-1)
    {
        exit(0);
    }
    
    
    debug_fprintf(stderr, "Waiting for data\n");
    while(run_fifo_read==1)
    {
        // read packet from fifo
        numbytes = read(input_fifo, (void *) nextpacket, PACKET_SIZE_BYTES);
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
            
            process_packet(nextpacket);
        }
    }
    
    debug_fprintf(stderr, "Received %d packets, %lld bytes\n", numpackets, totalbytes);
    debug_fprintf(stderr, "Closing fifo\n");
    close(input_fifo);
    free(nextpacket);
    return 0;
}

static void cleanup(int signal)
{
    debug_fprintf(stderr, "Ctrl-C received... cleaning up\n");
	run_fifo_read = 0;
}

