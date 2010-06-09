#ifndef PASP_CONFIG_H
#define PASP_CONFIG_H

/*
 *  pasp_config.h
 *  
 *
 *  Created by Terry E. Filiba on 2/24/09.
 *
 */



#include <stdint.h>
#include "debug_macros.h"
#include <cufft.h>

#define RAW_UDP_FILE_NAME           "pipes/raw_udp_pipe"
#define CHANNEL_FILE_BASE   "pipes/channel%d_pol%d_pipe"
#define CHANNEL_FILE_NAME_SIZE  100

//parameters from the simulink design
#define NUM_IPS             4
#define NUM_CHANNELS        16
#define SAMPLES_PER_PACKET  1024

//the packet size will be 64 bits for ever sample plus a 
//64 bit counter and 64 bit channel id
#define PACKET_SIZE_BITS    (SAMPLES_PER_PACKET*64+64+64)
#define PACKET_SIZE_BYTES   (PACKET_SIZE_BITS/8)

#define CHANNELS_PER_PACKET (NUM_CHANNELS/NUM_IPS)
#define SAMPLES_PER_CHANNEL (SAMPLES_PER_PACKET/CHANNELS_PER_PACKET)
#define CHANNEL_BUFFER_SIZE (SAMPLES_PER_CHANNEL*sizeof(cufftComplex))




//4 ips 16 channels
//each sample has 2 pols
typedef struct single_pol_sample{
    int8_t re;
    int8_t im;
}single_pol_sample;

typedef struct dual_pol_sample{
    int8_t pol0_re;
    int8_t pol0_im;
    int8_t pol1_re;
    int8_t pol1_im;
}dual_pol_sample;

//4 ips 16 channels
//each sample has 2 pols
typedef struct accumulated_channel{
    int pol0_re;
    int pol0_im;
    int pol1_re;
    int pol1_im;
} accumulated_channel;

typedef struct pasp_packet{
    uint64_t seq_no;
    uint64_t id_no;
    dual_pol_sample samples[SAMPLES_PER_CHANNEL][CHANNELS_PER_PACKET];
} pasp_packet;

extern int pasp_channel_enable[NUM_CHANNELS][2];




#endif /*PASP_CONFIG_H*/
