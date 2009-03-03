/*
 *  pasp_config.h
 *  
 *
 *  Created by Terry E. Filiba on 2/24/09.
 *
 */

#include <stdint.h>
#include "debug_macros.h"

#define FILE_NAME           "pol0"
#define DATA_LEN            50

//parameters from the simulink design
#define NUM_IPS             16
#define NUM_CHANNELS        16
#define SAMPLES_PER_PACKET  64

//the packet size will be 64 bits for ever sample plus a 
//64 bit counter and 64 bit channel id
#define NUM_CHANNELS_PER_IP NUM_CHANNELS/NUM_IPS
#define PACKET_SIZE_BITS    (SAMPLES_PER_PACKET*64+64+64)

typedef struct sample{
    uint32_t count[2];
}sample;

typedef struct pasp_packet{
    uint64_t seq_no;
    uint64_t id_no;
    sample samples[SAMPLES_PER_PACKET];
} pasp_packet;




