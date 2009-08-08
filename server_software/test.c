#include <stdio.h>
#include <stdint.h>

uint64_t ntohll(uint64_t host_longlong)
{
    int x = 1;
    
    /* little endian */
    if(*(char *)&x == 1)
        return ((((uint64_t)ntohl(host_longlong)) << 32) + ntohl(host_longlong >> 32));
    
    /* big endian */
    else
        return host_longlong;
    
}

int main()
{
    uint64_t channel, pktnum;
    int i;
    FILE * packets = fopen("pasp_recording","r");
    for(i=0; i<10; i++)
    {
    fread(&pktnum, sizeof(uint64_t), 1, packets);
    fread(&channel, sizeof(uint64_t), 1, packets);
    printf("Packet number: %llx Channel number %llx\n", ntohll(pktnum), ntohll(channel));
    fseek(packets, 1024, SEEK_CUR);
    }


    fclose(packets); 
}
