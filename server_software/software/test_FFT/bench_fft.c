/*
 * singleGpuSpectrometer
 * 
 * Version 2.0, April 12 2010
 *
 * This program was written by Hirofumi Kondo at the Supercomputing Engineering Laboratory,
 * Graduate School of Information Science and Technology, Osaka University, Japan.
 *
 * Copyright 2010 Supercomputing Engineering Laboratory, Graduate School of Information
 * Science and Technology, Osaka University, Japan
 *
 *
 * Compile : 
 *   nvcc -o singleGpuSpectrometer singleGpuSpectrometer.cu -I /usr/local/cuda/NVIDIA_GPU_Computing_SDK/common/inc
 *                                                             /usr/local/cuda/NVIDIA_GPU_Computing_SDK/C/lib/libcutil.a
 *                                                          -L /usr/local/cuda/lib -l cufft
 *
 * Usage : ./singleGpuSpectrometer [options]
 *   -length           : signal length of this spectrometer handle in M-points 
 *   -boxcar           : length of boxcar for smoothing
 *   -threshold        : value of threshold
 *   -max_detect_point : value of maximum detected points over threshold in each boxcar
 *   -output_file      : filename of output file
 *
 * Output file format :
 *   The file format is binary format.
 *   The output file records all spikes whose power exceed (boxcar_mean) * (threashold).
 *   The file contains 3 data
 *     1) index of signal
 *     2) the power of signal
 *     3) mean power of boxcar which the signal is in
 * 
 * Special Instruction
 *   1) Memory capacity
 *     The memory capacity that this GPU spectrometer requires is changed by the signal length.
 *     If you want to analyze 128M-points signal, GPU has to have 4GB VRAM.
 *     The maximum length that 1GB VRAM GPU can handle is 32M-points.
 *
 *   2) CUDA
 *     We recommend that you use CUDA 2.3 and CUFFT 2.3.
 *     This is not necessary condition.
 *     But the execution time is wrong if you use CUDA 2.2 and CUFFT 2.2.
 */


// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <cuda.h>
//#include <cutil_inline.h>
//#include <cufft.h>

//#include "bench_fft.h"


//#include "output_greg.c"


/* 
 * Prototype declaration
 */

void do_analyze_on_gpu();



/*
 * Program main
 */
int main(int argc, char** argv){
 

    int i;
    int signalLength = 1024 * 1024;


    for (i=1;i<argc;i++) {
        if (!strcmp(argv[i], "-length")) {
        	signalLength = atoi(argv[++i]) * 1024 * 1024;
        } /*else if (!strcmp(argv[i], "-boxcar")){
		boxcar = atoi(argv[++i]);
	} else if (!strcmp(argv[i], "-threshold")){
		threshold = atoi(argv[++i]);
	} else if (!strcmp(argv[i], "-max_detect_point")){
		maximumDetectPointInBoxcar = atoi(argv[++i]);
	} else if (!strcmp(argv[i], "-output_file")){
		strncpy(outputFileName, argv[++i], FILENAME_BUFSIZE);
		if(outputFileName[FILENAME_BUFSIZE-1]!='\0'){
			fprintf(stderr,"Error : Too long output file name. maximum length = %d\n", FILENAME_BUFSIZE-1);
			exit(-1);
		}
	} else {
		fprintf(stderr,"Error : wrong argument\n");
        }*/
    }

    // Analyze signal on GPU
    printf("Calling GPU\n");
    do_analyze_on_gpu(signalLength);

    return 0;
}



