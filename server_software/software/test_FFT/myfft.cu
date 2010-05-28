#include "myfft.h"

dim3 calc_grid(int);


/*
 * do_myfft
 */
void do_myfft(Complex *data, int batch, int mode){
	FFT16_twiddle_device<<< calc_grid(batch/FFT16_THREAD), FFT16_THREAD>>>(data, batch);
	cudaThreadSynchronize();	
}



/*
 * calc_grid
 */
dim3 calc_grid(int size){
	
	int x,y;

	x = size;
	y = 1;
	while(x>MYFFT_MAX_GRID){
		x /= 2;
		y *= 2;
	}

	return dim3(x, y, 1);
}

