#define FOUR_STEP_TRANSPOSE_DIM 16

#include "myfft.cu"
#include "cufftFunc.cu"
#include "fourStepFFT_kernel.cu"

// prototype decleration
void power_transpose_four_step_exec(cufftComplex *FFTData, float *powerData, int x, int y);


/*
 * do_four_step_fft
 */
void do_four_step_fft(cufftComplex *devFFTData, float *devPowerData, int matrixX, int matrixY){

	// FFT along column
	do_myfft(devFFTData, matrixX, 1);

	// FFT along row
	exec_cufft(devFFTData, matrixX, matrixY);

	// matrix transpose with calculating power spectrum
	power_transpose_four_step_exec(devFFTData, devPowerData, matrixX, matrixY);
}



/*
 * power_transpose_four_step_exec
 */
void power_transpose_four_step_exec(cufftComplex *FFTData, float *powerData, int x, int y){
	int width;
	int iter;
	int i;

	iter = 1;
	width = x / FOUR_STEP_TRANSPOSE_DIM;
	while(width>MAX_GRID){
		width /= 2;
		iter  *= 2;
	}

	dim3 grid( width, y / FOUR_STEP_TRANSPOSE_DIM, 1);
	dim3 threads( FOUR_STEP_TRANSPOSE_DIM, FOUR_STEP_TRANSPOSE_DIM, 1);

	for(i=0; i<iter; i++){
		power_transpose_four_step<<<grid, threads>>>(&FFTData[width * FOUR_STEP_TRANSPOSE_DIM * i], &powerData[FOUR_STEP_TRANSPOSE_DIM * FOUR_STEP_TRANSPOSE_DIM * width * i], x, y);
		cudaThreadSynchronize();
	}
}

