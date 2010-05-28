#include "convertToFloat_kernel.cu"
#include "calcThreshold_kernel.cu"

void convert_to_float_exec(char *devSignalData, float *devPartSumRe, float *devPartSumIm, float *devAvgRe, float *devAvgIm, cufftComplex *devFFTData, int signalLength){
	int avgGrid, subGrid;
	int avgThread, subThread;
	int avgArea, subArea;

	avgGrid = SUM_MAX_THREAD;
	avgThread = SUM_MAX_THREAD;
	avgArea   = (signalLength * 2) / (avgGrid * avgThread);

	calc_partialsum_signal_data<<<avgGrid, avgThread>>>(devSignalData, devPartSumRe, devPartSumIm, avgThread, avgArea);
	cudaThreadSynchronize();

	calc_avg_signal_data<<<1,avgThread>>>(devPartSumRe,devAvgRe,avgThread,signalLength);
	cudaThreadSynchronize();

	calc_avg_signal_data<<<1,avgThread>>>(devPartSumIm,devAvgIm,avgThread,signalLength);
	cudaThreadSynchronize();

	subGrid = SUB_MAX_THREAD;
	subThread = SUB_MAX_THREAD;
	subArea = (signalLength * 2) / (subGrid * subThread);

	calc_subtract_signal_data<<<subGrid, subThread>>>(devSignalData, devFFTData, devAvgRe, devAvgIm, subThread, subArea);
	cudaThreadSynchronize();
}



void calc_over_threshold_exec(float *devPowerData, outputStruct *devOutputData, int signalLength, int boxcar, int threshold, int maximumDetectPointInBoxcar){
	int threshThread = MAX_THREAD;
    int threshArea   = boxcar;
    int threshGrid   = signalLength / threshArea;


	calcOverThreshold<<<threshGrid, threshThread>>>(devPowerData, devOutputData, threshold, maximumDetectPointInBoxcar, threshThread, threshArea);
	cudaThreadSynchronize();
}

