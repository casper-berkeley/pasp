



void output_spectrum_to_file(FILE *outputFilePointer, int signalLength, char *hostSignalData, float *hostPowerData, float *hostcuFFTRData){
	int i;
	for(i=0; i<signalLength; i++){
        	fprintf(outputFilePointer,"%d %d %f %f\n", i, hostSignalData[i], hostPowerData[i], hostcuFFTRData[i]); 
	}
}


void output_spectrum_to_file_float(FILE *outputFilePointer, int signalLength, cufftComplex* hostcuFFTData, float *hostPowerData, cufftReal *hostFFTData){
//void output_spectrum_to_file_float(FILE *outputFilePointer, int signalLength, cufftComplex* hostcuFFTData, float *hostPowerData, float *hostcuFFTRData){
	int i;
	for(i=0; i<signalLength; i++){
        	// fprintf(outputFilePointer,"%d %d %f %f %f\n", i, hostSignalData[i], hostPowerData[i], hostcuFFTData[i].x,hostcuFFTData[i].y); 
        	fprintf(outputFilePointer,"%d %f %f %f %f\n", i, hostcuFFTData[i].x, hostcuFFTData[i].y, hostPowerData[i], hostFFTData[i]); 
	}
}


void output_spectrum_to_file_float_threshold(FILE *outputFilePointer, int signalLength, cufftComplex* hostFFTData, float *hostPowerData, outputStruct *data, int boxcar, int maximumDetectPointInBoxcar){
	int i, j, index;
	for(i=0; i<signalLength; i++){
          fprintf(outputFilePointer,"%d %f %f %f\n", i, hostFFTData[i].x, hostFFTData[i].y, hostPowerData[i]); 
	}

	for(i=0; i<(signalLength / boxcar); i++){
                index = i * maximumDetectPointInBoxcar;
                for(j=0; j<maximumDetectPointInBoxcar; j++){
                        if(data[index + j].power >= 0.0f){
                                if(data[index + j].index < signalLength/2){
                                        //fwrite( &(data[index+j].index), sizeof(int), 1, outputFilePointer);
                                        //fwrite( &(data[index+j].mean), sizeof(float), 1, outputFilePointer);
                                        //fwrite( &(data[index+j].power), sizeof(float), 1, outputFilePointer);
                                        fprintf(outputFilePointer, "%d %f %f\n", data[index+j].index, data[index+j].mean,  data[index+j].power);
                                } else if(data[index+j].index >= signalLength/2){
                                        //fwrite( &(data[index+j].index), sizeof(int), 1, outputFilePointer);
                                        //fwrite( &(data[index+j].mean), sizeof(float), 1, outputFilePointer);
                                        //fwrite( &(data[index+j].power), sizeof(float), 1, outputFilePointer);
                                        fprintf(outputFilePointer, "%d %f %f\n", data[index+j].index, data[index+j].mean,  data[index+j].power);
                                }
                        } else {
                                break;
                        }
                }
        }
}





void terminate_output_file(FILE *outputFilePointer){

	fclose(outputFilePointer);

}


