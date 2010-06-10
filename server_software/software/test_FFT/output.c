int output_spectrum(outputStruct *,int,int);
int init_output_file();
void terminate_output_file();
void output_spectrum_to_file(outputStruct *);

extern int outputFclosePeriod;
extern int signalLength;
extern FILE *outputFilePointer;
extern int  boxcar;
extern int maximumDetectPointInBoxcar;
extern int   outputCounter;
extern char outputFileName[];


int output_spectrum(outputStruct *data, int iter, int flag){

	output_spectrum_to_file(data);


	if(flag==1){
		if(((iter+1)%outputFclosePeriod)==0){
			fclose(outputFilePointer);
			return init_output_file();
		}
	}

	return 1;
}



void output_spectrum_to_file(outputStruct *data){

	int i,j;
	int index;

	for(i=0; i<(signalLength / boxcar); i++){
		index = i * maximumDetectPointInBoxcar;
		for(j=0; j<maximumDetectPointInBoxcar; j++){
			if(data[index + j].power >= 0.0f){
				
				if(data[index + j].index < signalLength/2){
					fwrite( &(data[index+j].index), sizeof(int), 1, outputFilePointer);
					fwrite( &(data[index+j].mean), sizeof(float), 1, outputFilePointer);
					fwrite( &(data[index+j].power), sizeof(float), 1, outputFilePointer);
					fprintf(stderr, "%d\t%f\t%f\n", data[index+j].index, data[index+j].mean,  data[index+j].power);
				} else if(data[index+j].index >= signalLength/2){
					fwrite( &(data[index+j].index), sizeof(int), 1, outputFilePointer);
					fwrite( &(data[index+j].mean), sizeof(float), 1, outputFilePointer);
					fwrite( &(data[index+j].power), sizeof(float), 1, outputFilePointer);
					fprintf(stderr, "%d\t%f\t%f\n", data[index+j].index, data[index+j].mean,  data[index+j].power);
				}
			} else {
				break;
			}
		}
	}

}



void terminate_output_file(){

	fclose(outputFilePointer);

}



int init_output_file(){

	char buf[FILENAME_BUFSIZE];
	int  result;

	result = sprintf(buf,"%d_%s",outputCounter,outputFileName);
	if(result==EOF){
		fprintf(stderr,"Error : sprintf failed in init_output_file()\n");
		return 0;
	}

	outputFilePointer = fopen(buf,"wb");
	//outputFilePointer = fopen(buf,"w");
	if(outputFilePointer==NULL){
		fprintf(stderr,"Error : fopen failed int init_output_file()\n");
		return 0;
	}

	outputCounter++;
	
	return 1;
}
