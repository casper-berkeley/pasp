#define DEF_CUFFT_FUNC

void exec_part_cufft(cufftComplex *, int, int);
void exec_cufft(cufftComplex *, int, int);
void check_cufft_plan_error(cufftResult );
void check_cufft_exec_error(cufftResult );
void output_cufft_plan_error(char *);
void output_cufft_exec_error(char *);


void exec_part_cufft(cufftComplex *devData, int fftLen, int fftNum){

	// for using cufft
	cufftResult  planResult;
	cufftHandle  plan;
	cufftResult  fftResult;

	// create cufft plan
	planResult = cufftPlan1d(&plan, fftLen, CUFFT_C2C, fftNum/2);
	check_cufft_plan_error(planResult);

	// do fft first
	fftResult = cufftExecC2C(plan, devData, devData, CUFFT_FORWARD);
	check_cufft_exec_error(fftResult);
	cudaThreadSynchronize();

	// do fft second
	fftResult = cufftExecC2C(plan, &devData[fftLen * fftNum / 2], &devData[fftLen * fftNum / 2], CUFFT_FORWARD);
	check_cufft_exec_error(fftResult);
	cudaThreadSynchronize();

	// destroy cufft plan
	cufftDestroy(plan);

	return;
}


void exec_cufft(cufftComplex *devData, int fftLen, int fftNum){

	// for using cufft
	cufftResult  planResult;
	cufftHandle  plan;
	cufftResult  fftResult;

	// create cufft plan
	planResult = cufftPlan1d(&plan, fftLen, CUFFT_C2C, fftNum);
	check_cufft_plan_error(planResult);

	// do fft first
	fftResult = cufftExecC2C(plan, devData, devData, CUFFT_FORWARD);
	check_cufft_exec_error(fftResult);
	cudaThreadSynchronize();

	// destroy cufft plan
	cufftDestroy(plan);

	return;
}



void check_cufft_plan_error(cufftResult result){

	switch(result){
		case CUFFT_SETUP_FAILED:
			output_cufft_plan_error("CUFFT library failed to initialize");
			break;
		case CUFFT_INVALID_SIZE:
			output_cufft_plan_error("The nx parameter is not a supported size");
			break;
		case CUFFT_INVALID_TYPE:
			output_cufft_plan_error("The type parameter is not supported");
			break;
		case CUFFT_ALLOC_FAILED:
			output_cufft_plan_error("Allocation of GPU resources for the plan failed.");
			break;
		default :
			break;
	}

	return;
}


void check_cufft_exec_error(cufftResult result){

	switch(result){
		case CUFFT_SETUP_FAILED :
			output_cufft_exec_error("CUFFT library failed to initialize");
			break;
		case CUFFT_INVALID_PLAN :
			output_cufft_exec_error("The plan parameter is not a valid handle");
			break;
		case CUFFT_INVALID_VALUE :
			output_cufft_exec_error("The idata, odata, and/or direction parameter is not valid.");
			break;
		case CUFFT_EXEC_FAILED :
			output_cufft_exec_error("CUFFT failed to execute the transform on GPU");
			break;
		default:
			break;
	}

	return;
}

void output_cufft_plan_error(char *str){
	fprintf(stderr,"CUFFT PLAN ERROR : %s\n",str);

	return;
}

void output_cufft_exec_error(char *str){
	fprintf(stderr,"CUFFT EXEC ERROR : %s\n",str);

	return;
}

