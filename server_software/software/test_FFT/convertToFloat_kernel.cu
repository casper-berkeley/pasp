/*
 * calc_partialsum_signal_data
 */
__global__ void calc_partialsum_signal_data(char *indata, float *partialSumRe, float *partialSumIm, int thread, int area){
	int tid;
	int bid;
	int index;
	int iter;
	int s;
	float sumRe;
	float sumIm;

	__shared__ float tmp[SUM_MAX_THREAD*2];

	tid = threadIdx.x;
	bid = blockIdx.x;
	index = bid * area * thread + tid;

	sumRe = 0.0f;
	sumIm = 0.0f;

	for(iter=0; iter<area/2; iter++){
		tmp[tid] = indata[index];
		tmp[tid + thread] = indata[index + thread];
		__syncthreads();


		tmp[tid] = tmp[tid] * 16.0f;
		tmp[tid + thread] = tmp[tid + thread] * 16.0f;
		__syncthreads();

		sumRe += ( tmp[2*tid] + 1) / 16.0f;
		sumIm += ( tmp[2*tid+1] + 1) / 16.0f;
		index += thread * 2;
		__syncthreads();
	}

	tmp[tid] = sumRe;
	tmp[tid+thread] = sumIm;
	__syncthreads();

	for(s=thread/2; s>0; s=s/2){
		if(tid<s) {
			tmp[tid] += tmp[tid + s];
			tmp[tid + thread] += tmp[tid + thread + s];
		}
		__syncthreads();
	}

	if(tid==0) {
		partialSumRe[bid] = tmp[tid];
		partialSumIm[bid] = tmp[tid+thread];
	}

}


/*
 * calc_avg_signal_data
 */
__global__ void calc_avg_signal_data(float *partialSum, float *avg, int thread, int size){
	int tid;
	int s;

	__shared__ float tmp[SUM_MAX_THREAD];

	tid = threadIdx.x;

	tmp[tid] = partialSum[tid];
	__syncthreads();

	for(s=thread/2; s>0; s=s/2){
		if(tid<s) tmp[tid] += tmp[tid + s];
		__syncthreads();
	}

	if(tid==0) {
		*avg = (tmp[tid] / size);
	}

}

/*
 * calc_subtract_signal_data
 */
__global__ void calc_subtract_signal_data(char *indata, cufftComplex *data, float *avgRe, float *avgIm, int thread, int area){
	int tid;
	int bid;
	int iter;
	int index;
	int outindex;
	float averageRe;
	float averageIm;

	__shared__ float tmp[SUB_MAX_THREAD*2];

	tid = threadIdx.x;
	bid = blockIdx.x;

	index = bid * thread * area + tid;
	outindex = bid * thread * (area / 2) + tid;
	averageRe = *avgRe;
	averageIm = *avgIm;
	__syncthreads();

	for(iter=0; iter<area/2; iter++){
		tmp[tid] = indata[index];
		tmp[tid + thread] = indata[index + thread];
		__syncthreads();

		tmp[tid] = tmp[tid] * 16.0f;
		tmp[tid+thread] = tmp[tid+thread] * 16.0f;
		__syncthreads();

		tmp[tid] = (tmp[tid] + 1) / 16.0f;
		tmp[tid+thread] = (tmp[tid + thread] + 1) / 16.0f;
		__syncthreads();

		data[outindex].x = tmp[2*tid] - averageRe;
		data[outindex].y = tmp[2*tid+1] - averageIm;
		__syncthreads();

		index += thread*2;
		outindex += thread;
	}	
}

