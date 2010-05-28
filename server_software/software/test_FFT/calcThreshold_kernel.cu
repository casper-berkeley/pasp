__global__ void calcOverThreshold(float *inData, outputStruct *outData, int threshold, int outArrayWidth, int thread, int area){
	int tid;
	int bid;
	int index;
	int outIndex;
	int iter;
	int s;
	float sum;

	__shared__ float tmp[MAX_THREAD];
	__shared__ int   count;
	__shared__ int   flag;

	tid = threadIdx.x;
	bid = blockIdx.x;
	index = bid * area + tid;


	// 平均の計算
	sum = 0;

	for(iter=0;iter<(area/thread);iter++){
		sum += inData[index + iter * thread];
		__syncthreads();
	}

	tmp[tid] = sum;
	__syncthreads();

	for(s=thread/2; s>0; s=s/2){
		if(tid<s) tmp[tid] += tmp[tid + s];
		__syncthreads();
	}


	// 閾値を超えるbinの抽出の前準備
	sum = tmp[0] / area;
	index = bid * area + tid;
	outIndex = outArrayWidth * bid;
	if(tid==0){
		count = 0;
		flag  = 0;
		outData[outIndex].power = -1.0f;
	}
	__syncthreads();

	for(iter=0;iter<(area/thread);iter++){
		tmp[tid] = 0.0f;
		__syncthreads();

		if(inData[index + iter * thread] > (sum * threshold)){
			tmp[tid] = inData[index + iter * thread];
			count++;
		}
		__syncthreads();

		if((tid==0)&&(count>0)){
			for(s=0;s<thread;s++){
				if(tmp[s]!=0.0f){
					outData[outIndex].index = index + iter * thread + s;
					outData[outIndex].power = tmp[s];
					outData[outIndex].mean = sum;
					outIndex = outIndex + 1;
					if(outIndex==(outArrayWidth * (bid + 1))) {
						flag = 1;
						break;
					}
					outData[outIndex].power = -1.0f;
				}
			}
			count = 0;
		}

		__syncthreads();

		if(flag) return;
	}
	
}
