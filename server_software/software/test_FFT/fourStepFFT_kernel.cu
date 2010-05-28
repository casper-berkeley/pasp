__global__ void power_transpose_four_step(cufftComplex *idata, float *odata, int width, int height){

	__shared__ float block[FOUR_STEP_TRANSPOSE_DIM][FOUR_STEP_TRANSPOSE_DIM+1];
	
	// read the matrix tile into shared memory
	unsigned int xIndex = blockIdx.x * FOUR_STEP_TRANSPOSE_DIM + threadIdx.x;
	unsigned int yIndex = blockIdx.y * FOUR_STEP_TRANSPOSE_DIM + threadIdx.y;
	
	cufftComplex tmp;

	if((xIndex < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex;
		tmp = idata[index_in];

		block[threadIdx.y][threadIdx.x] = tmp.x * tmp.x + tmp.y * tmp.y;
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	xIndex = blockIdx.y * FOUR_STEP_TRANSPOSE_DIM + threadIdx.x;
	yIndex = blockIdx.x * FOUR_STEP_TRANSPOSE_DIM + threadIdx.y;
	if((xIndex < height) && (yIndex < width))
	{
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[threadIdx.x][threadIdx.y];
	}	
}

