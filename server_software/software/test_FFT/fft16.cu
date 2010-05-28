inline __device__ void FFT16(Complex *);


__global__ void FFT16_device(Complex *data, int batch){
    
    int     tid = threadIdx.x;
    Complex a[16];
    int     i;

    // calc initial memory address
    data += (blockIdx.y * gridDim.x + blockIdx.x) * FFT16_THREAD + tid;

    // load data
#pragma unroll
    for(i=0; i<16; i++){
        a[i] = data[i*batch];
    }

    // do 16-pt fft
    FFT16(a);

    // store data
	data[0] = a[0];
	data[1*batch] = a[8];
	data[2*batch] = a[4];
	data[3*batch] = a[12];
	data[4*batch] = a[2];
	data[5*batch] = a[10];
	data[6*batch] = a[6];
	data[7*batch] = a[14];
	data[8*batch] = a[1];
	data[9*batch] = a[9];
	data[10*batch] = a[5];
	data[11*batch] = a[13];
	data[12*batch] = a[3];
	data[13*batch] = a[11];
	data[14*batch] = a[7];
	data[15*batch] = a[15];
}



__global__ void FFT16_twiddle_device(Complex *data, int batch){
    
    int     tid = threadIdx.x;
    Complex a[16];
    int     i;
	int     column;
	float   theta;

    // calc initial memory address
	column = (blockIdx.y * gridDim.x + blockIdx.x) * FFT16_THREAD + tid;
    data += column;


    // load data
#pragma unroll
	for(i=0; i<16; i++){
        a[i] = data[i*batch];
    }

    // do 16-pt fft
    FFT16(a);

	// multiply twiddle factor
	theta = MYFFT_PI * -2.0f * column / (16.0f * batch);

    // store data
	data[0] = a[0];
	data[1*batch] = ComplexMul(a[8], make_float2(__cosf(theta), __sinf(theta)));
	data[2*batch] = ComplexMul(a[4], make_float2(__cosf(theta * 2), __sinf(theta * 2)));
	data[3*batch] = ComplexMul(a[12], make_float2(__cosf(theta * 3), __sinf(theta * 3)));
	data[4*batch] = ComplexMul(a[2], make_float2(__cosf(theta * 4), __sinf(theta * 4)));;
	data[5*batch] = ComplexMul(a[10], make_float2(__cosf(theta * 5), __sinf(theta * 5)));
	data[6*batch] = ComplexMul(a[6], make_float2(__cosf(theta * 6), __sinf(theta * 6)));
	data[7*batch] = ComplexMul(a[14], make_float2(__cosf(theta * 7), __sinf(theta * 7)));
	data[8*batch] = ComplexMul(a[1], make_float2(__cosf(theta * 8), __sinf(theta * 8)));
	data[9*batch] = ComplexMul(a[9], make_float2(__cosf(theta * 9), __sinf(theta * 9)));
	data[10*batch] = ComplexMul(a[5], make_float2(__cosf(theta * 10), __sinf(theta * 10)));
	data[11*batch] = ComplexMul(a[13], make_float2(__cosf(theta * 11), __sinf(theta * 11)));
	data[12*batch] = ComplexMul(a[3], make_float2(__cosf(theta * 12), __sinf(theta * 12)));
	data[13*batch] = ComplexMul(a[11], make_float2(__cosf(theta * 13), __sinf(theta * 13)));
	data[14*batch] = ComplexMul(a[7], make_float2(__cosf(theta * 14), __sinf(theta * 14)));
	data[15*batch] = ComplexMul(a[15], make_float2(__cosf(theta * 15), __sinf(theta * 15)));
}



inline __device__ void FFT16(Complex *a){
    FFT4( &a[0], &a[4], &a[8], &a[12]);
    FFT4( &a[1], &a[5], &a[9], &a[13]);
    FFT4( &a[2], &a[6], &a[10], &a[14]);
    FFT4( &a[3], &a[7], &a[11], &a[15]);

    a[5] = ComplexMul(a[5], MYFFT_E_1_4);
    a[5] = ComplexScale(a[5], MYFFT_SQRT1_2);

    a[6] = ComplexMul(a[6], MYFFT_E_1_2);

    a[7] = ComplexMul(a[7], MYFFT_E_3_4);
    a[7] = ComplexScale(a[7], MYFFT_SQRT1_2);

    a[9] = ComplexMul(a[9], MYFFT_E_1_8);

    a[10] = ComplexMul(a[10], MYFFT_E_1_4);
    a[10] = ComplexScale(a[10], MYFFT_SQRT1_2);

    a[11] = ComplexMul(a[11], MYFFT_E_3_8);

    a[13] = ComplexMul(a[13], MYFFT_E_3_8);

    a[14] = ComplexMul(a[14], MYFFT_E_3_4);
    a[14] = ComplexScale(a[14], MYFFT_SQRT1_2);

    a[15] = ComplexMul(a[15], MYFFT_E_9_8);
    
    FFT4( &a[0], &a[1], &a[2], &a[3]);
    FFT4( &a[4], &a[5], &a[6], &a[7]);
    FFT4( &a[8], &a[9], &a[10], &a[11]);
    FFT4( &a[12], &a[13], &a[14], &a[15]);    
}


