inline __device__ void FFT4(Complex *, Complex *, Complex *, Complex *);

inline __device__ void FFT4(Complex *a0, Complex *a1, Complex *a2, Complex *a3){
    FFT2( a0, a2);
    FFT2( a1, a3);

    *a3 = ComplexMul( *a3, MYFFT_E_1_2);

    FFT2( a0, a1);
    FFT2( a2, a3);
}
