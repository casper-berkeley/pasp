inline __device__ void FFT2(Complex *, Complex *);

inline __device__ void FFT2(Complex *a0, Complex *a1){
    Complex tmp;

    tmp = *a0;
    *a0 = ComplexAdd(tmp, *a1);
    *a1 = ComplexSub(tmp, *a1);
}
