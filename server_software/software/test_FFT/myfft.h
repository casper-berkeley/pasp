#define COS_PI_8  0.923879533f  // cos(1/8 PI)
#define SIN_PI_8  0.382683432f  // sin(1/8 PI)

// copy from "math.h"
#define MYFFT_PI        3.141592653f  // pi
#define MYFFT_PI_2      1.570796326f  // pi/2
#define MYFFT_PI_4      0.785398163f  // pi/4
#define MYFFT_SQRT2     1.414213562f  // sqrt(2)
#define MYFFT_SQRT1_2   0.707106781f  // 1/sqrt(2)

#define MYFFT_E_1_4     make_float2(1.0f, -1.0f)  // e^{-1 * 1/8 * 2 * PI}
#define MYFFT_E_1_2     make_float2(0.0f, -1.0f)  // e^{-1 * 1/2 * PI * i}
#define MYFFT_E_3_4     make_float2(-1.0f, -1.0f) // e^{-1 * 3/8 * 2 * PI}

#define MYFFT_E_1_8  make_float2(  COS_PI_8, -SIN_PI_8 ) // e^{-1 * 1/16 * 2 * PI}
#define MYFFT_E_3_8  make_float2(  SIN_PI_8, -COS_PI_8 ) // e^{-1 * 3/16 * 2 * PI}
#define MYFFT_E_5_8  make_float2( -SIN_PI_8, -COS_PI_8 ) // e^{-1 * 5/16 * 2 * PI}
#define MYFFT_E_7_8  make_float2( -COS_PI_8, -SIN_PI_8 ) // e^{-1 * 7/16 * 2 * PI}
#define MYFFT_E_9_8  make_float2( -COS_PI_8,  SIN_PI_8 ) // e^{-1 * 9/16 * 2 * PI}

/*
 * thread size
 */
#define FFT16_THREAD 64 

/*
 * max grid size
 */
#define MYFFT_MAX_GRID 32 * 1024

/*
 * include program which operate "Complex" data 
 */
#include "complexOp.cu"

/*
 * include fft kernels
 */
#include "fft2.cu"
#include "fft4.cu"
#include "fft16.cu"
