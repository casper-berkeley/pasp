/*
 * define new type
 */
typedef cufftComplex Complex;

// Complex addition
__device__ inline Complex ComplexAdd(Complex a, Complex b){
    Complex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

// Complex subtraction
__device__ inline Complex ComplexSub(Complex a, Complex b){
    Complex c;
    c.x = a.x - b.x;
    c.y = a.y - b.y;
    return c;
}

// Complex scale
__device__ inline Complex ComplexScale(Complex a, float s)
{
    Complex c;
    c.x = s * a.x;
    c.y = s * a.y;
    return c;
}

// Complex multiplication
__device__ inline Complex ComplexMul(Complex a, Complex b)
{
    Complex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

