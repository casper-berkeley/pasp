CC = nvcc
CFLAGS = -g -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/common/inc -L/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/lib
DEBUG = -D DEBUG
CUDA_INCLUDE_LIB = -lcufft -lrt /home/tfiliba/NVIDIA_GPU_Computing_SDK/C/lib/libcutil_x86_64.a
NVCCFLAGS = -g -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/common/inc -L/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/lib

TARGETS = cuffttest
LIB_TARGETS = cuda_fft_library.a
.PHONY: clean

default: cuffttest

%.o: %.cu Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	
%.o: %.c Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	
cuda_fft_library.a: cuda_fft_library.o
	$(CC) $(CFLAGS) $(CUDA_INCLUDE_LIB) $(DEBUG) cuda_fft_library.o -lib -o cuda_fft_library.a

$(TARGETS): %: %.o
	$(CC) $(CFLAGS) $(CUDA_INCLUDE_LIB) $(DEBUG) $*.o -o $*

clean:
	@rm -vf *.o
	
realclean: clean
	@rm -vf $(LIB_TARGETS)
	@rm -vf $(TARGETS)
