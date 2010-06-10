NVIDIA_SDK_DIR = /opt/CUDA/NVIDIA_GPU_Computing_SDK
NVCCFLAGS = -g -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64/ -I$(NVIDIA_SDK_DIR)/C/common/inc $(NVIDIA_SDK_DIR)/C/lib/libcutil_x86_64.a -lcufft -lm 
TARGETS = bench_fft bench_fft_th
LIB_TARGETS = libgpufft.a

default: bench_fft libgpufft.a

libgpufft.a : fft_gpu.cu Makefile output.c
	nvcc $(NVCCFLAGS) fft_gpu.cu -c fft_gpu.o
	ar rcs libgpufft.a fft_gpu.o

bench_fft : bench_fft.cu fft_gpu.cu output.c Makefile
	cc -c random.c
	nvcc $(NVCCFLAGS) -c fft_gpu.cu
	nvcc $(NVCCFLAGS) -o bench_fft bench_fft.cu fft_gpu.o random.o


bench_fft_th : bench_fft.cu fft_gpu_threshold.cu Makefile
	cc -c random.c
	nvcc $(NVCCFLAGS) -c fft_gpu_threshold.cu
	nvcc $(NVCCFLAGS) -o bench_fft_th bench_fft.cu fft_gpu_threshold.o random.o

clean: 
	@rm -vf *.o
	
realclean: clean
	@rm -vf $(LIB_TARGETS)
	@rm -vf $(TARGETS)

