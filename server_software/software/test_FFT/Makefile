TARGETS = bench_fft bench_fft_th
LIB_TARGETS = libgpufft.a

default: bench_fft libgpufft.a

libgpufft.a : fft_gpu.cu Makefile output.c
#	cc -c random.c
	nvcc -g -c fft_gpu.cu -I /home/mwagner/NVIDIA_GPU_Computing_SDK/C/common/inc /home/mwagner/NVIDIA_GPU_Computing_SDK/C/lib/libcutil_x86_64.a -L /home/mwagner/lib -l cufft -lm -I /usr/local/cuda/include
	ar rcs libgpufft.a fft_gpu.o

bench_fft : bench_fft.cu fft_gpu.cu output.c Makefile
	cc -c random.c
#	nvcc -o output.o output.c
	nvcc -g -c fft_gpu.cu -I /home/mwagner/NVIDIA_GPU_Computing_SDK/C/common/inc /home/mwagner/NVIDIA_GPU_Computing_SDK/C/lib/libcutil_x86_64.a -L /home/mwagner/lib -l cufft -lm -I /usr/local/cuda/include
	nvcc -g -o bench_fft bench_fft.cu fft_gpu.o random.o -I /home/mwagner/NVIDIA_GPU_Computing_SDK/C/common/inc /home/mwagner/NVIDIA_GPU_Computing_SDK/C/lib/libcutil_x86_64.a -L /home/mwagner/lib  -L /usr/local/cuda/lib64/ -lcufft -lm -I /usr/local/cuda/include


bench_fft_th : bench_fft.c fft_gpu_threshold.cu Makefile
	cc -c random.c
	nvcc -g -c fft_gpu_threshold.cu -I /home/mwagner/NVIDIA_GPU_Computing_SDK/C/common/inc /home/mwagner/NVIDIA_GPU_Computing_SDK/C/lib/libcutil_x86_64.a -L /home/mwagner/lib -l cufft -lm -I /usr/local/cuda/include
	cc -Wall -o bench_fft_th bench_fft.c fft_gpu_threshold.o random.o -I /home/mwagner/NVIDIA_GPU_Computing_SDK/C/common/inc /home/mwagner/NVIDIA_GPU_Computing_SDK/C/lib/libcutil_x86_64.a -L /home/mwagner/lib  -L /usr/local/cuda/lib64/ -lcufft -lm -I /usr/local/cuda/include

clean: 
	@rm -vf *.o
	
realclean: clean
	@rm -vf $(LIB_TARGETS)
	@rm -vf $(TARGETS)

