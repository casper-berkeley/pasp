CC = nvcc
NVIDIA_SDK_DIR = /opt/CUDA/NVIDIA_GPU_Computing_SDK
CFLAGS = -g -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I$(NVIDIA_SDK_DIR)/C/common/inc -L$(NVIDIA_SDK_DIR)/C/lib
DEBUG = -D DEBUG
CUDA_INCLUDE_LIB = -lcufft -lrt $(NVIDIA_SDK_DIR)/C/lib/libcutil_x86_64.a

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
