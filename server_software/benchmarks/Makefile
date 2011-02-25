CC = nvcc
NVIDIA_SDK_DIR = /opt/CUDA/NVIDIA_GPU_Computing_SDK
CUTIL_LIB_NAME = cutil_x86_64
CFLAGS = -g -O2
#DEBUG = -D DEBUG
CUDA_INCLUDE_HEADER = -I/usr/local/cuda/include -I$(NVIDIA_SDK_DIR)/C/common/inc 
CUDA_INCLUDE_LIB = -L/usr/local/cuda/lib64 -L$(NVIDIA_SDK_DIR)/C/lib -lcufft -lcuda -l$(CUTIL_LIB_NAME)

TARGETS = cuffttest
LIB_TARGETS = cuda_fft_library.a
.PHONY: clean

default: cuffttest

%.o: %.cu Makefile
	$(CC) $(CFLAGS) $(CUDA_INCLUDE_HEADER) $(DEBUG) -c $<
	
%.o: %.c Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	
cuda_fft_library.a: cuda_fft_library.o
	ar rcs cuda_fft_library.a cuda_fft_library.o 

$(TARGETS): %: %.o
	$(CC) $(CFLAGS) $(CUDA_INCLUDE_LIB) $(DEBUG) $*.o -o $*

clean:
	@rm -vf *.o
	
realclean: clean
	@rm -vf $(LIB_TARGETS)
	@rm -vf $(TARGETS)
