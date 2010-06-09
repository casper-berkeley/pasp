CC = nvcc
CFLAGS = -g -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/common/inc -L/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/lib
DEBUG = -D DEBUG
CUDA_INCLUDE_LIB = -lcufft -lrt /home/tfiliba/NVIDIA_GPU_Computing_SDK/C/lib/libcutil_x86_64.a
NVCCFLAGS = -g -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/common/inc -L/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/lib


PASP_TARGETS := packet_generator pasp_recv pasp_record  pasp_distribute
PROCESSING_TARGETS := pasp_process japan_pasp_process
TEST_TARGETS := pasp_process_test japan_pasp_process_test
TARGETS := $(PASP_TARGETS) $(PROCESSING_TARGETS) $(TEST_TARGETS)
.PHONY: pasp_test pasp_rec clean realclean new

SUBDIRS = cudafft test_FFT

default: cuda_pasp_processor japan_pasp_processor

all: $(TARGETS)

pasp_test: packet_generator pasp

pasp_rec_test: packet_generator pasp_rec

pasp_rec: pasp_recv pasp_record

cuda_pasp_processor: pasp_recv pasp_distribute cuda_pasp_process 

japan_pasp_processor: pasp_recv pasp_distribute japan_pasp_process 


%.o: %.c %.h pasp_config.h Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	
%.o: %.cu pasp_config.h Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	
%.o: %.c pasp_config.h Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	
pasp_process.o: pasp_process.cu pasp_process.h fft_library.h Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	
pasp_process_test.o: pasp_process_test.cu fft_library.h Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	



$(PASP_TARGETS): %: %.o
	$(CC) $(CFLAGS) $(DEBUG) $*.o -o $*
	
japanfftlib:
	make -C test_FFT libgpufft.a
	
japan_pasp_process: pasp_process.o japanfftlib
	$(CC) $(NVCCFLAGS) $(DEBUG) $(CUDA_INCLUDE_LIB) pasp_process.o test_FFT/libgpufft.a -o japan_pasp_process
	
japan_pasp_process_test: pasp_process_test.o japanfftlib
	$(CC) $(NVCCFLAGS) $(DEBUG) $(CUDA_INCLUDE_LIB) pasp_process_test.o test_FFT/libgpufft.a -o japan_pasp_process_test

cudafftlib:
		make -C cudafft cuda_fft_library.a

cuda_pasp_process: pasp_process.o cudafftlib
	$(CC) $(CFLAGS) $(DEBUG) $(CUDA_INCLUDE_LIB) pasp_process.o cudafft/cuda_fft_library.a -o cuda_pasp_process
	
cuda_pasp_process_test:  pasp_process_test.o cudafftlib
	$(CC) $(NVCCFLAGS) $(DEBUG) $(CUDA_INCLUDE_LIB) pasp_process_test.o cudafft/cuda_fft_library.a -o cuda_pasp_process_test
	
# delete all object files
clean: 
	-for d in $(SUBDIRS); do make clean -C $$d; done
	@rm -vf *.o

# delete all object and binary files
realclean:
	-for d in $(SUBDIRS); do make realclean -C $$d; done
	@rm -vf *.o
	@rm -vf $(TARGETS)
	
# recompile everything
new: realclean default

