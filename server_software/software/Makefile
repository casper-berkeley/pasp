CC = nvcc
CFLAGS = -g -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/common/inc -L/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/lib
DEBUG = -D DEBUG
CUDA_INCLUDE_LIB = -lcufft -lrt /home/tfiliba/NVIDIA_GPU_Computing_SDK/C/lib/libcutil_x86_64.a
NVCCFLAGS = -g -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/common/inc -L/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/lib


PASP_TARGETS := packet_generator pasp_recv pasp_record pasp_process pasp_distribute
TEST_TARGETS := pasp_process_test
TARGETS := $(PASP_TARGETS) $(TEST_TARGETS)
.PHONY: pasp_test pasp_rec clean realclean new

SUBDIRS = cudafft test_FFT

default: pasp_process_test

all: $(TARGETS)

pasp_test: packet_generator pasp

pasp_rec_test: packet_generator pasp_rec

pasp_rec: pasp_recv pasp_record

pasp_proc: pasp_recv pasp_distribute pasp_process 


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
	
japan_pasp_process_test: pasp_process_test.o
	make -C test_FFT
	$(CC) $(NVCCFLAGS) $(DEBUG) $(CUDA_INCLUDE_LIB) pasp_process_test.o test_FFT/libgpufft.a -o japan_pasp_process_test

pasp_process: pasp_process.o
	make -C cudafft
	$(CC) $(CFLAGS) $(DEBUG) $(CUDA_INCLUDE_LIB) cudafft/cuda_fft_library.a cuda_fft_library.o -o pasp_process
	
pasp_process_test:  pasp_process_test.o
	make -C cudafft
	$(CC) $(NVCCFLAGS) $(DEBUG) $(CUDA_INCLUDE_LIB) pasp_process_test.o cudafft/cuda_fft_library.a -o pasp_process_test
	
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

