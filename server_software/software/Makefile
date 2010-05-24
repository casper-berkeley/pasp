CC = nvcc
CFLAGS = -g -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/common/inc -L/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/lib
DEBUG = -D DEBUG
CUDA_INCLUDE_LIB = -lcufft -lrt /home/tfiliba/NVIDIA_GPU_Computing_SDK/C/lib/libcutil_x86_64.a
NVCCFLAGS = -g -O2 -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -I/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/common/inc -L/home/tfiliba/NVIDIA_GPU_Computing_SDK/C/lib


PASP_TARGETS := packet_generator pasp_recv pasp_record pasp_process
TEST_TARGETS := pasp_process_test
TARGETS := $(PASP_TARGETS) $(TEST_TARGETS)
.PHONY: pasp_test pasp_rec clean realclean new

SUBDIRS = cudafft

default: pasp_proc

all: $(TARGETS)

pasp_test: packet_generator pasp

pasp_rec_test: packet_generator pasp_rec

pasp_rec: pasp_recv pasp_record

pasp_proc: pasp_recv pasp_distribute pasp_process 




%.o: %.c %.h Makefile pasp_config.h Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	
%.o: %.cu Makefile pasp_config.h Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	
%.o: %.c Makefile pasp_config.h Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	
pasp_process.o: pasp_process.c pasp_process.h fft_library.h Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	
pasp_process_test.o: pasp_process_test.cu fft_library.h Makefile
	$(CC) $(CFLAGS) $(DEBUG) -c $<
	



$(PASP_TARGETS): %: %.o
	$(CC) $(CFLAGS) $(DEBUG) $*.o -o $*

pasp_process: pasp_process.o cudafft/cuda_fft_library.o
	$(CC) $(CFLAGS) $(DEBUG) $(CUDA_INCLUDE_LIB) pasp_process.o cuda_fft_library.o -o pasp_process
	
pasp_process_test:  pasp_process_test.o cuda_fft_library.o
	$(CC) $(NVCCFLAGS) $(DEBUG) $(CUDA_INCLUDE_LIB) pasp_process_test.o cuda_fft_library.o -o pasp_process_test
	
clean: 
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
	done
	@rm -vf *.o

realclean: clean
	@rm -vf $(TARGETS)
	
new: realclean default

