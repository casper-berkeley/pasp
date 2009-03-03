CC = gcc
CFLAGS = -g -Wall
DEBUG = -D DEBUG

TARGETS := create_fifo_test read_fifo_test packet_generator pasp_recv pasp_process
.PHONY: fifo_test pasp_test pasp clean realclean

default: all

all: pasp_test

%.o: %.c %.h
	$(CC) $(CFLAGS) $(DEBUG) -c $<

fifo_test: create_fifo_test read_fifo_test

create_fifo_test: Makefile create_fifo_test.o
	$(CC) $(CFLAGS) $(DEBUG) create_fifo_test.o -o create_fifo_test

read_fifo_test: Makefile read_fifo_test.o
	$(CC) $(CFLAGS) $(DEBUG) read_fifo_test.o -o read_fifo_test

pasp_test: packet_generator pasp

packet_generator: Makefile packet_generator.o
	$(CC) $(CFLAGS) $(DEBUG) packet_generator.o -o packet_generator

pasp: pasp_recv pasp_process

pasp_recv: Makefile pasp_recv.o
	$(CC) $(CFLAGS) $(DEBUG) pasp_recv.o -o pasp_recv

pasp_process: Makefile pasp_process.o
	$(CC) $(CFLAGS) $(DEBUG) pasp_process.o -o pasp_process

clean:
	@rm -vf *.o

realclean: clean
	@rm -vf $(TARGETS)

