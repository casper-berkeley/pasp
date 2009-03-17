CC = gcc
CFLAGS = -g -Wall
DEBUG = -D DEBUG

TARGETS := packet_generator pasp_recv pasp_record pasp_process
.PHONY: pasp_test pasp pasp_rec clean realclean

default: all

all: pasp_test

%.o: %.c %.h
	$(CC) $(CFLAGS) $(DEBUG) -c $<

pasp_test: packet_generator pasp

packet_generator: Makefile packet_generator.o
	$(CC) $(CFLAGS) $(DEBUG) packet_generator.o -o packet_generator

pasp: pasp_recv pasp_process

pasp_rec: pasp_recv pasp_record

pasp_recv: Makefile pasp_recv.o
	$(CC) $(CFLAGS) $(DEBUG) pasp_recv.o -o pasp_recv

pasp_process: Makefile pasp_process.o
	$(CC) $(CFLAGS) $(DEBUG) pasp_process.o -o pasp_process

pasp_record: Makefile pasp_record.o
	$(CC) $(CFLAGS) $(DEBUG) pasp_record.o -o pasp_record

clean:
	@rm -vf *.o

realclean: clean
	@rm -vf $(TARGETS)
	@rm -vf pol0

