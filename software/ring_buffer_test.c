/*
 * file: ring_buffer.c
 * auth: William Mallard
 * mail: wjm@berkeley.edu
 * date: 2008-10-20
 */

#include "ring_buffer.h"

int main (int argc, char **argv)
{
	int item_cnt = 4;
	int buf_size = 8192;

	printf("Creating %d byte ring buffer with %d items.\n", buf_size, item_cnt);
	RING_BUFFER *rb = ring_buffer_create(item_cnt, buf_size);
	printf("Success!\n");

	printf("Deleting ring buffer.\n");
	ring_buffer_delete(rb);
	printf("Success!\n");

	return 0;
}
