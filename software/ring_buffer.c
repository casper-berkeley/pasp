/* file: ring_buffer.c
 * auth: William Mallard
 * mail: wjm@berkeley.edu
 * date: 2008-04-02
 */

#include "ring_buffer.h"

/*
 * Construct and initialize a RING_BUFFER.
 */
RING_BUFFER *ring_buffer_create(size_t item_count, size_t buf_size)
{
	// create buffer
	void *buffer = create_data_buffer(buf_size);

	// create list items
	RING_ITEM *head_item = (RING_ITEM *)calloc(item_count, sizeof(RING_ITEM));
	int i;
	for(i=0; i<item_count; i++)
	{
		RING_ITEM *this_item = &head_item[i];
		RING_ITEM *next_item = &head_item[(i + 1) % item_count];

		this_item->next = next_item;
		sem_init(&this_item->write_mutex, 0, 1);
		sem_init(&this_item->read_mutex, 0, 0);
		this_item->data = buffer;
		this_item->size = 0;
	}

	// create ring buffer
	RING_BUFFER *rb = (RING_BUFFER *)calloc(1, sizeof(RING_BUFFER));
	rb->buffer_ptr = buffer;
	rb->buffer_size = buf_size;
	rb->list_ptr = head_item;
	rb->list_length = item_count;
	rb->write_ptr = head_item;
	rb->read_ptr = head_item;

	return rb;
}

void ring_buffer_delete(RING_BUFFER *rb)
{
	int status = 0;

	// delete mmap'd buffer
	void *buffer = rb->buffer_ptr;
	size_t buf_size = rb->buffer_size;
	memset(buffer, 0, buf_size << 1);
	status = munmap(buffer, buf_size << 1);
	if (status)
	{
		// ERROR
	}
	buffer = NULL;

	// delete list items
	RING_ITEM *head_item = rb->list_ptr;
	size_t item_count = rb->list_length;
	int i;
	for(i=0; i<item_count; i++)
	{
		RING_ITEM *this_item = &head_item[i];

		sem_destroy(&this_item->write_mutex);
		sem_destroy(&this_item->read_mutex);
		memset(this_item, 0, sizeof(RING_ITEM));
	}
	free(head_item);
	head_item = NULL;

	// delete ring buffer
	memset(rb, 0, sizeof(RING_BUFFER));
	free(rb);
	rb = NULL;
}

void *create_data_buffer(size_t size)
{
	void *buffer = NULL;

	buffer = create_primary_buffer(size << 1);
	create_shadow_buffer(buffer, size);
	memset(buffer, 0, size << 1);

	return buffer;
}

void *create_primary_buffer(size_t buf_size)
{
	void *buf_addr = NULL;

	void *addr = NULL;
	size_t len = -1;
	int prot = -1;
	int flags = -1;
	int fd = -1;
	off_t off = -1;

	addr = NULL;
	len = buf_size;
	prot = PROT_NONE;
	flags = MAP_PRIVATE | MAP_ANON;
	fd = -1;
	off = 0;

	buf_addr = mmap(addr, len, prot, flags, fd, off);
	if (buf_addr == MAP_FAILED)
	{
		// ERROR
	}

	return buf_addr;
}

void create_shadow_buffer(void *buf_addr, size_t buf_size)
{
	void *shm_addr = NULL;
	int status = 0;

	void *addr = NULL;
	size_t len = -1;
	int prot = -1;
	int flags = -1;
	int fd = -1;
	off_t off = -1;

	// create shadow buffers
	len = buf_size;
	prot = PROT_READ | PROT_WRITE;
	flags = MAP_SHARED | MAP_FIXED;
	fd = create_anonymous_file(buf_size);
	off = 0;

	addr = buf_addr;
	shm_addr = mmap(addr, len, prot, flags, fd, off);
	if (shm_addr != addr)
	{
		// ERROR
	}

	addr = buf_addr + buf_size;
	shm_addr = mmap(addr, len, prot, flags, fd, off);
	if (shm_addr != addr)
	{
		// ERROR
	}

	status = close(fd);
	if (status)
	{
		// ERROR
	}
}

int create_anonymous_file(size_t size)
{
	char path[] = "/dev/shm/ring-buffer-XXXXXX";
	int fd = -1;
	int status = 0;

	// create a unique file in memory
	fd = mkstemp(path);
	if (fd < 0)
	{
		// ERROR
	}

	// unlink file from file system
	status = unlink(path);
	if (status)
	{
		// ERROR
	}

	// truncate file to exact buffer size
	status = ftruncate(fd, size);
	if (status)
	{
		// ERROR
	}

	return fd;
}
