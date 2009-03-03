/*
 * file: net2hdd.c
 * auth: William Mallard
 * mail: wjm@berkeley.edu
 * date: 2008-12-22
 */

#include "net2hdd.h"

#define LISTEN_PORT 8888
#define RX_BUFFER_SIZE 81920
#define MAX_PAYLOAD_LEN 8192
#define CAPTURE_FILE "raw_capture.dat"

/*
 * Main()
 */
int main(int argc, char **argv)
{
	size_t buffer_size = RX_BUFFER_SIZE;
	size_t list_length = RX_BUFFER_SIZE / MAX_PAYLOAD_LEN;
	RING_BUFFER *pkt_buffer = ring_buffer_create(list_length, buffer_size);

	NET_THREAD_ARGS net_thread_args;
	net_thread_args.pkt_buffer = pkt_buffer;

	HDD_THREAD_ARGS hdd_thread_args;
	hdd_thread_args.pkt_buffer = pkt_buffer;

	// start listening for Ctrl-C
	//signal(SIGINT, cleanup);

	// make stdout unbuffered
	setbuf(stdout, NULL);

	pthread_t net_thread, hdd_thread;
	pthread_create(&net_thread, NULL, net_thread_function, &net_thread_args);
	pthread_create(&hdd_thread, NULL, hdd_thread_function, &hdd_thread_args);

	pthread_join(net_thread, NULL);
	pthread_join(hdd_thread, NULL);

	return 0;
}

/*
 * Read data from the network.
 * Write data to ring buffer.
 */
void *net_thread_function(void *arg)
{
	NET_THREAD_ARGS *args = (NET_THREAD_ARGS *)arg;
	RING_BUFFER *pkt_buffer = args->pkt_buffer;

	RING_ITEM *this_slot = pkt_buffer->write_ptr;
	RING_ITEM *next_slot = NULL;

	socket_t sock = setup_network_listener();
	void *buffer = NULL;
	size_t length = MAX_PAYLOAD_LEN;
	int flags = 0;
	SA_in addr; // packet source's address
	socklen_t addr_len = sizeof(addr);
	ssize_t num_bytes = 0;

	debug_fprintf(stderr, "Entering network thread loop.\n");

	/*
	 * loop forever:
	 *   update relevant local pointers,
	 *   wait for next free buffer slot,
	 *   grab current buffer slot write_mutex,
	 *   read data from network into the slot,
	 *   release the buffer slot read_mutex,
	 *   validate received data based on length,
	 *   advance write pointer to next buffer slot.
	 */
	while (run_net_thread)
	{
		next_slot = this_slot->next;
		buffer = this_slot->data;

		sem_wait(&this_slot->write_mutex);
		num_bytes = recvfrom(sock, buffer, length, flags, (SA *)&addr, &addr_len);
		this_slot->size = num_bytes;
		sem_post(&this_slot->read_mutex);

		if (num_bytes == -1)
		{
			perror("Unable to receive packet.\n");
			exit(1);
		}
		else
		{
			//debug_fprintf(stderr, "[net thread] Received %ld bytes.\n", num_bytes);
		}

		this_slot = next_slot;
	} // end while

	debug_fprintf(stderr, "Exiting network thread loop.\n");

	close(sock);

	run_hdd_thread = 0;

	return NULL;
}

/*
 * Read data from ring buffer.
 * Write data to file on disk.
 */
void *hdd_thread_function(void *arg)
{
	HDD_THREAD_ARGS *args = (HDD_THREAD_ARGS *)arg;
	RING_BUFFER *pkt_buffer = args->pkt_buffer;

	RING_ITEM *this_slot = pkt_buffer->write_ptr;
	RING_ITEM *next_slot = NULL;

	int fd = open_output_file(CAPTURE_FILE);
	ssize_t num_bytes = 0;

	debug_fprintf(stderr, "Entering hard disk thread loop.\n");

	/*
	 * loop forever:
	 *   update relevant local pointers,
	 *   wait for next full buffer slot,
	 *   grab current buffer slot read_mutex,
	 *   write data from the buffer to a file,
	 *   release the buffer slot write_mutex,
	 *   advance read pointer to next buffer slot.
	 */
	while (run_hdd_thread)
	{
		next_slot = this_slot->next;

		sem_wait(&this_slot->read_mutex);
		num_bytes = write(fd, this_slot->data, this_slot->size);
		sem_post(&this_slot->write_mutex);

		if (num_bytes == -1)
		{
			perror("Unable to write packet.\n");
			exit(1);
		}
		else
		{
			//debug_fprintf(stderr, "[hdd thread] Wrote %ld bytes.\n", num_bytes);
		}

		this_slot = next_slot;
	} // end while

	debug_fprintf(stderr, "Exiting hard disk thread loop.\n");

	close(fd);

	return NULL;
}

/*
 * Bind to a port and listen for incoming data.
 */
int setup_network_listener()
{
	int sock = -1;
	struct sockaddr_in my_addr; // server's address information
	int ret = 0;

	// create a new UDP socket descriptor
	sock = socket(PF_INET, SOCK_DGRAM, 0);
	if (sock == -1)
	{
		perror("Unable to set socket descriptor.\n");
		exit(1);
	}

	// initialize local address struct
	my_addr.sin_family = AF_INET; // host byte order
	my_addr.sin_port = htons(LISTEN_PORT); // short, network byte order
	my_addr.sin_addr.s_addr = htonl(INADDR_ANY); // listen on all interfaces
	memset(my_addr.sin_zero, 0, sizeof(my_addr.sin_zero));

	// bind socket to local address
	ret = bind(sock, (SA *)&my_addr, sizeof(my_addr));
	if (ret == -1)
	{
		perror("Unable to bind to socket.\n");
		exit(1);
	}

	// prevent "address already in use" errors
	const int on = 1;
	ret = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (void *)&on, sizeof(on));
	if (ret == -1)
	{
		perror("setsockopt");
		exit(1);
	}

	debug_fprintf(stderr, "Listening on IP address %s on port %i\n", inet_ntoa(my_addr.sin_addr), LISTEN_PORT);

	return sock;
}

int open_output_file(const char *path)
{
	int flags = O_CREAT|O_TRUNC|O_WRONLY;
	mode_t mode = S_IREAD | S_IWUSR;
	int fd = -1;

	fd = open(path, flags, mode);
	if (fd == -1)
	{
		perror("Unable to open capture file.\n");
		exit(1);
	}

	return fd;
}

void cleanup(int signal)
{
	run_net_thread = 0;
}
