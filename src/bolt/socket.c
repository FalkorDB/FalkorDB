/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "socket.h"

#include "RG.h"
#include <string.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/ioctl.h>
#include <sys/socket.h>

#define CLIENT_QUEUE_LEN 32

socket_t socket_bind
(
	uint16_t port
) {
	int on = 1;
	socket_t fd = socket(AF_INET6, SOCK_STREAM, IPPROTO_TCP);
	if (fd == -1) {
		return -1;
	}

	if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (void *)&on, sizeof(on)) != 0)
	{
		close(fd);
		return -1;
	}

	if (ioctl(fd, FIONBIO, (void *)&on) != 0)
	{
		close(fd);
		return -1;
	}

	struct sockaddr_in6 serveraddr, cli;
	memset(&serveraddr, 0, sizeof(serveraddr));

	serveraddr.sin6_family = AF_INET6;
	serveraddr.sin6_addr = in6addr_any;
	serveraddr.sin6_port = htons(port);

	if (bind(fd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) != 0) {
		close(fd);
		return -1;
	}

	if (listen(fd, CLIENT_QUEUE_LEN) != 0) {
		close(fd);
		return -1;
	}

	return fd;
}