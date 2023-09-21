/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "socket.h"

#include "RG.h"
#include <string.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <arpa/inet.h>
#include <netinet/in.h>
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

	if(!socket_set_non_blocking(fd)) {
		close(fd);
		return -1;
	}

	if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (void *)&on, sizeof(on)) != 0)
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

bool socket_set_non_blocking
(
	socket_t socket
) {
	int on = 1;
	if (ioctl(socket, FIONBIO, (void *)&on) != 0) {
		return false;
	}

	return true;
}

bool socket_write_all
(
	socket_t socket,
	const char *buff,
	uint32_t size
) {
	uint32_t res = 0;
	while(res < size) {
		int n = socket_write(socket, buff + res, size - res);
		if(n < 0) {
			if(errno == EAGAIN || errno == EWOULDBLOCK) {
				fd_set wfds;
				FD_ZERO(&wfds);
				FD_SET(socket, &wfds);
				select(socket + 1, NULL, &wfds, NULL, NULL);
				continue;
			} else {
				return false;
			}
		}
		res += n;
	}
	return true;
}
