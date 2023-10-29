/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma one

#include <stdint.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <unistd.h>

typedef int socket_t;

#define socket_accept(socket) accept(socket, NULL, NULL)

#define socket_write(socket, buff, size) send(socket, buff, size, 0)

#define socket_read(socket, buff, size) recv(socket, buff, size, 0)

#define socket_close(socket) close(socket)

socket_t socket_bind
(
	uint16_t port
);

bool socket_set_non_blocking
(
	socket_t socket
);

bool socket_write_all
(
	socket_t socket,
	const char *buff,
	uint32_t size
);
