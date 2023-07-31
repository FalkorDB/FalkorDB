#pragma one

#include <stdint.h>
#include <stdbool.h>
#include <sys/socket.h>
#include <unistd.h>

typedef int socket_t;

socket_t socket_bind
(
    uint16_t port
);

#define socket_accept(socket) accept(socket, NULL, NULL)

#define socket_read(socket, buff, size) recv(socket, buff, size, 0)

#define socket_write(socket, buff, size) send(socket, buff, size, 0)

#define socket_close(socket) close(socket)
