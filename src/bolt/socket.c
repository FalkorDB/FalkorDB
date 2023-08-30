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
#include <sys/select.h>

socket_t socket_bind
(
    uint16_t port
) {
    int on = 1;
    socket_t fd = socket(AF_INET, SOCK_STREAM, 0);
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

    struct sockaddr_in serveraddr, cli;
    memset(&serveraddr, 0, sizeof(serveraddr));

    serveraddr.sin_family = AF_INET;
    serveraddr.sin_addr.s_addr = htonl(INADDR_ANY);
    serveraddr.sin_port = htons(port);

    if (bind(fd, (struct sockaddr *)&serveraddr, sizeof(serveraddr)) != 0) {
        close(fd);
        return -1;
    }

    if (listen(fd, 32) != 0) {
        close(fd);
        return -1;
    }

    return fd;
}

bool socket_read
(
    socket_t socket,
    char* buff,
    size_t size
) {
    int nread = 0;

    while(nread < size) {
        int n = recv(socket, buff + nread, size - nread, 0);
        if(n <= 0) {
            return false;
        }
        nread += n;
    }

    ASSERT(nread == size);

    return true;
}