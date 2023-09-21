/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "RG.h"
#include "socket.h"

#define BUFFER_CHUNK_SIZE 4096

typedef struct buffer_index_t buffer_index_t;
typedef struct buffer_t buffer_t;

struct buffer_index_t {
    buffer_t *buf;
    int32_t chunk;
    uint32_t offset;
};

struct buffer_t {
    char **chunks;
    buffer_index_t read;
    buffer_index_t write;
};

void buffer_index
(
    buffer_t *buf,
    buffer_index_t *index,
    uint32_t offset
);

char *buffer_index_read
(
    buffer_index_t *index,
    uint32_t size
);

uint16_t buffer_index_diff
(
    buffer_index_t *a,
    buffer_index_t *b
);

void buffer_new
(
    buffer_t *buf
);

uint8_t buffer_read_uint8
(
    buffer_index_t *buf
);

uint16_t buffer_read_uint16
(
    buffer_index_t *buf
);

uint32_t buffer_read_uint32
(
    buffer_index_t *buf
);

uint64_t buffer_read_uint64
(
    buffer_index_t *buf
);

void buffer_read
(
    buffer_index_t *buf,
    buffer_index_t *dst,
    uint32_t size
);

bool buffer_socket_read
(
    buffer_t *buf,
    socket_t socket
);

bool buffer_socket_write
(
    buffer_index_t *buf,
    socket_t socket
);

void buffer_write_uint8
(
    buffer_index_t *buf,
    uint8_t value
);

void buffer_write_uint16
(
    buffer_index_t *buf,
    uint16_t value
);

void buffer_write_uint32
(
    buffer_index_t *buf,
    uint32_t value
);

void buffer_write_uint64
(
    buffer_index_t *buf,
    uint64_t value
);

void buffer_write
(
    buffer_index_t *buf,
    const char *data,
    uint32_t size
);

void buffer_free
(
    buffer_t *buf
);