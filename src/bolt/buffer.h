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
	buffer_t *buf;    // buffer
	int32_t chunk;    // chunk index
	uint32_t offset;  // offset in chunk
};

struct buffer_t {
	char **chunks;         // array of chunks
	buffer_index_t read;   // read index
	buffer_index_t write;  // write index
};

// set buffer index to offset
void buffer_index_set
(
	buffer_index_t *index,  // index
	buffer_t *buf,          // buffer
	uint32_t offset         // offset
);

// copy the data and increment the index
void buffer_index_read
(
	buffer_index_t *index,  // index
	char *ptr,              // pointer
	uint32_t size           // size
);

// advance the index
void buffer_index_advance
(
	buffer_index_t *index,  // index
	uint32_t n              // # bytes
);

// the length between two indexes
uint64_t buffer_index_diff
(
	buffer_index_t *a,  // index a
	buffer_index_t *b   // index b
);

// the length of the buffer index
uint64_t buffer_index_length
(
	buffer_index_t *index  // index
);

// read until a delimiter
char *buffer_index_read_until
(
	buffer_index_t *index,  // index
	char delimiter          // delimiter
);

// initialize a new buffer
void bolt_buffer_new
(
	buffer_t *buf  // buffer
);

// read a uint8_t from the buffer
uint8_t buffer_read_uint8
(
	buffer_index_t *buf  // buffer
);

// read a uint16_t from the buffer
uint16_t buffer_read_uint16
(
	buffer_index_t *buf  // buffer
);

// read a uint32_t from the buffer
uint32_t buffer_read_uint32
(
	buffer_index_t *buf  // buffer
);

// read a uint64_t from the buffer
uint64_t buffer_read_uint64
(
	buffer_index_t *buf  // buffer
);

// copy data from the buffer to the destination
void buffer_read
(
	buffer_index_t *buf,  // buffer
	buffer_index_t *dst,  // destination
	uint32_t size         // size
);

// read data from the socket to the buffer
bool buffer_socket_read
(
	buffer_t *buf,   // buffer
	socket_t socket  // socket
);

// write data from the buffer to the socket
bool buffer_socket_write
(
	buffer_index_t *from_buf,  // from buffer
	buffer_index_t *to_buf,    // to buffer
	socket_t socket            // socket
);

// write a uint8_t to the buffer
void buffer_write_uint8
(
	buffer_index_t *buf,  // buffer
	uint8_t value         // value
);

// write a uint16_t to the buffer
void buffer_write_uint16
(
	buffer_index_t *buf,  // buffer
	uint16_t value        // value
);

// write a uint32_t to the buffer
void buffer_write_uint32
(
	buffer_index_t *buf,  // buffer
	uint32_t value        // value
);

// write a uint64_t to the buffer
void buffer_write_uint64
(
	buffer_index_t *buf,  // buffer
	uint64_t value        // value
);

// write data to the buffer
void buffer_write
(
	buffer_index_t *buf,  // buffer
	const char *data,     // data
	uint32_t size         // size
);

// apply the mask to the buffer
void buffer_apply_mask
(
	buffer_index_t buf,    // buffer
	uint32_t masking_key,  // masking key
	uint64_t payload_len   // payload length
);

// free the buffer
void bolt_buffer_free
(
	buffer_t *buf  // buffer
);
