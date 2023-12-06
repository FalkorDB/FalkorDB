/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "buffer.h"
#include "../util/arr.h"

// set buffer index to offset
void buffer_index
(
	buffer_t *buf,          // buffer
	buffer_index_t *index,  // index
	uint32_t offset         // offset
) {
	ASSERT(buf != NULL);
	ASSERT(index != NULL);
	ASSERT(offset < BUFFER_CHUNK_SIZE * array_len(buf->chunks));

	index->buf = buf;
	index->chunk = offset / BUFFER_CHUNK_SIZE;
	index->offset = offset % BUFFER_CHUNK_SIZE;
}

// add offset to index
void buffer_index_add
(
	buffer_index_t *index,
	uint32_t offset
) {
	ASSERT(index != NULL);

	index->offset += offset;
	if(index->offset > BUFFER_CHUNK_SIZE) {
		index->chunk += index->offset / BUFFER_CHUNK_SIZE;
		index->offset %= BUFFER_CHUNK_SIZE;
	}
}

// copy the data and increment the index
void buffer_index_read
(
	buffer_index_t *index,  // index
	char *ptr,              // pointer
	uint32_t size           // size
) {
	ASSERT(index != NULL);
	ASSERT(buffer_index_diff(&index->buf->write, index) >= size);

	buffer_index_t start = *index;
	char *from = index->buf->chunks[index->chunk] + index->offset;
	buffer_index_add(index, size);
	if(ptr != NULL) {
		while (start.chunk < index->chunk) {
			memcpy(ptr, from, BUFFER_CHUNK_SIZE - start.offset);
			ptr += BUFFER_CHUNK_SIZE - start.offset;
			start.chunk++;
			start.offset = 0;
			from = index->buf->chunks[start.chunk];
		}
		memcpy(ptr, from, index->offset - start.offset);
	}
}

// the length between two indexes
uint16_t buffer_index_diff
(
	buffer_index_t *a,  // index a
	buffer_index_t *b   // index b
) {
	ASSERT(a != NULL);
	ASSERT(b != NULL);

	uint16_t diff = (a->chunk - b->chunk) * BUFFER_CHUNK_SIZE + (a->offset - b->offset);
	ASSERT(diff >= 0);
	return diff;
}

// initialize a new buffer
void buffer_new
(
	buffer_t *buf  // buffer
) {
	ASSERT(buf != NULL);

	buf->chunks = array_new(char *, 0);
	array_append(buf->chunks, rm_malloc(BUFFER_CHUNK_SIZE));
	buffer_index(buf, &buf->read, 0);
	buffer_index(buf, &buf->write, 0);
}

// read a uint8_t from the buffer
uint8_t buffer_read_uint8
(
	buffer_index_t *buf  // buffer
) {
	ASSERT(buf != NULL);

	uint8_t res;
	buffer_index_read(buf, (char *)&res, 1);
	return res;
}

// read a uint16_t from the buffer
uint16_t buffer_read_uint16
(
	buffer_index_t *buf  // buffer
) {
	ASSERT(buf != NULL);

	uint16_t res;
	buffer_index_read(buf, (char *)&res, 2);
	return res;
}

// read a uint32_t from the buffer
uint32_t buffer_read_uint32
(
	buffer_index_t *buf  // buffer
) {
	ASSERT(buf != NULL);

	uint32_t res;
	buffer_index_read(buf, (char *)&res, 4);
	return res;
}

// read a uint64_t from the buffer
uint64_t buffer_read_uint64
(
	buffer_index_t *buf  // buffer
) {
	ASSERT(buf != NULL);

	uint64_t res;
	buffer_index_read(buf, (char *)&res, 8);
	return res;
}

// copy data from the buffer to the destination
void buffer_read
(
	buffer_index_t *buf,  // buffer
	buffer_index_t *dst,  // destination
	uint32_t size         // size
) {
	ASSERT(buf != NULL);
	ASSERT(dst != NULL);
	ASSERT(buffer_index_diff(&buf->buf->write, buf) >= size);

	char *src_ptr;
	char *dst_ptr;
	uint32_t dst_available_size;
	uint32_t src_available_size;
	while(size > 0) {
		src_ptr = buf->buf->chunks[buf->chunk] + buf->offset;
		dst_ptr = dst->buf->chunks[dst->chunk] + dst->offset;
		dst_available_size = BUFFER_CHUNK_SIZE - dst->offset;
		src_available_size = BUFFER_CHUNK_SIZE - buf->offset;
		if(size < src_available_size && size < dst_available_size) {
			memcpy(dst_ptr, src_ptr, size);
			buffer_index_add(buf, size);
			buffer_index_add(dst, size);
			return;
		}
		
		if(src_available_size < dst_available_size) {
			memcpy(dst_ptr, src_ptr, src_available_size);
			size -= src_available_size;
			buf->offset = 0;
			buf->chunk++;
			dst->offset += src_available_size;
		} else {
			memcpy(dst_ptr, src_ptr, dst_available_size);
			size -= dst_available_size;
			dst->offset = 0;
			dst->chunk++;
			if(array_len(dst->buf->chunks) == dst->chunk) {
				array_append(dst->buf->chunks, rm_malloc(BUFFER_CHUNK_SIZE));
			}
			buf->offset += dst_available_size;
		}
	}
}

// read data from the socket to the buffer
bool buffer_socket_read
(
	buffer_t *buf,   // buffer
	socket_t socket  // socket
) {
	ASSERT(buf != NULL);
	ASSERT(socket > 0);

	char *ptr = buf->chunks[buf->write.chunk] + buf->write.offset;
	int nread = socket_read(socket, ptr, BUFFER_CHUNK_SIZE - buf->write.offset);
	if(nread < 0 || nread == 0 && buf->write.offset < BUFFER_CHUNK_SIZE) {
		return false;
	}

	buffer_index_add(&buf->write, nread);
	while(buf->write.offset == BUFFER_CHUNK_SIZE) {
		buf->write.offset = 0;
		buf->write.chunk++;
		array_append(buf->chunks, rm_malloc(BUFFER_CHUNK_SIZE));
		nread = socket_read(socket, buf->chunks[buf->write.chunk], BUFFER_CHUNK_SIZE);
		if(nread < 0) {
			return false;
		}
		buffer_index_add(&buf->write, nread);
	}
	return true;
}

// write data from the buffer to the socket
bool buffer_socket_write
(
	buffer_index_t *from_buf,  // from buffer
	buffer_index_t *to_buf,    // to buffer
	socket_t socket            // socket
) {
	ASSERT(from_buf != NULL);
	ASSERT(to_buf != NULL);
	ASSERT(from_buf->buf == to_buf->buf);
	ASSERT(buffer_index_diff(to_buf, from_buf) > 0);
	ASSERT(socket > 0);

	if(from_buf->chunk == to_buf->chunk) {
		return socket_write_all(socket, from_buf->buf->chunks[from_buf->chunk] + from_buf->offset, to_buf->offset - from_buf->offset);
	}
	if(!socket_write_all(socket, from_buf->buf->chunks[from_buf->chunk] + from_buf->offset, BUFFER_CHUNK_SIZE - from_buf->offset)) {
		return false;
	}
	for(int32_t i = from_buf->chunk + 1; i < to_buf->chunk; i++) {
		if(!socket_write_all(socket, from_buf->buf->chunks[i], BUFFER_CHUNK_SIZE)) {
			return false;
		}
	}
	return socket_write_all(socket, to_buf->buf->chunks[to_buf->chunk], to_buf->offset);
}

// write a uint8_t to the buffer
void buffer_write_uint8
(
	buffer_index_t *buf,  // buffer
	uint8_t value         // value
) {
	ASSERT(buf != NULL);

	buffer_write(buf, (char *)&value, 1);
}

// write a uint16_t to the buffer
void buffer_write_uint16
(
	buffer_index_t *buf,  // buffer
	uint16_t value        // value
) {
	ASSERT(buf != NULL);

	buffer_write(buf, (char *)&value, 2);
}

// write a uint32_t to the buffer
void buffer_write_uint32
(
	buffer_index_t *buf,  // buffer
	uint32_t value        // value
) {
	ASSERT(buf != NULL);

	buffer_write(buf, (char *)&value, 4);
}

// write a uint64_t to the buffer
void buffer_write_uint64
(
	buffer_index_t *buf,  // buffer
	uint64_t value        // value
) {
	ASSERT(buf != NULL);

	buffer_write(buf, (char *)&value, 8);
}

// write data to the buffer
void buffer_write
(
	buffer_index_t *buf,  // buffer
	const char *data,     // data
	uint32_t size         // size
) {
	ASSERT(buf != NULL);
	ASSERT(data != NULL);

	while(buf->offset + size > BUFFER_CHUNK_SIZE) {
		uint32_t n = BUFFER_CHUNK_SIZE - buf->offset;
		memcpy(buf->buf->chunks[buf->chunk] + buf->offset, data, n);
		buffer_index_add(buf, n);
		data += n;
		size -= n;
		buf->chunk++;
		buf->offset = 0;
		if(array_len(buf->buf->chunks) == buf->chunk) {
			array_append(buf->buf->chunks, rm_malloc(BUFFER_CHUNK_SIZE));
		}
	}
	char *ptr = buf->buf->chunks[buf->chunk] + buf->offset;
	memcpy(ptr, data, size);
	buffer_index_add(buf, size);
}

// free the buffer
void buffer_free
(
	buffer_t *buf  // buffer
) {
	ASSERT(buf != NULL);

	array_free_cb(buf->chunks, rm_free);
}