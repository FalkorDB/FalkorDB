/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "buffer.h"
#include "../util/arr.h"

// pointer to buffer's read position
#define BUFFER_READ_POSITION(buff) \
	buff->chunks[buff->read.chunk] + buff->read.offset

// pointer to buffer's write position
#define BUFFER_WRITE_POSITION(buff) \
	buff->chunks[buff->write.chunk] + buff->write.offset

// compute how much space is available in the buffer's current write chunk
#define BUFFER_CHUNK_AVAILABLE_SIZE(buff) \
	(BUFFER_CHUNK_SIZE - buff->write.offset)

#define BUFFER_ADVANCE_WRITE(buff, n) \
	buff->write.offset += n;

// reset buffer's read and write pointers
static void _buffer_reset_pointers
(
	buffer_t *buf  // buffer to reset
) {
	ASSERT(buf != NULL);

	// set read & write chunk and offset to 0
	buf->read.chunk   = 0;
	buf->read.offset  = 0;
	buf->write.chunk  = 0;
	buf->write.offset = 0;
}

// set buffer index to offset
void buffer_index_set
(
	buffer_index_t *index,  // index
	buffer_t *buf,          // buffer
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
bool buffer_index_advance
(
	buffer_index_t *index,
	uint32_t n
) {
	ASSERT(index != NULL);
	// check if there is enough data to read
	if(buffer_index_length(index) < n) {
		return false;
	}

	index->offset += n;
	if(index->offset > BUFFER_CHUNK_SIZE) {
		index->chunk += index->offset / BUFFER_CHUNK_SIZE;
		index->offset %= BUFFER_CHUNK_SIZE;
	}
	return true;
}

// read n bytes from buffer
bool buffer_read_n
(
	buffer_index_t *index,  // buffer to read from
	char *ptr,              // read data into this pointer
	uint32_t size           // number of bytes to read
) {
	ASSERT(ptr   != NULL);
	ASSERT(index != NULL);

	// return false if there is not enough data to read
	if(size > buffer_index_length(index)) {
		return false;
	}

	buffer_index_t start = *index;
	char *from = start.buf->chunks[start.chunk] + start.offset;
	buffer_index_advance(index, size);

	while(start.chunk < index->chunk) {
		memcpy(ptr, from, BUFFER_CHUNK_SIZE - start.offset);
		ptr += BUFFER_CHUNK_SIZE - start.offset;
		start.chunk++;
		start.offset = 0;
		from = index->buf->chunks[start.chunk];
	}

	memcpy(ptr, from, index->offset - start.offset);
	return true;
}

// the length between two indexes
uint64_t buffer_index_diff
(
	buffer_index_t *a,  // index a
	buffer_index_t *b   // index b
) {
	ASSERT(a != NULL);
	ASSERT(b != NULL);
	ASSERT(a->buf == b->buf);

	uint64_t diff = (a->chunk - b->chunk) * BUFFER_CHUNK_SIZE + (a->offset - b->offset);
	ASSERT(diff >= 0);
	return diff;
}

// the length of the buffer index
uint64_t buffer_index_length
(
	buffer_index_t *index  // index
) {
	ASSERT(index != NULL);

	return buffer_index_diff(&index->buf->write, index);
}

// read until a delimiter
bool buffer_index_read_until
(
	buffer_index_t *index,  // index
	char delimiter,         // delimiter
	char **ptr              // pointer
) {
	ASSERT(index != NULL);
	ASSERT(ptr != NULL && *ptr == NULL);

	char *res = NULL;
	char *from = index->buf->chunks[index->chunk] + index->offset;
	uint32_t size = 0;
	while(index->chunk < index->buf->write.chunk) {
		char *p = memchr(from, delimiter, BUFFER_CHUNK_SIZE - index->offset);
		if(p != NULL) {
			size += p - from;
			res = rm_malloc(size + 1);
			if(!buffer_read_n(index, res, size)) {
				rm_free(res);
				return false;
			}
			res[size] = '\0';
			*ptr = res;
			return true;
		}
		size += BUFFER_CHUNK_SIZE - index->offset;
		index->chunk++;
		index->offset = 0;
		from = index->buf->chunks[index->chunk];
	}
	char *p = memchr(from, delimiter, index->buf->write.offset - index->offset);
	if(p != NULL) {
		size += p - from;
		res = rm_malloc(size + 1);
		if(!buffer_read_n(index, res, size)) {
			rm_free(res);
			return false;
		}
		res[size] = '\0';
		*ptr = res;
		return true;
	}
	return false;
}

// initialize a new buffer
void buffer_new
(
	buffer_t *buf  // buffer
) {
	ASSERT(buf != NULL);

	buf->chunks = array_new(char *, 0);
	array_append(buf->chunks, rm_malloc(BUFFER_CHUNK_SIZE));
	buffer_index_set(&buf->read, buf, 0);
	buffer_index_set(&buf->write, buf, 0);
}

// declare buffer read function
#define buffer_read_t(type)                                      \
bool buffer_read_##type                                          \
(                                                                \
	buffer_index_t *buf,  /* buffer */                           \
	type *value           /* value */                            \
) {                                                              \
	ASSERT(buf != NULL);                                         \
	ASSERT(value != NULL);                                       \
	return buffer_read_n(buf, (char *)value, sizeof(type));      \
}                                                                \

// declare all buffer read functions
buffer_read_t(int8_t)
buffer_read_t(uint8_t)
buffer_read_t(int16_t)
buffer_read_t(uint16_t)
buffer_read_t(int32_t)
buffer_read_t(uint32_t)
buffer_read_t(int64_t)
buffer_read_t(uint64_t)

// copy data from the buffer to the destination
bool buffer_copy
(
	buffer_index_t *buf,  // buffer
	buffer_index_t *dst,  // destination
	uint32_t size         // size
) {
	ASSERT(buf != NULL);
	ASSERT(dst != NULL);
	if(buffer_index_length(buf) < size) {
		return false;
	}

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
			buf->offset += size;
			dst->offset += size;
			return true;
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
	return true;
}

// read data from the socket to the buffer
bool buffer_socket_read
(
	buffer_t *buf,   // buffer
	socket_t socket  // socket
) {
	ASSERT(buf != NULL);
	ASSERT(socket > 0);

	// in case the buffer is empty, reset the read and write pointers
	if(BUFFER_READ_POSITION(buf) == BUFFER_WRITE_POSITION(buf)) {
		_buffer_reset_pointers(buf);
	}

	char *ptr = BUFFER_WRITE_POSITION(buf);
	int nread = socket_read(socket, ptr, BUFFER_CHUNK_AVAILABLE_SIZE(buf));

	if(nread < 0 || (nread == 0 && buf->write.offset < BUFFER_CHUNK_SIZE)) {
		// failed to read from socket
		return false;
	}

	// update write position
	BUFFER_ADVANCE_WRITE(buf, nread);

	// TODO: what happens if we've read exactly BUFFER_CHUNK_SIZE bytes?
	// and there's no more data to read from the socket?
	// will socket_read block forever?
	while(buf->write.offset == BUFFER_CHUNK_SIZE) {
		// update write position
		buf->write.offset = 0;
		buf->write.chunk++;
		char *chunk;
		if(array_len(buf->chunks) == buf->write.chunk) {
			// create a new chunk
			chunk = rm_malloc(BUFFER_CHUNK_SIZE);
			
			// add chunk to the buffer
			array_append(buf->chunks, chunk);
		} else {
			chunk = buf->chunks[buf->write.chunk];
		}

		// read from socket into the new chunk
		nread = socket_read(socket, chunk, BUFFER_CHUNK_SIZE);
		if(nread < 0) {
			return false;
		}

		// update write position
		BUFFER_ADVANCE_WRITE(buf, nread);
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

#define buffer_write_t(type)                                      \
void buffer_write_##type                                          \
(                                                                 \
	buffer_index_t *buf,  /* buffer */                            \
	type value            /* value */                             \
) {                                                               \
	ASSERT(buf != NULL);                                          \
	buffer_write_n(buf, (char *)&value, sizeof(type));            \
}                                                                 \

// declare all buffer write functions
buffer_write_t(int8_t)
buffer_write_t(uint8_t)
buffer_write_t(int16_t)
buffer_write_t(uint16_t)
buffer_write_t(int32_t)
buffer_write_t(uint32_t)
buffer_write_t(int64_t)
buffer_write_t(uint64_t)
buffer_write_t(double)

// write data to the buffer
void buffer_write_n
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
		buffer_index_advance(buf, n);
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
	buf->offset += size;
}

// apply the mask to a single chunk in the buffer
static void buffer_apply_mask_single
(
	buffer_index_t buf,    // buffer
	uint32_t masking_key,  // masking key
	uint32_t payload_len,  // payload length
	int *offset            // in/out offset
) {
	char *payload = buf.buf->chunks[buf.chunk] + buf.offset;
	int local_offset = *offset;
	uint32_t double_mask[4] = {masking_key, masking_key, masking_key, masking_key};
	uint64_t offset_mask = *(uint64_t *)((char *)double_mask + local_offset);
	int i = 0;
	for(; i + 8 <= payload_len; i+=8) {
		*(uint64_t *)(payload + i) ^= offset_mask;
	}
	for(; i < payload_len; i++) {
		payload[i] ^= ((char*)double_mask)[local_offset++];
	}
	*offset = local_offset % 4;
}

// apply the mask to the buffer
void buffer_apply_mask
(
	buffer_index_t buf,    // buffer
	uint32_t masking_key,  // masking key
	uint64_t payload_len   // payload length
) {
	ASSERT(buffer_index_length(&buf) >= payload_len);

	buffer_index_t end = buf;
	buffer_index_advance(&end, payload_len);
	int offset = 0;
	while(buf.chunk < end.chunk) {
		buffer_apply_mask_single(buf, masking_key, BUFFER_CHUNK_SIZE - buf.offset, &offset);
		buf.chunk++;
		buf.offset = 0;
	}
	buffer_apply_mask_single(buf, masking_key, end.offset - buf.offset, &offset);
}

// free the buffer
void buffer_free
(
	buffer_t *buf  // buffer
) {
	ASSERT(buf != NULL);

	array_free_cb(buf->chunks, rm_free);
}
