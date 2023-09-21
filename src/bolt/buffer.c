/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "buffer.h"
#include "../util/arr.h"

void buffer_index
(
    buffer_t *buf,
    buffer_index_t *index,
    uint32_t offset
) {
    index->buf = buf;
    index->chunk = offset / BUFFER_CHUNK_SIZE;
    index->offset = offset % BUFFER_CHUNK_SIZE;
}

void buffer_index_add
(
    buffer_index_t *index,
    uint32_t offset
) {
    index->offset += offset;
    if(index->offset > BUFFER_CHUNK_SIZE) {
        index->chunk += index->offset / BUFFER_CHUNK_SIZE;
        index->offset %= BUFFER_CHUNK_SIZE;
    }
}

char *buffer_index_read
(
    buffer_index_t *index,
    uint32_t size
) {
    char *ptr = index->buf->chunks[index->chunk] + index->offset;
    buffer_index_add(index, size);
    return ptr;
}

void buffer_write
(
    buffer_index_t *buf,
    const char *data,
    uint32_t size
) {
    while(buf->offset + size > BUFFER_CHUNK_SIZE) {
        uint32_t first_chunk_size = BUFFER_CHUNK_SIZE - buf->offset;
        memcpy(buf->buf->chunks[buf->chunk] + buf->offset, data, first_chunk_size);
        buffer_index_add(buf, first_chunk_size);
        data += first_chunk_size;
        size -= first_chunk_size;
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

uint16_t buffer_index_diff
(
    buffer_index_t *a,
    buffer_index_t *b
) {
    return (a->chunk - b->chunk) * BUFFER_CHUNK_SIZE + (a->offset - b->offset);
}

void buffer_new
(
    buffer_t *buf
) {
    buf->chunks = array_new(char *, 0);
    array_append(buf->chunks, rm_malloc(BUFFER_CHUNK_SIZE));
    buffer_index(buf, &buf->read, 0);
    buffer_index(buf, &buf->write, 0);
}

uint8_t buffer_read_uint8
(
    buffer_index_t *buf
) {
    return *buffer_index_read(buf, 1);
}

uint16_t buffer_read_uint16
(
    buffer_index_t *buf
) {
    return *(uint16_t *)buffer_index_read(buf, 2);
}

uint32_t buffer_read_uint32
(
    buffer_index_t *buf
) {
    return *(uint32_t *)buffer_index_read(buf, 4);
}

uint64_t buffer_read_uint64
(
    buffer_index_t *buf
) {
    return *(uint64_t *)buffer_index_read(buf, 8);
}

void buffer_read
(
    buffer_index_t *buf,
    buffer_index_t *dst,
    uint32_t size
) {
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

bool buffer_socket_read
(
    buffer_t *buf,
    socket_t socket
) {
    char *ptr = buf->chunks[buf->write.chunk] + buf->write.offset;
	int nread = socket_read(socket, ptr, BUFFER_CHUNK_SIZE - buf->write.offset);
    if(nread < 0) {
        return false;
    }
    if(nread == 0 && buf->write.offset < BUFFER_CHUNK_SIZE) {
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

bool buffer_socket_write
(
    buffer_index_t *buf,
    socket_t socket
) {
    for(int32_t i = 0; i < buf->chunk; i++) {
        if(!socket_write_all(socket, buf->buf->chunks[i], BUFFER_CHUNK_SIZE)) {
            return false;
        }
    }
    return socket_write_all(socket, buf->buf->chunks[buf->chunk], buf->offset);
}

void buffer_write_uint8
(
    buffer_index_t *buf,
    uint8_t value
) {
    buffer_write(buf, (char *)&value, 1);
}

void buffer_write_uint16
(
    buffer_index_t *buf,
    uint16_t value
) {
    buffer_write(buf, (char *)&value, 2);
}

void buffer_write_uint32
(
    buffer_index_t *buf,
    uint32_t value
) {
    buffer_write(buf, (char *)&value, 4);
}

void buffer_write_uint64
(
    buffer_index_t *buf,
    uint64_t value
) {
    buffer_write(buf, (char *)&value, 8);
}

void buffer_free
(
    buffer_t *buf
) {
    array_free_cb(buf->chunks, rm_free);
}