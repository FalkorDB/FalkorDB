/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma one

#include "buffer.h"

// check if the request is a websocket handshake
// write the response to the response buffer
bool ws_handshake
(
    buffer_index_t *request,  // the request buffer
    buffer_index_t *response  // the response buffer
);

// read a websocket frame header returning the payload length
uint64_t ws_read_frame
(
    buffer_index_t *buf  // the buffer to read from
);

// write an empty websocket frame header
uint64_t ws_write_empty_header
(
    buffer_index_t *buf  // the buffer to write to
);

// write a websocket frame header
void ws_write_frame_header
(
	buffer_index_t *buf,  // the buffer to write to
	uint64_t n            // the payload length
);
