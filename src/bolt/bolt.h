/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "RG.h"
#include "buffer.h"
#include "bolt_client.h"

typedef enum bolt_value_type {
	BVT_NULL,
	BVT_BOOL,
	BVT_INT8,
	BVT_INT16,
	BVT_INT32,
	BVT_INT64,
	BVT_FLOAT,
	BVT_STRING,
	BVT_BYTES,
	BVT_LIST,
	BVT_MAP,
	BVT_STRUCTURE
} bolt_value_type;

//------------------------------------------------------------------------------
// Write functions
//------------------------------------------------------------------------------

// write null to client response buffer
void bolt_reply_null
(
	bolt_client_t *client  // client to write to
);

// write bool value to client response buffer
void bolt_reply_bool
(
	bolt_client_t *client,  // client to write to
	bool data               // bool value to write
);

// write tiny int value to client response buffer
// tiny int: -16 to 127
void bolt_reply_tiny_int
(
	bolt_client_t *client,  // client to write to
	int8_t data             // tiny int value to write
);

// write int8 value to client response buffer
void bolt_reply_int8
(
	bolt_client_t *client,  // client to write to
	int8_t data             // int8 value to write
);

// write int16 value to client response buffer
void bolt_reply_int16
(
	bolt_client_t *client,  // client to write to
	int16_t data            // int16 value to write
);

// write int32 value to client response buffer
void bolt_reply_int32
(
	bolt_client_t *client,  // client to write to
	int32_t data            // int32 value to write
);

// write int64 value to client response buffer
void bolt_reply_int64
(
	bolt_client_t *client,  // client to write to
	int64_t data            // int64 value to write
);

// write int value to client response buffer
// using the minimal representation
// if the minimal representation is known use it for better performance
void bolt_reply_int
(
	bolt_client_t *client,  // client to write to
	int64_t data            // int value to write
);

// write float value to client response buffer
void bolt_reply_float
(
	bolt_client_t *client,  // client to write to
	double data             // float value to write
);

// write string value to client response buffer
// using the minimal representation
void bolt_reply_string
(
	bolt_client_t *client,  // client to write to
	const char *data,       // string value to write
	uint32_t size           // string size
);

// write list header to client response buffer
// expected 'size' number of items to follow
void bolt_reply_list
(
	bolt_client_t *client,  // client to write to
	uint32_t size           // number of items to follow
);

// write map header to client response buffer
// expected 'size' number of key-value pairs to follow
// key should be string
// value can be any type
void bolt_reply_map
(
	bolt_client_t *client,  // client to write to
	uint32_t size           // number of key-value pairs to follow
);

// write structure header to client response buffer
// expected 'size' number of items to follow
void bolt_reply_structure
(
	bolt_client_t *client,     // client to write to
	bolt_structure_type type,  // structure type
	uint32_t size              // number of items to follow
);

//------------------------------------------------------------------------------
// Read functions
//------------------------------------------------------------------------------

// read value type from buffer
bolt_value_type bolt_read_type
(
	buffer_index_t data  // buffer to read from
);

// read null value from buffer
void bolt_read_null
(
	buffer_index_t *data  // buffer to read from
);

// read bool value from buffer
bool bolt_read_bool
(
	buffer_index_t *data  // buffer to read from
);

// read int8 value from buffer
int8_t bolt_read_int8
(
	buffer_index_t *data  // buffer to read from
);

// read int16 value from buffer
int16_t bolt_read_int16
(
	buffer_index_t *data  // buffer to read from
);

// read int32 value from buffer
int32_t bolt_read_int32
(
	buffer_index_t *data  // buffer to read from
);

// read int64 value from buffer
int64_t bolt_read_int64
(
	buffer_index_t *data  // buffer to read from
);

// read float value from buffer
double bolt_read_float
(
	buffer_index_t *data  // buffer to read from
);

// read string size from buffer
void bolt_read_string_size
(
	buffer_index_t *data,  // buffer to read from
	uint32_t *size         // string size
);

// read string value from buffer
// notice: the string is not null terminated
void bolt_read_string
(
	buffer_index_t *data,  // buffer to read from
	char *str              // string buffer
);

// read list size from buffer
uint32_t bolt_read_list_size
(
	buffer_index_t *data
);

// read map size from buffer
uint32_t bolt_read_map_size
(
	buffer_index_t *data  // buffer to read from
);

// read structure type from buffer
bolt_structure_type bolt_read_structure_type
(
	buffer_index_t *data  // buffer to read from
);

// read structure size from buffer
uint32_t bolt_read_structure_size
(
	buffer_index_t *data  // buffer to read from
);
