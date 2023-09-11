/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdint.h>
#include <stdbool.h>
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

typedef enum bolt_structure_type {
	BST_HELLO = 0x01,
	BST_LOGON = 0x6A,
	BST_LOGOFF = 0x6B,
	BST_GOODBYE = 0x02,
	BST_RESET = 0x0F,
	BST_RUN = 0x10,
	BST_DISCARD = 0x2F,
	BST_PULL = 0x3F,
	BST_BEGIN = 0x11,
	BST_COMMIT = 0x12,
	BST_ROLLBACK = 0x13,
	BST_ROUTE = 0x66,
	BST_SUCCESS = 0x70,
	BST_IGNORED = 0x7E,
	BST_FAILURE = 0x7F,
	BST_RECORD = 0x71,
	BST_NODE = 0x4E,
	BST_RELATIONSHIP = 0x52,
	BST_UNBOUND_RELATIONSHIP = 0x72,
	BST_PATH = 0x50,
	BST_POINT2D = 0x58
} bolt_structure_type;

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
	int8_t data            // tiny int value to write
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

// read value type from buffer
bolt_value_type bolt_read_type
(
	char *data  // buffer to read from
);

// read bool value from buffer
bool bolt_read_bool
(
	char *data  // buffer to read from
);

// read int8 value from buffer
int8_t bolt_read_int8
(
	char *data  // buffer to read from
);

// read int16 value from buffer
int16_t bolt_read_int16
(
	char *data  // buffer to read from
);

// read int32 value from buffer
int32_t bolt_read_int32
(
	char *data  // buffer to read from
);

// read int64 value from buffer
int64_t bolt_read_int64
(
	char *data  // buffer to read from
);

// read float value from buffer
double bolt_read_float
(
	char *data  // buffer to read from
);

// read string size from buffer
uint32_t bolt_read_string_size
(
	char *data  // buffer to read from
);

// read string value from buffer
// notice: the string is not null terminated
char *bolt_read_string
(
	char *data  // buffer to read from
);

// read list size from buffer
uint32_t bolt_read_list_size
(
	char *data
);

// read list item from buffer
char *bolt_read_list_item
(
	char *data,     // buffer to read from
	uint32_t index  // index of item to read
);

// read map size from buffer
uint32_t bolt_read_map_size
(
	char *data  // buffer to read from
);

// read map key from buffer
char *bolt_read_map_key
(
	char *data,     // buffer to read from
	uint32_t index  // index of key to read
);

// read map value from buffer
char *bolt_read_map_value
(
	char *data,     // buffer to read from
	uint32_t index  // index of value to read
);

// read structure type from buffer
bolt_structure_type bolt_read_structure_type
(
	char *data  // buffer to read from
);

// read structure size from buffer
uint32_t bolt_read_structure_size
(
	char *data  // buffer to read from
);

// read structure value from buffer
char *bolt_read_structure_value
(
	char *data,     // buffer to read from
	uint32_t index  // index of value to read
);
