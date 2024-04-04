/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "bolt.h"
#include "string.h"
#include "endian.h"

#define NULL_MARKER 0xC0
#define TRUE_MARKER 0xC3
#define FALSE_MARKER 0xC2
#define TINY_INT8_MIN 0xF0
#define TINY_INT8_MAX 0x7F
#define INT8_MARKER 0xC8
#define INT16_MARKER 0xC9
#define INT32_MARKER 0xCA
#define INT64_MARKER 0xCB
#define FLOAT_MARKER 0xC1
#define TINY_STRING_BASE_MARKER 0x80
#define STRING8_MARKER 0xD0
#define STRING16_MARKER 0xD1
#define STRING32_MARKER 0xD2
#define TINY_LIST_BASE_MARKER 0x90
#define LIST8_MARKER 0xD4
#define LIST16_MARKER 0xD5
#define LIST32_MARKER 0xD6
#define BYTES8_MARKER 0xCC
#define BYTES16_MARKER 0xCD
#define BYTES32_MARKER 0xCE
#define TINY_MAP_BASE_MARKER 0xA0
#define MAP8_MARKER 0xD8
#define MAP16_MARKER 0xD9
#define MAP32_MARKER 0xDA
#define STRUCTURE_BASE_MARKER 0xB0

#define TINY_SIZE 16
#define TINY_MARKER_CHECK(base, marker) (marker >= base && marker <= base + 0x0F)

//------------------------------------------------------------------------------
// Write functions
//------------------------------------------------------------------------------

// write null to client response buffer
void bolt_reply_null
(
	bolt_client_t *client  // client to write to
) {
	ASSERT(client != NULL);

	buffer_write_uint8(&client->write_buf.write, NULL_MARKER);
}

// write bool value to client response buffer
void bolt_reply_bool
(
	bolt_client_t *client,  // client to write to
	bool data               // bool value to write
) {
	ASSERT(client != NULL);

	buffer_write_uint8(&client->write_buf.write, data ? TRUE_MARKER : FALSE_MARKER);
}

// write tiny int value to client response buffer
// tiny int: -16 to 127
void bolt_reply_tiny_int
(
	bolt_client_t *client,  // client to write to
	int8_t data             // tiny int value to write
) {
	ASSERT(client != NULL);
	ASSERT(data >= TINY_INT8_MIN && data <= TINY_INT8_MAX);

	buffer_write_uint8(&client->write_buf.write, data);
}

// write int8 value to client response buffer
void bolt_reply_int8
(
	bolt_client_t *client,  // client to write to
	int8_t data             // int8 value to write
) {
	ASSERT(client != NULL);

	int8_t values[2] = {INT8_MARKER, data};
    buffer_write(&client->write_buf.write, values, 2);
}

// write int16 value to client response buffer
void bolt_reply_int16
(
	bolt_client_t *client,  // client to write to
	int16_t data            // int16 value to write
) {
	ASSERT(client != NULL);

	int8_t values[3] = {INT16_MARKER, 0, 0};
	*(int16_t *)(values + 1) = htons(data);
	buffer_write(&client->write_buf.write, values, 3);
}

// write int32 value to client response buffer
void bolt_reply_int32
(
	bolt_client_t *client,  // client to write to
	int32_t data            // int32 value to write
) {
	ASSERT(client != NULL);

	int8_t values[5] = {INT32_MARKER, 0, 0, 0, 0};
	*(int32_t *)(values + 1) = htonl(data);
	buffer_write(&client->write_buf.write, values, 5);
}

// write int64 value to client response buffer
void bolt_reply_int64
(
	bolt_client_t *client, // client to write to
	int64_t data           // int64 value to write
) {
	ASSERT(client != NULL);

	int8_t values[9] = {INT64_MARKER, 0, 0, 0, 0, 0, 0, 0, 0};
	*(int64_t *)(values + 1) = htonll(data);
	buffer_write(&client->write_buf.write, values, 9);
}

// write int value to client response buffer
// using the minimal representation
// if the minimal representation is known use it for better performance
void bolt_reply_int
(
	bolt_client_t *client,  // client to write to
	int64_t data            // int value to write
) {
	ASSERT(client != NULL);

	if(data >= TINY_INT8_MIN && data <= TINY_INT8_MAX) {
		bolt_reply_tiny_int(client, data);
	} else if(INT8_MIN <= data && data <= INT8_MAX) {
		bolt_reply_int8(client, data);
	} else if(INT16_MIN <= data && data <= INT16_MAX) {
		bolt_reply_int16(client, data);
	} else if(INT32_MIN <= data && data <= INT32_MAX) {
		bolt_reply_int32(client, data);
	} else {
		bolt_reply_int64(client, data);
	}
}

// write float value to client response buffer
void bolt_reply_float
(
	bolt_client_t *client,  // client to write to
	double data             // float value to write
) {
	ASSERT(client != NULL);

	buffer_write_uint8(&client->write_buf.write, FLOAT_MARKER);
	char *buf = (char *)&data;
	for (int i = sizeof(double); i > 0; i--) {
		buffer_write_uint8(&client->write_buf.write, buf[i - 1]);
	}
}

// write string value to client response buffer
// using the minimal representation
void bolt_reply_string
(
	bolt_client_t *client,  // client to write to
	const char *data,       // string value to write
	uint32_t size           // string size
) {
	ASSERT(client != NULL);
	ASSERT(data != NULL);

	if (size < TINY_SIZE) {
		buffer_write_uint8(&client->write_buf.write, TINY_STRING_BASE_MARKER + size);
	} else if (size <= UINT8_MAX) {
		int8_t values[2] = {STRING8_MARKER, size};
    	buffer_write(&client->write_buf.write, values, 2);
	} else if (size <= UINT16_MAX) {
		int8_t values[3] = {STRING16_MARKER, 0, 0};
		*(uint16_t *)(values + 1) = htons(size);
		buffer_write(&client->write_buf.write, values, 3);
	} else {
		int8_t values[5] = {STRING32_MARKER, 0, 0, 0, 0};
		*(uint32_t *)(values + 1) = htonl(size);
		buffer_write(&client->write_buf.write, values, 5);
	}
	buffer_write(&client->write_buf.write, data, size);
}

// write list header to client response buffer
// expected 'size' number of items to follow
void bolt_reply_list
(
	bolt_client_t *client,
	uint32_t size
) {
	ASSERT(client != NULL);

	if (size < TINY_SIZE) {
		buffer_write_uint8(&client->write_buf.write, TINY_LIST_BASE_MARKER + size);
	} else if (size <= UINT8_MAX) {
		int8_t values[2] = {LIST8_MARKER, size};
    	buffer_write(&client->write_buf.write, values, 2);
	} else if (size <= UINT16_MAX) {
		int8_t values[3] = {LIST16_MARKER, 0, 0};
		*(uint16_t *)(values + 1) = htons(size);
		buffer_write(&client->write_buf.write, values, 3);
	} else {
		int8_t values[5] = {LIST32_MARKER, 0, 0, 0, 0};
		*(uint32_t *)(values + 1) = htonl(size);
		buffer_write(&client->write_buf.write, values, 5);
	}
}

// write map header to client response buffer
// expected 'size' number of key-value pairs to follow
// key should be string
// value can be any type
void bolt_reply_map
(
	bolt_client_t *client,  // client to write to
	uint32_t size           // number of key-value pairs to follow
) {
	ASSERT(client != NULL);

	if (size < TINY_SIZE) {
		buffer_write_uint8(&client->write_buf.write, TINY_MAP_BASE_MARKER + size);
	} else if (size <= UINT8_MAX) {
		int8_t values[2] = {MAP8_MARKER, size};
    	buffer_write(&client->write_buf.write, values, 2);
	} else if (size <= UINT16_MAX) {
		int8_t values[3] = {MAP16_MARKER, 0, 0};
		*(uint16_t *)(values + 1) = htons(size);
		buffer_write(&client->write_buf.write, values, 3);
	} else {
		int8_t values[5] = {MAP32_MARKER, 0, 0, 0, 0};
		*(uint32_t *)(values + 1) = htonl(size);
		buffer_write(&client->write_buf.write, values, 5);
	}
}

// write structure header to client response buffer
// expected 'size' number of items to follow
void bolt_reply_structure
(
	bolt_client_t *client,     // client to write to
	bolt_structure_type type,  // structure type
	uint32_t size              // number of items to follow
) {
	ASSERT(client != NULL);

	int8_t values[2] = {STRUCTURE_BASE_MARKER + size, type};
    buffer_write(&client->write_buf.write, values, 2);
}

//------------------------------------------------------------------------------
// Read functions
//------------------------------------------------------------------------------

// read value type from buffer
bolt_value_type bolt_read_type
(
	buffer_index_t data  // buffer to read from
) {
	uint8_t marker = buffer_read_uint8(&data);
	switch (marker)
	{
		case NULL_MARKER:
			return BVT_NULL;
		case FLOAT_MARKER:
			return BVT_FLOAT;
		case TRUE_MARKER:
		case FALSE_MARKER:
			return BVT_BOOL;
		case INT8_MARKER:
			return BVT_INT8;
		case INT16_MARKER:
			return BVT_INT16;
		case INT32_MARKER:
			return BVT_INT32;
		case INT64_MARKER:
			return BVT_INT64;
		case BYTES8_MARKER:
		case BYTES16_MARKER:
		case BYTES32_MARKER:
			return BVT_BYTES;
		case STRING8_MARKER:
		case STRING16_MARKER:
		case STRING32_MARKER:
			return BVT_STRING;
		case LIST8_MARKER:
		case LIST16_MARKER:
		case LIST32_MARKER:
			return BVT_LIST;
		case MAP8_MARKER:
		case MAP16_MARKER:
		case MAP32_MARKER:
			return BVT_MAP;
		default:
			if(marker >= TINY_INT8_MIN || marker <= TINY_INT8_MAX) {
				return BVT_INT8;
			}
			if(TINY_MARKER_CHECK(TINY_STRING_BASE_MARKER, marker)) {
				return BVT_STRING;
			}
			if(TINY_MARKER_CHECK(TINY_LIST_BASE_MARKER, marker)) {
				return BVT_LIST;
			}
			if(TINY_MARKER_CHECK(TINY_MAP_BASE_MARKER, marker)) {
				return BVT_MAP;
			}
			if(TINY_MARKER_CHECK(STRUCTURE_BASE_MARKER, marker)) {
				return BVT_STRUCTURE;
			}
			ASSERT(false);
			return BVT_NULL;
	}
}

// read null value from buffer
void bolt_read_null
(
	buffer_index_t *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	ASSERT(marker == NULL_MARKER);
}

// read bool value from buffer
bool bolt_read_bool
(
	buffer_index_t *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	switch (marker)
	{
		case FALSE_MARKER:
			return false;
		case TRUE_MARKER:
			return true;
		default:
			ASSERT(false);
			return false;
	}
}

// read int8 value from buffer
int8_t bolt_read_int8
(
	buffer_index_t *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	switch (marker)
	{
		case INT8_MARKER:
			return buffer_read_uint8(data);
		default:
			if(marker >= TINY_INT8_MIN || marker <= TINY_INT8_MAX) {
				return marker;
			}
			ASSERT(false);
			return 0;
	}
}

// read int16 value from buffer
int16_t bolt_read_int16
(
	buffer_index_t *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	switch (marker)
	{
		case INT16_MARKER:
			return ntohs(buffer_read_uint16(data));
		default:
			ASSERT(false);
			return 0;
	}
}

// read int32 value from buffer
int32_t bolt_read_int32
(
	buffer_index_t *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	switch (marker)
	{
		case INT32_MARKER:
			return ntohl(buffer_read_uint32(data));
		default:
			ASSERT(false);
			return 0;
	}
}

// read int64 value from buffer
int64_t bolt_read_int64
(
	buffer_index_t *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	switch (marker)
	{
		case INT64_MARKER:
			return ntohll(buffer_read_uint64(data));
		default:
			ASSERT(false);
			return 0;
	}
}

// read float value from buffer
double bolt_read_float
(
	buffer_index_t *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	switch (marker)
	{
		case FLOAT_MARKER: {
			double d;
			char *buf = (char *)&d;
			for (int i = sizeof(double); i > 0; i--) {
				buf[i - 1] = buffer_read_uint8(data);
			}
			return d;
		}
		default:
			ASSERT(false);
			return 0;
	}
}

// read string size from buffer
void _bolt_read_string_size
(
	buffer_index_t *data,  // buffer to read from
	uint32_t *size         // string size
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	switch (marker)
	{
		case STRING8_MARKER: {
			*size = buffer_read_uint8(data);
			break;
		}
		case STRING16_MARKER: {
			*size = ntohs(buffer_read_uint16(data));
			break;
		}
		case STRING32_MARKER: {
			*size = ntohl(buffer_read_uint32(data));
			break;
		}
		default:
			if(TINY_MARKER_CHECK(TINY_STRING_BASE_MARKER, marker)) {
				*size = marker - TINY_STRING_BASE_MARKER;
				break;
			}
			ASSERT(false);
			*size = 0;
			break;
	}
}

// read string size from buffer
void bolt_read_string_size
(
	buffer_index_t *data,  // buffer to read from
	uint32_t *size         // string size
) {
	ASSERT(data != NULL);

	buffer_index_t _data = *data;
	_bolt_read_string_size(&_data, size);
}

// read string value from buffer
// notice: the string is not null terminated
void bolt_read_string
(
	buffer_index_t *data,  // buffer to read from
	char *str              // string buffer
) {
	ASSERT(data != NULL);

	uint32_t size;
	_bolt_read_string_size(data, &size);

	buffer_index_t start = *data;
	buffer_index_read(data, str, size);
}

// read bytes size from buffer
uint32_t bolt_read_list_size
(
	buffer_index_t *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	switch (marker)
	{
		case LIST8_MARKER:
			return buffer_read_uint8(data);
		case LIST16_MARKER:
			return ntohs(buffer_read_uint16(data));
		case LIST32_MARKER:
			return ntohl(buffer_read_uint32(data));
		default:
			if(TINY_MARKER_CHECK(TINY_LIST_BASE_MARKER, marker)) {
				return marker - TINY_LIST_BASE_MARKER;
			}
			ASSERT(false);
			return 0;
	}
}

// read map size from buffer
uint32_t bolt_read_map_size
(
	buffer_index_t *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	switch (marker)
	{
		case MAP8_MARKER:
			return buffer_read_uint8(data);
		case MAP16_MARKER:
			return ntohs(buffer_read_uint16(data));
		case MAP32_MARKER:
			return ntohs(buffer_read_uint32(data));
		default:
			if(TINY_MARKER_CHECK(TINY_MAP_BASE_MARKER, marker)) {
				return marker - TINY_MAP_BASE_MARKER;
			}
			ASSERT(false);
			return 0;
	}
}

// read structure type from buffer
bolt_structure_type bolt_read_structure_type
(
	buffer_index_t *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	if(TINY_MARKER_CHECK(STRUCTURE_BASE_MARKER, marker)) {
		return buffer_read_uint8(data);
	}

	ASSERT(false);
	return 0;
}

// read structure size from buffer
uint32_t bolt_read_structure_size
(
	buffer_index_t *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	if(TINY_MARKER_CHECK(STRUCTURE_BASE_MARKER, marker)) {
		return marker - STRUCTURE_BASE_MARKER;
	}

	ASSERT(false);
	return 0;
}
