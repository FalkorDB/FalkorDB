/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "bolt.h"
#include "string.h"
#include "endian.h"

// write null to client response buffer
void bolt_reply_null
(
	bolt_client_t *client  // client to write to
) {
	ASSERT(client != NULL);

	buffer_write_uint8(&client->write_buf.write, 0xC0);
}

// write bool value to client response buffer
void bolt_reply_bool
(
	bolt_client_t *client,  // client to write to
	bool data               // bool value to write
) {
	ASSERT(client != NULL);

	buffer_write_uint8(&client->write_buf.write, data ? 0xC3 : 0xC2);
}

// write tiny int value to client response buffer
// tiny int: -16 to 127
void bolt_reply_tiny_int
(
	bolt_client_t *client,  // client to write to
	int8_t data            // tiny int value to write
) {
	ASSERT(client != NULL);
	ASSERT(data >= -16 && data <= 127);

	buffer_write_uint8(&client->write_buf.write, data);
}

// write int8 value to client response buffer
void bolt_reply_int8
(
	bolt_client_t *client,  // client to write to
	int8_t data             // int8 value to write
) {
	ASSERT(client != NULL);

	buffer_write_uint8(&client->write_buf.write, 0xC8);
	buffer_write_uint8(&client->write_buf.write, data);
}

// write int16 value to client response buffer
void bolt_reply_int16
(
	bolt_client_t *client,  // client to write to
	int16_t data            // int16 value to write
) {
	ASSERT(client != NULL);

	buffer_write_uint8(&client->write_buf.write, 0xC9);
	buffer_write_uint16(&client->write_buf.write, htons(data));
}

// write int32 value to client response buffer
void bolt_reply_int32
(
	bolt_client_t *client,  // client to write to
	int32_t data            // int32 value to write
) {
	ASSERT(client != NULL);

	buffer_write_uint8(&client->write_buf.write, 0xCA);
	buffer_write_uint32(&client->write_buf.write, htonl(data));
}

// write int64 value to client response buffer
void bolt_reply_int64
(
	bolt_client_t *client, // client to write to
	int64_t data           // int64 value to write
) {
	ASSERT(client != NULL);

	buffer_write_uint8(&client->write_buf.write, 0xCB);
	buffer_write_uint64(&client->write_buf.write, htonll(data));
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

	if(data >= 0xF0 && data <= 0x7F) {
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

	buffer_write_uint8(&client->write_buf.write, 0xC1);
	char *buf = (char *)&data;
	for (int i = 0; i < sizeof(double); i++) {
		buffer_write_uint8(&client->write_buf.write, buf[sizeof(double) - i - 1]);
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

	if (size < 0x10) {
		buffer_write_uint8(&client->write_buf.write, 0x80 + size);
		buffer_write(&client->write_buf.write, data, size);
	} else if (size < 0x100) {
		buffer_write_uint8(&client->write_buf.write, 0xD0);
		buffer_write_uint8(&client->write_buf.write, size);
		buffer_write(&client->write_buf.write, data, size);
	} else if (size < 0x10000) {
		buffer_write_uint8(&client->write_buf.write, 0xD1);
		buffer_write_uint16(&client->write_buf.write, htons(size));
		buffer_write(&client->write_buf.write, data, size);
	} else {
		buffer_write_uint8(&client->write_buf.write, 0xD2);
		buffer_write_uint32(&client->write_buf.write, htonl(size));
		buffer_write(&client->write_buf.write, data, size);
	}
}

// write list header to client response buffer
// expected 'size' number of items to follow
void bolt_reply_list
(
	bolt_client_t *client,
	uint32_t size
) {
	ASSERT(client != NULL);

	if (size < 0x10) {
		buffer_write_uint8(&client->write_buf.write, 0x90 + size);
	} else if (size < 0x100) {
		buffer_write_uint8(&client->write_buf.write, 0xD4);
		buffer_write_uint8(&client->write_buf.write, size);
	} else if (size < 0x10000) {
		buffer_write_uint8(&client->write_buf.write, 0xD5);
		buffer_write_uint16(&client->write_buf.write, htons(size));
	} else {
		buffer_write_uint8(&client->write_buf.write, 0xD6);
		buffer_write_uint32(&client->write_buf.write, htonl(size));
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

	if (size < 0x10) {
		buffer_write_uint8(&client->write_buf.write, 0xA0 + size);
	} else if (size < 0x100) {
		buffer_write_uint8(&client->write_buf.write, 0xD8);
		buffer_write_uint8(&client->write_buf.write, size);
	} else if (size < 0x10000) {
		buffer_write_uint8(&client->write_buf.write, 0xD9);
		buffer_write_uint16(&client->write_buf.write, htons(size));
	} else {
		buffer_write_uint8(&client->write_buf.write, 0xDA);
		buffer_write_uint32(&client->write_buf.write, htonl(size));
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

	buffer_write_uint8(&client->write_buf.write, 0xB0 + size);
	buffer_write_uint8(&client->write_buf.write, type);
}

// read value type from buffer
bolt_value_type bolt_read_type
(
	buffer_index_t data  // buffer to read from
) {
	uint8_t marker = buffer_read_uint8(&data);
	switch (marker)
	{
		case 0xC0:
			return BVT_NULL;
		case 0xC1:
			return BVT_FLOAT;
		case 0xC2:
		case 0xC3:
			return BVT_BOOL;
		case 0xC8:
			return BVT_INT8;
		case 0xC9:
			return BVT_INT16;
		case 0xCA:
			return BVT_INT32;
		case 0xCB:
			return BVT_INT64;
		case 0xCC:
		case 0xCD:
		case 0xCE:
			return BVT_BYTES;
		case 0x80:
		case 0x81:
		case 0x82:
		case 0x83:
		case 0x84:
		case 0x85:
		case 0x86:
		case 0x87:
		case 0x88:
		case 0x89:
		case 0x8A:
		case 0x8B:
		case 0x8C:
		case 0x8D:
		case 0x8E:
		case 0x8F:
		case 0xD0:
		case 0xD1:
		case 0xD2:
			return BVT_STRING;
		case 0x90:
		case 0x91:
		case 0x92:
		case 0x93:
		case 0x94:
		case 0x95:
		case 0x96:
		case 0x97:
		case 0x98:
		case 0x99:
		case 0x9A:
		case 0x9B:
		case 0x9C:
		case 0x9D:
		case 0x9E:
		case 0x9F:
		case 0xD4:
		case 0xD5:
		case 0xD6:
			return BVT_LIST;
		case 0xA0:
		case 0xA1:
		case 0xA2:
		case 0xA3:
		case 0xA4:
		case 0xA5:
		case 0xA6:
		case 0xA7:
		case 0xA8:
		case 0xA9:
		case 0xAA:
		case 0xAB:
		case 0xAC:
		case 0xAD:
		case 0xAE:
		case 0xAF:
		case 0xD8:
		case 0xD9:
		case 0xDA:
			return BVT_MAP;
		case 0xB0:
		case 0xB1:
		case 0xB2:
		case 0xB3:
		case 0xB4:
		case 0xB5:
		case 0xB6:
		case 0xB7:
		case 0xB8:
		case 0xB9:
		case 0xBA:
		case 0xBB:
		case 0xBC:
		case 0xBD:
		case 0xBE:
		case 0xBF:
			return BVT_STRUCTURE;
		default:
			if(marker >= 0xF0 || marker <= 0x7F) {
				return BVT_INT8;
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
	ASSERT(marker == 0xC0);
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
		case 0xC2:
			return false;
		case 0xC3:
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
		case 0xC8:
			return buffer_read_uint8(data);
		default:
			if(marker >= 0xF0 || marker <= 0x7F) {
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
		case 0xC9:
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
		case 0xCA:
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
		case 0xCB:
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
		case 0xC1: {
			double d;
			char *buf = (char *)&d;
			for (int i = 0; i < sizeof(double); i++) {
				buf[sizeof(double) - i - 1] = buffer_read_uint8(data);
			}
			return d;
		}
		default:
			ASSERT(false);
			return 0;
	}
}

// read string value from buffer
// notice: the string is not null terminated
buffer_index_t bolt_read_string
(
	buffer_index_t *data,  // buffer to read from
	uint32_t *size         // string size
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	switch (marker)
	{
		case 0x80:
		case 0x81:
		case 0x82:
		case 0x83:
		case 0x84:
		case 0x85:
		case 0x86:
		case 0x87:
		case 0x88:
		case 0x89:
		case 0x8A:
		case 0x8B:
		case 0x8C:
		case 0x8D:
		case 0x8E:
		case 0x8F: {
			*size = marker - 0x80;
			break;
		}
		case 0xD0: {
			*size = buffer_read_uint8(data);
			break;
		}
		case 0xD1: {
			*size = ntohs(buffer_read_uint16(data));
			break;
		}
		case 0xD2: {
			*size = ntohl(buffer_read_uint32(data));
			break;
		}
		default:
			ASSERT(false);
			*size = 0;
			break;
	}

	buffer_index_t ret = *data;
	buffer_index_read(data, *size);
	return ret;
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
		case 0x90:
		case 0x91:
		case 0x92:
		case 0x93:
		case 0x94:
		case 0x95:
		case 0x96:
		case 0x97:
		case 0x98:
		case 0x99:
		case 0x9A:
		case 0x9B:
		case 0x9C:
		case 0x9D:
		case 0x9E:
		case 0x9F:
			return marker - 0x90;
		case 0xD4:
			return buffer_read_uint8(data);
		case 0xD5:
			return ntohs(buffer_read_uint16(data));
		case 0xD6:
			return ntohl(buffer_read_uint32(data));
		default:
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
		case 0xA0:
		case 0xA1:
		case 0xA2:
		case 0xA3:
		case 0xA4:
		case 0xA5:
		case 0xA6:
		case 0xA7:
		case 0xA8:
		case 0xA9:
		case 0xAA:
		case 0xAB:
		case 0xAC:
		case 0xAD:
		case 0xAE:
		case 0xAF:
			return marker - 0xA0;
		case 0xD8:
			return buffer_read_uint8(data);
		case 0xD9:
			return ntohs(buffer_read_uint16(data));
		case 0xDA:
			return ntohs(buffer_read_uint32(data));
		default:
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
	switch (marker)
	{
		case 0xB0:
		case 0xB1:
		case 0xB2:
		case 0xB3:
		case 0xB4:
		case 0xB5:
		case 0xB6:
		case 0xB7:
		case 0xB8:
		case 0xB9:
		case 0xBA:
		case 0xBB:
		case 0xBC:
		case 0xBD:
		case 0xBE:
		case 0xBF:
			return buffer_read_uint8(data);
		default:
			ASSERT(false);
			return 0;
	}
}

// read structure size from buffer
uint32_t bolt_read_structure_size
(
	buffer_index_t *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = buffer_read_uint8(data);
	switch (marker)
	{
		case 0xB0:
		case 0xB1:
		case 0xB2:
		case 0xB3:
		case 0xB4:
		case 0xB5:
		case 0xB6:
		case 0xB7:
		case 0xB8:
		case 0xB9:
		case 0xBA:
		case 0xBB:
		case 0xBC:
		case 0xBD:
		case 0xBE:
		case 0xBF:
			return marker - 0xB0;
		default:
			ASSERT(false);
			return 0;
	}
}
