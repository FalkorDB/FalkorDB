/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "bolt.h"
#include "string.h"
#include "endian.h"

#define write(t, v) *(t *)client->current_write = v; \
	client->current_write += sizeof(t);

// write null to client response buffer
void bolt_reply_null
(
	bolt_client_t *client  // client to write to
) {
	ASSERT(client != NULL);

	client->current_write++[0] = 0xC0;
}

// write bool value to client response buffer
void bolt_reply_bool
(
	bolt_client_t *client,  // client to write to
	bool data               // bool value to write
) {
	ASSERT(client != NULL);

	client->current_write++[0] = data ? 0xC3 : 0xC2;
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

	client->current_write++[0] = data;
}

// write int8 value to client response buffer
void bolt_reply_int8
(
	bolt_client_t *client,  // client to write to
	int8_t data             // int8 value to write
) {
	ASSERT(client != NULL);

	client->current_write++[0] = 0xC8;
	client->current_write++[0] = data;
}

// write int16 value to client response buffer
void bolt_reply_int16
(
	bolt_client_t *client,  // client to write to
	int16_t data            // int16 value to write
) {
	ASSERT(client != NULL);

	client->current_write++[0] = 0xC9;
	write(int16_t, htons(data));
}

// write int32 value to client response buffer
void bolt_reply_int32
(
	bolt_client_t *client,  // client to write to
	int32_t data            // int32 value to write
) {
	ASSERT(client != NULL);

	client->current_write++[0] = 0xCA;
	write(int32_t, htonl(data));
}

// write int64 value to client response buffer
void bolt_reply_int64
(
	bolt_client_t *client, // client to write to
	int64_t data           // int64 value to write
) {
	ASSERT(client != NULL);

	client->current_write++[0] = 0xCB;
	write(int64_t, htonll(data));
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

	client->current_write++[0] = 0xC1;
	char *buf = (char *)&data;
	for (int i = 0; i < sizeof(double); i++) {
		client->current_write++[0] = buf[sizeof(double) - i - 1];
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
		client->current_write++[0] = 0x80 + size;
		memcpy(client->current_write, data, size);
		client->current_write += size;
	} else if (size < 0x100) {
		client->current_write++[0] = 0xD0;
		client->current_write++[0] = size;
		memcpy(client->current_write, data, size);
		client->current_write += size;
	} else if (size < 0x10000) {
		client->current_write++[0] = 0xD1;
		write(uint16_t, htons(size));
		memcpy(client->current_write, data, size);
		client->current_write += size;
	} else {
		client->current_write++[0] = 0xD2;
		write(uint32_t, htonl(size));
		memcpy(client->current_write, data, size);
		client->current_write += size;
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
		client->current_write++[0] = 0x90 + size;
	} else if (size < 0x100) {
		client->current_write++[0] = 0xD4;
		client->current_write++[0] = size;
	} else if (size < 0x10000) {
		client->current_write++[0] = 0xD5;
		write(uint16_t, htons(size));
	} else {
		client->current_write++[0] = 0xD6;
		write(uint32_t, htonl(size));
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
		client->current_write++[0] = 0xA0 + size;
	} else if (size < 0x100) {
		client->current_write++[0] = 0xD8;
		client->current_write++[0] = size;
	} else if (size < 0x10000) {
		client->current_write++[0] = 0xD9;
		write(uint16_t, htons(size));
	} else {
		client->current_write++[0] = 0xDA;
		write(uint32_t, htonl(size));
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

	client->current_write++[0] = 0xB0 + size;
	client->current_write++[0] = type;
}

// read value type from buffer
// return the pointer to the buffer after the value
static char *bolt_value_read
(
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
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
		case 0x8F:
			return data + 1 + marker - 0x80;
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
			data = data + 1;
			for (uint32_t i = 0; i < marker - 0x90; i++) {
				data = bolt_value_read(data);
			}
			return data;
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
			data = data + 1;
			for (uint32_t i = 0; i < marker - 0xA0; i++) {
				data = bolt_value_read(data);
				data = bolt_value_read(data);
			}
			return data;
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
			data = data + 2;
			for (uint32_t i = 0; i < marker - 0xB0; i++) {
				data = bolt_value_read(data);
			}
			return data;
		case 0xC0:
			return data + 1;
		case 0xC1:
			return data + 9;
		case 0xC2:
			return data + 1;
		case 0xC3:
			return data + 1;
		case 0xC8:
			return data + 2;
		case 0xC9:
			return data + 3;
		case 0xCA:
			return data + 5;
		case 0xCB:
			return data + 9;
		case 0xCC:
			return data + 2 + *(uint8_t *)(data + 1);
		case 0xCD:
			return data + 3 + ntohs(*(uint16_t *)(data + 1));
		case 0xCE:
			return data + 5 + ntohl(*(uint32_t *)(data + 1));
		case 0xD0:
			return data + 2 + *(uint8_t *)(data + 1);
		case 0xD1:
			return data + 3 + ntohs(*(uint16_t *)(data + 1));
		case 0xD2:
			return data + 5 + ntohl(*(uint32_t *)(data + 1));
		case 0xD4: {
			int n = (unsigned char)data[1];
			data = data + 2;
			for (uint32_t i = 0; i < n; i++) {
				data = bolt_value_read(data);
			}
			return data;
		}
		case 0xD5: {
			int n = ntohs(*(uint16_t *)(data + 1));
			data = data + 3;
			for (uint32_t i = 0; i < n; i++) {
				data = bolt_value_read(data);
			}
			return data;
		}
		case 0xD6: {
			int n = ntohl(*(uint32_t *)(data + 1));
			data = data + 5;
			for (uint32_t i = 0; i < n; i++) {
				data = bolt_value_read(data);
			}
			return data;
		}
		case 0xD8: {
			int n = (unsigned char)data[1];
			data = data + 2;
			for (uint32_t i = 0; i < n; i++) {
				data = bolt_value_read(data);
				data = bolt_value_read(data);
			}
			return data;
		}
		case 0xD9: {
			int n = ntohs(*(uint16_t *)(data + 1));
			data = data + 3;
			for (uint32_t i = 0; i < n; i++) {
				data = bolt_value_read(data);
				data = bolt_value_read(data);
			}
			return data;
		}
		case 0xDA: {
			int n = ntohl(*(uint32_t *)(data + 1));
			data = data + 5;
			for (uint32_t i = 0; i < n; i++) {
				data = bolt_value_read(data);
				data = bolt_value_read(data);
			}
			return data;
		}
		default:
			if(marker >= 0xF0 || marker <= 0x7F) {
				return data + 1;
			}
			ASSERT(false);
			break;
	}
}

// read value type from buffer
bolt_value_type bolt_read_type
(
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
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
			break;
	}
}

// read bool value from buffer
bool bolt_read_bool
(
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
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
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
	switch (marker)
	{
		case 0xC8:
			return data[1];
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
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
	switch (marker)
	{
		case 0xC9:
			return ntohs(*(uint16_t *)(data + 1));
		default:
			ASSERT(false);
			return 0;
	}
}

// read int32 value from buffer
int32_t bolt_read_int32
(
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
	switch (marker)
	{
		case 0xCA:
			return ntohl(*(uint32_t *)(data + 1));
		default:
			ASSERT(false);
			return 0;
	}
}

// read int64 value from buffer
int64_t bolt_read_int64
(
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
	switch (marker)
	{
		case 0xCB:
			return ntohll(*(uint64_t *)(data + 1));
		default:
			ASSERT(false);
			return 0;
	}
}

// read float value from buffer
double bolt_read_float
(
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
	switch (marker)
	{
		case 0xC1: {
			double d;
			char *buf = (char *)&d;
			for (int i = 0; i < sizeof(double); i++) {
				buf[i] = data[sizeof(double) - i];
			}
			return d;
		}
		default:
			ASSERT(false);
			return 0;
	}
}

// read string size from buffer
uint32_t bolt_read_string_size
(
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
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
		case 0x8F:
			return marker - 0x80;
		case 0xD0:
			return *(uint8_t *)(data + 1);
		case 0xD1:
			return ntohs(*(uint16_t *)(data + 1));
		case 0xD2:
			return ntohl(*(uint32_t *)(data + 1));
		default:
			ASSERT(false);
			return 0;
	}
}

// read string value from buffer
// notice: the string is not null terminated
char *bolt_read_string
(
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
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
		case 0x8F:
			return data + 1;
		case 0xD0:
			return data + 2;
		case 0xD1:
			return data + 3;
		case 0xD2:
			return data + 5;
		default:
			ASSERT(false);
			return 0;
	}
}

// read bytes size from buffer
uint32_t bolt_read_list_size
(
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
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
			return *(uint8_t *)(data + 1);
		case 0xD5:
			return ntohs(*(uint16_t *)(data + 1));
		case 0xD6:
			return ntohl(*(uint32_t *)(data + 1));
		default:
			ASSERT(false);
			return 0;
	}
}

// read list item from buffer
char *bolt_read_list_item
(
	char *data,     // buffer to read from
	uint32_t index  // index of the item to read
) {
	ASSERT(data != NULL);
	ASSERT(index < bolt_read_list_size(data));

	uint8_t marker = data[0];
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
			data = data + 1;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
			}
			return data;
		case 0xD4:
			data = data + 2;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
			}
			return data;
		case 0xD5:
			data = data + 3;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
			}
			return data;
		case 0xD6:
			data = data + 5;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
			}
			return data;
		default:
			ASSERT(false);
			return 0;
	}
}

// read map size from buffer
uint32_t bolt_read_map_size
(
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
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
			return *(uint8_t *)(data + 1);
		case 0xD9:
			return ntohs(*(uint16_t *)(data + 1));
		case 0xDA:
			return ntohs(*(uint32_t *)(data + 1));
		default:
			ASSERT(false);
			return 0;
	}
}

// read map key from buffer
char *bolt_read_map_key
(
	char *data,     // buffer to read from
	uint32_t index  // index of the key to read
) {
	ASSERT(data != NULL);
	ASSERT(index < bolt_read_map_size(data));

	uint8_t marker = data[0];
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
			data = data + 1;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
				data = bolt_value_read(data);
			}
			return data;
		case 0xD8:
			data = data + 2;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
				data = bolt_value_read(data);
			}
			return data;
		case 0xD9:
			data = data + 3;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
				data = bolt_value_read(data);
			}
			return data;
		case 0xDA:
			data = data + 5;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
				data = bolt_value_read(data);
			}
			return data;
		default:
			ASSERT(false);
			return 0;
	}
}

// read map value from buffer
char *bolt_read_map_value
(
	char *data,     // buffer to read from
	uint32_t index  // index of the value to read
) {
	ASSERT(data != NULL);
	ASSERT(index < bolt_read_map_size(data));

	uint8_t marker = data[0];
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
			data = data + 1;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
				data = bolt_value_read(data);
			}
			data = bolt_value_read(data);
			return data;
		case 0xD8:
			data = data + 2;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
				data = bolt_value_read(data);
			}
			data = bolt_value_read(data);
			return data;
		case 0xD9:
			data = data + 3;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
				data = bolt_value_read(data);
			}
			data = bolt_value_read(data);
			return data;
		case 0xDA:
			data = data + 5;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
				data = bolt_value_read(data);
			}
			data = bolt_value_read(data);
			return data;
		default:
			ASSERT(false);
			return 0;
	}
}

// read structure type from buffer
bolt_structure_type bolt_read_structure_type
(
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
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
			return data[1];
		default:
			ASSERT(false);
			return 0;
	}
}

// read structure size from buffer
uint32_t bolt_read_structure_size
(
	char *data  // buffer to read from
) {
	ASSERT(data != NULL);

	uint8_t marker = data[0];
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

// read structure value from buffer
char *bolt_read_structure_value
(
	char *data,     // buffer to read from
	uint32_t index  // index of the value to read
) {
	ASSERT(data != NULL);
	ASSERT(index < bolt_read_structure_size(data));

	uint8_t marker = data[0];
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
			data = data + 2;
			for (uint32_t i = 0; i < index; i++) {
				data = bolt_value_read(data);
			}
			return data;
		default:
			ASSERT(false);
			return 0;
	}
}
