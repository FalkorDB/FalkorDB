/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "rmalloc.h"
#include "uuid.h"

char *UUID_New() {
	/* Implementation is based on https://www.cryptosys.net/pki/uuid-rfc4122.html */

	// Generate 16 random bytes.
	unsigned char r[16];
	int i;

	for(i = 0; i < 16; i++) {
		r[i] = rand() % 0xff;
	}

	char *uuid = rm_malloc(37 * sizeof(char));
	sprintf(uuid, "%08x-%04x-%04x-%04x-%04x%08x",
			*((uint32_t *)r),
			*((uint16_t *)(r + 4)),
			// Set the four most significant bits of the 7th byte to 0100'B, so the high nibble is "4".
			(*((uint16_t *)(r + 6)) & 0b0000111111111111) | 0b0100000000000000,
			// Set the two most significant bits of the 9th byte to 10'B, so the high nibble will be one of "8", "9", "A", or "B" (see Note 1).
			(*((uint16_t *)(r + 8)) & 0b0011111111111111) | 0b1000000000000000,
			*((uint16_t *)(r + 10)),
			*((uint32_t *)(r + 12)));

	uuid[36] = '\0';
	return uuid;
}
