/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "index_doc_key.h"
#include "RG.h"

#include <stdint.h>

static const char _hex_digits[] = "0123456789abcdef";

// Lookup table: hex char ('0'-'9', 'a'-'f', 'A'-'F') -> nibble value.
// All other entries are 0xFF (sentinel for "not a hex char").
static const uint8_t _hex_lookup[256] = {
	['0'] = 0x0, ['1'] = 0x1, ['2'] = 0x2, ['3'] = 0x3,
	['4'] = 0x4, ['5'] = 0x5, ['6'] = 0x6, ['7'] = 0x7,
	['8'] = 0x8, ['9'] = 0x9,
	['a'] = 0xa, ['b'] = 0xb, ['c'] = 0xc, ['d'] = 0xd,
	['e'] = 0xe, ['f'] = 0xf,
	['A'] = 0xa, ['B'] = 0xb, ['C'] = 0xc, ['D'] = 0xd,
	['E'] = 0xe, ['F'] = 0xf,
};

static inline void _encode(const uint8_t *src, size_t src_len, char *out) {
	for(size_t i = 0; i < src_len; i++) {
		out[i * 2]     = _hex_digits[(src[i] >> 4) & 0xf];
		out[i * 2 + 1] = _hex_digits[src[i] & 0xf];
	}
	out[src_len * 2] = '\0';
}

static inline void _decode(const char *in, uint8_t *out, size_t out_len) {
	for(size_t i = 0; i < out_len; i++) {
		out[i] = (_hex_lookup[(uint8_t)in[i * 2]] << 4)
		       | _hex_lookup[(uint8_t)in[i * 2 + 1]];
	}
}

void IndexDocKey_EncodeNode
(
	EntityID id,
	char out[NODE_DOC_KEY_BUF]
) {
	_encode((const uint8_t *)&id, sizeof(EntityID), out);
}

void IndexDocKey_DecodeNode
(
	const char *in,
	EntityID *out
) {
	ASSERT(in  != NULL);
	ASSERT(out != NULL);
	_decode(in, (uint8_t *)out, sizeof(EntityID));
}

void IndexDocKey_EncodeEdge
(
	const EdgeIndexKey *key,
	char out[EDGE_DOC_KEY_BUF]
) {
	ASSERT(key != NULL);
	_encode((const uint8_t *)key, sizeof(EdgeIndexKey), out);
}

void IndexDocKey_DecodeEdge
(
	const char *in,
	EdgeIndexKey *out
) {
	ASSERT(in  != NULL);
	ASSERT(out != NULL);
	_decode(in, (uint8_t *)out, sizeof(EdgeIndexKey));
}
