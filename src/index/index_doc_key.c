/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "index_doc_key.h"
#include "RG.h"

#include <stdint.h>

static const char _hex_digits[] = "0123456789abcdef";

// Decode a single hex char to its nibble value.
// Returns 0xFF for any non-hex byte (sentinel for "not a hex char") — a
// previous version used a `static const uint8_t [256]` lookup table whose
// comment claimed unspecified slots were 0xFF, but C zero-init leaves them
// at 0x00, so malformed input silently decoded to zero instead of failing.
static inline uint8_t _hex_value(uint8_t c) {
	if(c >= '0' && c <= '9') return c - '0';
	if(c >= 'a' && c <= 'f') return c - 'a' + 0xa;
	if(c >= 'A' && c <= 'F') return c - 'A' + 0xa;
	return 0xFF;
}

static inline void _encode(const uint8_t *src, size_t src_len, char *out) {
	for(size_t i = 0; i < src_len; i++) {
		out[i * 2]     = _hex_digits[(src[i] >> 4) & 0xf];
		out[i * 2 + 1] = _hex_digits[src[i] & 0xf];
	}
	out[src_len * 2] = '\0';
}

static inline void _decode(const char *in, uint8_t *out, size_t out_len) {
	for(size_t i = 0; i < out_len; i++) {
		uint8_t hi = _hex_value((uint8_t)in[i * 2]);
		uint8_t lo = _hex_value((uint8_t)in[i * 2 + 1]);
		// Doc keys come from RediSearch which roundtrips what we encoded;
		// any non-hex byte indicates corruption — assert in debug builds
		// rather than silently decoding to zero.
		ASSERT(hi != 0xFF && lo != 0xFF);
		out[i] = (hi << 4) | lo;
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
