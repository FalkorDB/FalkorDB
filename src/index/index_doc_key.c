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

static inline bool _decode(const char *in, uint8_t *out, size_t out_len) {
	for(size_t i = 0; i < out_len; i++) {
		uint8_t hi = _hex_value((uint8_t)in[i * 2]);
		uint8_t lo = _hex_value((uint8_t)in[i * 2 + 1]);
		// Doc keys round-trip from RediSearch; a non-hex byte means corruption
		// -- assert in debug, fail in release rather than silently decoding to
		// a wrong value.
		if(hi == 0xFF || lo == 0xFF) {
			ASSERT(false);
			return false;
		}
		out[i] = (hi << 4) | lo;
	}
	return true;
}

void IndexDocKey_EncodeNode
(
	EntityID id,
	char out[NODE_DOC_KEY_BUF_SIZE]
) {
	_encode((const uint8_t *)&id, sizeof(EntityID), out);
}

bool IndexDocKey_DecodeNode
(
	const char *in,
	size_t in_len,
	EntityID *out
) {
	ASSERT(in  != NULL);
	ASSERT(out != NULL);
	if(in_len != NODE_DOC_KEY_LEN) {
		ASSERT(false);
		return false;
	}
	return _decode(in, (uint8_t *)out, sizeof(EntityID));
}

void IndexDocKey_EncodeEdge
(
	const EdgeIndexKey *key,
	char out[EDGE_DOC_KEY_BUF_SIZE]
) {
	ASSERT(key != NULL);
	_encode((const uint8_t *)key, sizeof(EdgeIndexKey), out);
}

bool IndexDocKey_DecodeEdge
(
	const char *in,
	size_t in_len,
	EdgeIndexKey *out
) {
	ASSERT(in  != NULL);
	ASSERT(out != NULL);
	if(in_len != EDGE_DOC_KEY_LEN) {
		ASSERT(false);
		return false;
	}
	return _decode(in, (uint8_t *)out, sizeof(EdgeIndexKey));
}
