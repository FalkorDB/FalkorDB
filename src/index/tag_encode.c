/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "tag_encode.h"
#include "RG.h"
#include "../util/rmalloc.h"

#include <ctype.h>
#include <stdbool.h>
#include <stdint.h>

static const char _hex_digits[] = "0123456789abcdef";

static inline bool _needs_escape(uint8_t b) {
	if(b == 0x5c) return true;        // '\\' -- read-side strip target
	if(b == 0x5f) return true;        // '_'  -- reserved escape prefix
	if(b == 0x01) return true;        // '\1' -- FalkorDB tag separator
	if(b < 0x20)  return true;        // other control bytes
	if(isspace(b)) return true;       // whitespace
	return false;
}

void TagEncode_Lower
(
	const char *src,
	size_t src_len,
	char **out,
	size_t *out_len
) {
	ASSERT(src     != NULL);
	ASSERT(out     != NULL);
	ASSERT(out_len != NULL);

	// upper bound: every byte expands to 3 chars
	char *buf = rm_malloc(src_len * 3 + 1);
	size_t j = 0;

	for(size_t i = 0; i < src_len; i++) {
		uint8_t b = (uint8_t)src[i];
		if(_needs_escape(b)) {
			buf[j++] = '_';
			buf[j++] = _hex_digits[(b >> 4) & 0xf];
			buf[j++] = _hex_digits[b & 0xf];
		} else {
			buf[j++] = (char)b;
		}
	}
	buf[j] = '\0';

	*out     = buf;
	*out_len = j;
}
