/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// hex <-> bytes, shared by the v3 wire tests
//
// Every v3 encoder test states its expectation as a hex string copied from the
// conformance corpus, and reports a mismatch as hex so the diff can be read
// against the fixture. Both directions lived in two test files with two
// different _unhex: one tolerant of separators, one not - so the same fixture
// could parse in one file and not the other.
//
// static inline in a header rather than a .c, so no unit test target has to
// gain a source file to use it.

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

// one hex digit's value, or -1 if it is not one
static inline int EffectsV3Test_Nibble
(
	char c  // character
) {
	if(c >= '0' && c <= '9') return c - '0';
	if(c >= 'a' && c <= 'f') return c - 'a' + 10;
	if(c >= 'A' && c <= 'F') return c - 'A' + 10;
	return -1;
}

// decode a hex string into bytes, setting *n to how many
//
// NON-HEX CHARACTERS ARE SKIPPED, so a fixture can be broken across lines and
// spaced into fields without changing what it means. Caller frees.
static inline unsigned char *EffectsV3Test_Unhex
(
	const char *hex,  // hex string
	size_t *n         // out: number of bytes decoded
) {
	size_t len = 0;
	for(const char *p = hex; *p; p++) {
		if(EffectsV3Test_Nibble(*p) >= 0) len++;
	}

	unsigned char *out = (unsigned char *)malloc(len / 2 + 1);
	size_t k  = 0;
	int    hi = -1;

	for(const char *p = hex; *p; p++) {
		int v = EffectsV3Test_Nibble(*p);
		if(v < 0) continue;
		if(hi < 0) { hi = v; continue; }
		out[k++] = (unsigned char)((hi << 4) | v);
		hi = -1;
	}

	*n = k;
	return out;
}

// encode bytes as a lowercase hex string. Caller frees
static inline char *EffectsV3Test_Hex
(
	const unsigned char *b,  // bytes
	size_t n                 // how many
) {
	char *out = (char *)malloc(2 * n + 1);
	for(size_t i = 0; i < n; i++) {
		snprintf(out + 2 * i, 3, "%02x", b[i]);
	}
	out[2 * n] = '\0';
	return out;
}
