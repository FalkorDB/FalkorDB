/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stddef.h>

// Encode a FalkorDB TAG attribute value into a form that round-trips
// through RediSearch's TAG indexer and tag_strtolower untouched.
//
// FalkorDB configures its TAG fields with separator '\1' and the
// case-sensitive flag set, so RediSearch's TAG indexer stores the
// value as-is (no lowercasing, no splitting on commas). The query
// side still runs tag_strtolower, which strips a backslash whenever
// the following character is punct or whitespace. That asymmetry
// means a stored literal value containing 'backslash-then-punct' or
// any whitespace prefix/suffix does not match the same literal at
// query time.
//
// TagEncode_Lower (the name matches the design doc) replaces any
// byte that would create that asymmetry with an underscore-prefixed
// two-digit hex escape, leaving the rest unchanged. The encoding is
// reversible and stable, and applying it on both the write path and
// the exact-match query path produces a trie key that survives
// tag_strtolower without modification.
//
// Encoded bytes:
//   0x5c '\\' -- read-side tag_strtolower would strip this
//   0x5f '_'  -- our escape prefix; reserved
//   0x01 '\1' -- FalkorDB's TAG separator
//   any isspace() byte (' ', '\t', '\n', '\v', '\f', '\r')
//   any other control byte (< 0x20)
//
// `*out` is rm_malloc'd by the function; the caller owns it and is
// responsible for rm_free. The returned buffer is NUL-terminated; the
// reported `*out_len` excludes the terminator.
void TagEncode_Lower
(
	const char *src,    // source string
	size_t src_len,     // length of source, in bytes
	char **out,         // [out] encoded string
	size_t *out_len     // [out] length of encoded string, excluding NUL
);
