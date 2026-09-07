/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdio.h>

// read a length-prefixed, NUL-terminated string off an effects stream
//
// the length comes off the wire, so it is validated against the bytes actually
// remaining before it reaches an allocator, and the result is confirmed
// terminated - callers hand it straight to strcmp/strlen
//
// this replaces the `size_t l; fread(&l, ...); char buf[l];` pattern, which
// sized a stack array from a wire-supplied length
//
// returns NULL on a malformed string; on success the caller owns the result and
// must rm_free it
char *ReadWireString
(
	FILE *stream  // effects stream
);
