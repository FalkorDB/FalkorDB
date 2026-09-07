/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stddef.h>

// EffectsBytes is a growable append-only byte sequence
//
// held as a linked list of fixed-size blocks, so an append never reallocates
// and never moves bytes already written. This is the representation
// EffectsBuffer used to own directly; it is now its own type because more than
// one of them has to be alive at a time.
//
// v2 could write straight into a single sequence: it emits one record per
// effect, in arrival order, so the bytes are final the moment an effect is
// recorded. v3 cannot. A v3 payload is one record per (opcode, shape), and a
// record states its count and its shape ahead of its rows - so no record can
// be serialized until the query has finished producing effects, and every
// group has to accumulate on its own until then. A group needs two of these:
// its rows and its values grow independently, and are concatenated once the
// count is known.
//
// nothing here interprets the bytes. Widths, order and grouping belong to the
// callers.
typedef struct EffectsBytes EffectsBytes;

// create a new byte sequence, allocating space in blocks of block_size
EffectsBytes *EffectsBytes_New
(
	size_t block_size  // size of each block
);

// append n bytes from ptr
void EffectsBytes_Write
(
	EffectsBytes *bs,  // byte sequence
	const void *ptr,   // data to append
	size_t n           // number of bytes to append
);

// total number of bytes appended
size_t EffectsBytes_Len
(
	const EffectsBytes *bs  // byte sequence
);

// copy the whole sequence into dst, which must have room for
// EffectsBytes_Len(bs) bytes
//
// returns dst advanced past what was written, so consecutive sequences can be
// concatenated by threading the result
unsigned char *EffectsBytes_CopyInto
(
	const EffectsBytes *bs,  // byte sequence
	unsigned char *dst       // destination
);

// discard everything appended, retaining the first block for reuse
void EffectsBytes_Clear
(
	EffectsBytes *bs  // byte sequence
);

// free the byte sequence
void EffectsBytes_Free
(
	EffectsBytes *bs  // byte sequence
);
