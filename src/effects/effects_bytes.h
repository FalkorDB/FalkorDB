/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stddef.h>

// EffectsBytes is a growable append-only byte sequence
//
// Held as a linked list of fixed-size blocks, so an append never reallocates and
// never moves bytes already written. Its own type because more than one has to
// be alive at a time: v2 emits one record per effect in arrival order, so its
// bytes are final the moment an effect is recorded, while a v3 record states its
// count and shape ahead of its rows - so nothing can be serialized until the
// query stops producing effects and every group accumulates on its own. A group
// needs two, rows and values, concatenated once the count is known.
//
// Nothing here interprets the bytes. Widths, order and grouping are the
// caller's.
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
