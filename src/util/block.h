/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdlib.h>
#include <sys/types.h>

/* The Block is a type-agnostic block of continuous memory used to hold items of the same type.
 * Each block has a next pointer to another block, or NULL if this is the last block. */
typedef struct Block {
	size_t itemSize;        // Size of a single item in bytes.
	struct Block *next;     // Pointer to next block.
	unsigned char data[];   // Item array. MUST BE LAST MEMBER OF THE STRUCT!
} Block;

Block *Block_New(uint itemSize, uint capacity);
void Block_Free(Block *block);

