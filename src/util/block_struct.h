/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// a Block is a type-agnostic allocation of continuous memory used to hold items
// of the same type
// each block has a next pointer to another block
// or NULL if this is the last block
// the block's `data` is composed of a leading HEADERS section 1 byte per item
// 8 bytes aligned followed by the actual items, where each item must be 8 bytes
// algined
typedef struct Block {
	struct Block *next;       // linked list connectivity
	size_t itemSize;          // size of a single item in bytes
	uint32_t cap;             // block capacity
	int32_t offloaded_count;  // number of offloaded elements
	int32_t deleted_count;    // number of deleted entries

	// convenience pointer: points to data + ALIGN8(capacity)
	// this avoids recalculating the padding offset at runtime
	unsigned char *elements;

	// layout: [uint8_t headers[capacity]] [padding] [elements[capacity]]
	unsigned char data[];
} Block;

