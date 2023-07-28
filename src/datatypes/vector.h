/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"

// clones vector
SIValue SIVector_Clone
(
	SIValue vector // vector to clone
);

// returns vector's dimension
uint64_t SIVector_Dim
(
	SIValue vector // vector to get dimension of
);

// returns vector's elements
void *SIVector_Unpack
(
	SIValue *vector,  // vector to unpack
	size_t *vx_size   // size of output in bytes
);

// packs elements into vector
void SIVector_Pack
(
	SIValue *vector,  // vector to pack
	void **elements,  // elements to pack
	size_t vx_size    // size of elements in bytes
);

// write a string representation of vector to buf
void SIVector_ToString
(
	SIValue vector,       // vector to convert to string
	char **buf,           // output buffer
	size_t *bufferLen,    // output buffer length
	size_t *bytesWritten  // output bytes written
);

void SIVector_Free
(
	SIValue vector // vector to free
);

