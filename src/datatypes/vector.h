/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"

// creates a new float32 vector
SIValue SIVectorf32_New
(
	uint32_t dim  // vector's dimension
);

// clones vector
SIValue SIVector_Clone
(
	SIValue vector // vector to clone
);

// compares two vectors
// return values:
// 0 - vectors are equal
// >0 - a > b
// <0 - a < b
int SIVector_Compare
(
	SIValue a, // first vector to compare
	SIValue b  // second vector to compare
);

// compute vector hashcode
XXH64_hash_t SIVector_HashCode
(
	SIValue v  // vector to compute hashcode for
);

// returns vector's elements
void *SIVector_Elements
(
	SIValue vector // vector to get elements of
);

// returns vector's dimension
uint32_t SIVector_Dim
(
	SIValue vector // vector to get dimension of
);

// returns number of bytes used to represent vector's elements
// for vectorf32 this is 4 * vector's dimension
size_t SIVector_ElementsByteSize
(
	SIValue vector // vector to get binary size of
);

// computes the euclidean distance between two vectors
// distance = sqrt(sum((a[i] - b[i])^2))
float SIVector_EuclideanDistance
(
	SIValue a,  // first vector
	SIValue b   // second vector
);

// computes the cosine distance between two vectors
// distance = 1 - dot(a, b) / (||a|| * ||b||)
float SIVector_CosineDistance
(
	SIValue a,  // first vector
	SIValue b   // second vector
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

