/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"

// initialize a new SIValue array type with given capacity
SIValue SIArray_New
(
	u_int32_t initialCapacity
);

// appends a new SIValue to a given array
void SIArray_Append
(
	SIValue *siarray,
	SIValue value
);

// returns a volatile copy of the SIValue from an array in a given index
// if index is out of bound, SI_NullVal is returned
// caller is expected either to not free the returned value or take ownership on
// its own copy
SIValue SIArray_Get
(
	SIValue siarray,
	u_int32_t index
);

// returns the array length
u_int32_t SIArray_Length
(
	SIValue siarray
);

// returns true if any of the types in 't' are contained in the array
// or its nested array children, if any
bool SIArray_ContainsType
(
	SIValue siarray,
	SIType t
);

// returns true if the array contains an element equals to 'value'
bool SIArray_ContainsValue
(
	SIValue siarray,
	SIValue value,
	bool *comparedNull
);

// returns true if all of the elements in the array are of type 't'
bool SIArray_AllOfType
(
	SIValue siarray,
	SIType t
);

// sorts the array in place
void SIArray_Sort
(
	SIValue siarray,
	bool ascending
);

// returns a copy of the array
SIValue SIArray_Clone
(
	SIValue siarray
);

// prints an array into a given buffer
void SIArray_ToString
(
	SIValue list,
	char **buf,
	size_t *bufferLen,
	size_t *bytesWritten
);

// returns the array hash code
XXH64_hash_t SIArray_HashCode
(
	SIValue siarray
);

// encode array to binary stream
void SIArray_ToBinary
(
	SerializerIO stream,    // binary stream
	const SIValue *siarray  // array to encode
);

// read array from binary stream
SIValue SIArray_FromBinary
(
	SerializerIO stream  // binary stream
);

// free an array
void SIArray_Free
(
	SIValue siarray
);

