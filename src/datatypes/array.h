/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../value.h"

// initialize a new SIValue array type with given capacity
// returns initialized array
SIValue SIArray_New
(
	u_int32_t initialCapacity // initial capacity
);

// creates an SIArray from a raw 'arr.h' array
// SIArray takes ownership over 'raw'
SIValue SIArray_FromRaw
(
	SIValue **raw  // raw array
);

// appends a new SIValue to array
// the value is cloned before it is added to the array
void SIArray_Append
(
	SIValue *siarray,  // pointer to array
	SIValue value      // value to add
);

// appends value to array
// the value is added as is, no cloning is performed
// the array takes ownership over the value
void SIArray_AppendAsOwner
(
	SIValue *siarray,  // pointer to array
	SIValue *value     // value to add
);

// returns a volatile copy of the SIValue from an array in a given index
// if index is out of bound, SI_NullVal is returned
// caller is expected either to not free the returned value or take ownership on
// its own copy
// returns the value in the requested index
SIValue SIArray_Get
(
	SIValue siarray,  // siarray: array
	u_int32_t index   // index: index
);

// get the array length
u_int32_t SIArray_Length
(
	SIValue siarray  // array to return length of
);

// returns true if any of the types in 't' are contained in the array
// or its nested array children, if any
bool SIArray_ContainsType
(
	SIValue siarray,  // array to inspect
	SIType t          // bitmap of types to search for
);

// returns true if the array contains an element equals to 'value'
bool SIArray_ContainsValue
(
	SIValue siarray,    // array to search
	SIValue value,      // value to search for
	bool *comparedNull  // indicate if there was a null comparison
);

// returns true if all of the elements in the array are of type 't'
bool SIArray_AllOfType
(
	SIValue siarray,  // array to inspect
	SIType t          // type to compare against
);

// sorts the array in place
void SIArray_Sort
(
	SIValue siarray,  // array to sort
	bool ascending    // sorting order
);

// clones an array, caller needs to free the array
SIValue SIArray_Clone
(
	SIValue siarray  // array to clone
);

// prints an array into a given buffer
void SIArray_ToString
(
	SIValue siarray,      // array to print
	char **buf,           // print buffer
	size_t *bufferLen,    // print buffer length
	size_t *bytesWritten  // the actual number of bytes written to the buffer
);

 // returns the array hash code.
XXH64_hash_t SIArray_HashCode
(
	SIValue siarray  // array to hash
);

// creates an array from its binary representation
// this is the reverse of SIArray_ToBinary
// x = SIArray_FromBinary(SIArray_ToBinary(y));
// x == y
SIValue SIArray_FromBinary
(
	FILE *stream  // stream containing binary representation of an array
);

// free an array
void SIArray_Free
(
	SIValue siarray  // array to free
);

