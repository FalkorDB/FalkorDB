/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "array.h"
#include "../util/arr.h"
#include "../util/qsort.h"
#include "xxhash.h"

#include <limits.h>

// initialize a new SIValue array type with given capacity
// returns initialized array
SIValue SIArray_New
(
	u_int32_t initialCapacity // initial capacity
) {
	SIValue siarray;

	siarray.array      = array_new (SIValue, initialCapacity) ;
	siarray.type       = T_ARRAY ;
	siarray.allocation = M_SELF ;

	return siarray;
}

// creates an SIArray from a raw 'arr.h' array
// SIArray takes ownership over 'raw'
SIValue SIArray_FromRaw
(
	SIValue **raw  // raw array
) {
	ASSERT(raw  != NULL);
	ASSERT(*raw != NULL);

	SIValue siarray;

	siarray.array      = *raw;
	siarray.type       = T_ARRAY;
	siarray.allocation = M_SELF;

	*raw = NULL;

	return siarray;
}

// appends a new SIValue to array
// the value is cloned before it is added to the array
void SIArray_Append
(
	SIValue *siarray,  // pointer to array
	SIValue value      // value to add
) {
	ASSERT(siarray != NULL);

	// clone and persist incase of pointer values
	SIValue clone = SI_CloneValue(value);

	array_append(siarray->array, clone);
}

// appends value to array
// the value is added as is, no cloning is performed
// the array takes ownership over the value
void SIArray_AppendAsOwner
(
	SIValue *siarray,  // pointer to array
	SIValue *value     // value to add
) {
	ASSERT(value   != NULL);
	ASSERT(siarray != NULL);
	ASSERT(SI_ALLOCATION(value) != M_VOLATILE);

	// add value as is
	array_append(siarray->array, *value);
	SIValue_SetAllocationType(value, M_VOLATILE);
}

// returns a volatile copy of the SIValue from an array in a given index
// if index is out of bound, SI_NullVal is returned
// caller is expected either to not free the returned value or take ownership on
// its own copy
// returns the value in the requested index
SIValue SIArray_Get
(
	SIValue siarray,  // siarray: array
	u_int32_t index   // index: index
) {
	SIValue *v = SIArray_GetRef (siarray, index) ;

	if (unlikely (v == NULL)) {
		return SI_NullVal () ;
	} else {
		return SI_ShareValue (*v) ;
	}
}

// get a reference to the 'idx' element on the array
// if index is out of bounds NULL is returned
SIValue *SIArray_GetRef
(
	SIValue siarray,  // array
	u_int32_t index   // index
) {
	// check index
	if (unlikely (index >= SIArray_Length (siarray))) {
		return NULL ;
	}

	return siarray.array + index ;
}

// get the array length
u_int32_t SIArray_Length
(
	SIValue siarray  // array to return length of
) {
	return array_len(siarray.array);
}

// returns true if any of the types in 't' are contained in the array
// or its nested array children, if any
bool SIArray_ContainsType
(
	SIValue siarray,  // array to inspect
	SIType t          // bitmap of types to search for
) {
	uint array_len = SIArray_Length(siarray);
	for(uint i = 0; i < array_len; i++) {
		SIValue elem = siarray.array[i];
		if(SI_TYPE(elem) & t) return true;

		// recursively check nested arrays
		if(SI_TYPE(elem) == T_ARRAY) {
			bool type_is_nested = SIArray_ContainsType(elem, t);
			if(type_is_nested) return true;
		}
	}
	return false;
}

// returns true if the array contains an element equals to 'value'
bool SIArray_ContainsValue
(
	SIValue siarray,    // array to search
	SIValue value,      // value to search for
	bool *comparedNull  // indicate if there was a null comparison
) {
	// indicate if there was a null comparison during the array scan
	if(comparedNull) *comparedNull = false;
	uint array_len = SIArray_Length(siarray);
	for(uint i = 0; i < array_len; i++) {
		int disjointOrNull = 0;
		SIValue elem = siarray.array[i];
		int compareValue = SIValue_Compare(elem, value, &disjointOrNull);
		if(disjointOrNull == COMPARED_NULL) {
			if(comparedNull) *comparedNull = true;
			continue;
		}
		if(compareValue == 0) return true;
	}
	return false;
}

// returns true if all of the elements in the array are of type 't'
bool SIArray_AllOfType
(
	SIValue siarray,  // array to inspect
	SIType t          // type to compare against
) {
	uint array_len = SIArray_Length(siarray);
	for(uint i = 0; i < array_len; i++) {
		SIValue elem = siarray.array[i];
		if((SI_TYPE(elem) & t) == 0) return false;
	}

	return true;
}

// compare two SIValues, wrt ascending order
static int _siarray_compare_func_asc
(
	const void *a,
	const void *b,
	void *data
) {
	return SIValue_Compare(*(SIValue*)a, *(SIValue*)b, NULL);
}

// compare two SIValues, wrt ascending order
static int _siarray_compare_func_desc
(
	const void *a,
	const void *b,
	void *data
) {
	return SIValue_Compare(*(SIValue*)b, *(SIValue*)a, NULL);
}

// sorts the array in place
void SIArray_Sort
(
	SIValue siarray,  // array to sort
	bool ascending    // sorting order
) {
	uint32_t arrayLen = SIArray_Length(siarray);

	if(ascending) {
		sort_r(siarray.array, arrayLen, sizeof(SIValue),
				_siarray_compare_func_asc, (void *)&ascending);
	} else {
		sort_r(siarray.array, arrayLen, sizeof(SIValue),
				_siarray_compare_func_desc, (void *)&ascending);
	}
}

// clones an array, caller needs to free the array
SIValue SIArray_Clone
(
	SIValue siarray  // array to clone
) {
	uint arrayLen = SIArray_Length(siarray);
	SIValue newArray = SIArray_New(arrayLen);
	for(uint i = 0; i < arrayLen; i++) {
		SIArray_Append(&newArray, siarray.array[i]);
	}
	return newArray;
}

// prints an array into a given buffer
void SIArray_ToString
(
	SIValue siarray,      // array to print
	char **buf,           // print buffer
	size_t *bufferLen,    // print buffer length
	size_t *bytesWritten  // the actual number of bytes written to the buffer
) {
	if(*bufferLen - *bytesWritten < 64) {
		*bufferLen += 64;
		*buf = rm_realloc(*buf, sizeof(char) * *bufferLen);
	}

	// open array with "["
	*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "[");
	uint arrayLen = SIArray_Length(siarray);
	for(uint i = 0; i < arrayLen; i ++) {
		// write the next value
		SIValue_ToString(siarray.array[i], buf, bufferLen, bytesWritten);
		// if it is not the last element, add ", "
		if(i != arrayLen - 1) {
			if(*bufferLen - *bytesWritten < 64) {
				*bufferLen += 64;
				*buf = rm_realloc(*buf, sizeof(char) * *bufferLen);
			}
			*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, ", ");
		}
	}

	if(*bufferLen - *bytesWritten < 2) {
		*bufferLen += 2;
		*buf = rm_realloc(*buf, sizeof(char) * *bufferLen);
	}

	// close array with "]"
	*bytesWritten += snprintf(*buf + *bytesWritten, *bufferLen, "]");
}

 // returns the array hash code.
XXH64_hash_t SIArray_HashCode
(
	SIValue siarray
) {
	SIType t = T_ARRAY;
	XXH64_hash_t hashCode = XXH64(&t, sizeof(t), 0);

	uint arrayLen = SIArray_Length(siarray);
	for(uint i = 0; i < arrayLen; i++) {
		SIValue value = siarray.array[i];
		hashCode = 31 * hashCode + SIValue_HashCode(value);
	}

	return hashCode;
}

// creates an array from its binary representation
// this is the reverse of SIArray_ToBinary
// x = SIArray_FromBinary(SIArray_ToBinary(y));
// x == y
SIValue SIArray_FromBinary
(
	FILE *stream  // stream containing binary representation of an array
) {
	// read number of elements
	uint32_t n;
	fread_assert(&n, sizeof(uint32_t), stream);

	SIValue arr = SIArray_New(n);

	for(uint32_t i = 0; i < n; i++) {
		array_append(arr.array, SIValue_FromBinary(stream));
	}

	return arr;
}

// free an array
void SIArray_Free
(
	SIValue siarray  // array to free
) {
	uint arrayLen = SIArray_Length(siarray);
	for(uint i = 0; i < arrayLen; i++) {
		SIValue value = siarray.array[i];
		SIValue_Free(value);
	}
	array_free(siarray.array);
}

