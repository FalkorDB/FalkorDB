/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "vector.h"
#include "GraphBLAS.h"
#include "../util/rmalloc.h"

// clones vector
SIValue SIVector_Clone
(
	SIValue vector // vector to clone
) {
	SIType t = SI_TYPE(vector);
	ASSERT(t & T_VECTOR);

	// make a copy of the internal GraphBLAS vector
	GrB_Vector dest;
	GrB_Vector src = vector.vector;
	GrB_Info info = GrB_Vector_dup(&dest, src);
	ASSERT(info == GrB_SUCCESS);

	// clone vector
	SIValue clone = vector;
	clone.vector = dest;
	clone.allocation = M_SELF;

	return clone;
}

// returns vector's dimension
uint64_t SIVector_Dim
(
	SIValue vector  // vector to get dimension of
) {
	ASSERT(SI_TYPE(vector) == T_VECTOR);

	GrB_Index n;
	GrB_Vector v = vector.vector;

	GrB_Info info = GrB_Vector_size(&n, v);
	ASSERT(info == GrB_SUCCESS);

	return n;
}

// returns vector's elements
void *SIVector_Unpack
(
	SIValue *vector,  // vector to unpack
	size_t *vx_size   // size of output in bytes
) {
	ASSERT(vector != NULL);
	ASSERT(SI_TYPE(*vector) & T_VECTOR);

	GrB_Vector v = vector->vector;

	// unpack vector
	void *vx;
	GrB_Info info = GxB_Vector_unpack_Full(v, &vx, (GrB_Index*)vx_size, NULL,
			NULL);
	ASSERT(info == GrB_SUCCESS);

	// return vector's elements
	return vx;
}

// packs elements into vector
void SIVector_Pack
(
	SIValue *vector,  // vector to pack
	void **elements,  // elements to pack
	size_t vx_size    // size of elements in bytes
) {
	ASSERT(vector   != NULL);
	ASSERT(elements != NULL);
	ASSERT(SI_TYPE(*vector) & T_VECTOR);

	GrB_Vector v = vector->vector;

	// pack elements into vector
	GrB_Info info = GxB_Vector_pack_Full(v, elements, vx_size, false, NULL);
	ASSERT(info == GrB_SUCCESS);
}

// write a string representation of vector to buf
void SIVector_ToString
(
	SIValue vector,       // vector to convert to string
	char **buf,           // output buffer
	size_t *bufferLen,    // output buffer length
	size_t *bytesWritten  // output bytes written
) {
	ASSERT(buf          != NULL);
	ASSERT(bufferLen    != NULL);
	ASSERT(bytesWritten != NULL);
	ASSERT(SI_TYPE(vector) == T_VECTOR);

	// compute required buffer size
	// each element is represented by 24 characters
	uint64_t dim = SIVector_Dim(vector);
	size_t requiredLen = dim * 26 + 2 ;
	size_t availableLen = *bufferLen - *bytesWritten;
	requiredLen -= availableLen;

	// make sure buffer is large enough
	if(requiredLen > 0) {
		*bufferLen += requiredLen;
		*buf = rm_realloc(*buf, *bufferLen);
	}

	// write opening bracket
	*bytesWritten += sprintf(*buf + *bytesWritten, "[");

    // create an iterator
	GrB_Info info;
    GxB_Iterator it;
    GxB_Iterator_new(&it);
	GrB_Vector v = vector.vector;

	GrB_Type t;
	info = GxB_Vector_type(&t, v);
	ASSERT(info == GrB_SUCCESS);
	ASSERT(t == GrB_FP64 || t == GrB_FP32);

    // attach it to the vector v
    info = GxB_Vector_Iterator_attach(it, v, NULL);
	ASSERT(info == GrB_SUCCESS);

    // seek to the first entry
    info = GxB_Vector_Iterator_seek(it, 0);

	if(t == GrB_FP64) {
		while(info != GxB_EXHAUSTED) {
			// get the entry v(i)
			double vi = GxB_Iterator_get_FP64(it);

			// move to the next entry in v
			info = GxB_Vector_Iterator_next(it);

			// write current element to buffer
			*bytesWritten += sprintf(*buf + *bytesWritten, "%f, ", vi);
		}
	} else {
		while(info != GxB_EXHAUSTED) {
			// get the entry v(i)
			float vi = GxB_Iterator_get_FP32(it);

			// move to the next entry in v
			info = GxB_Vector_Iterator_next(it);

			// write current element to buffer
			*bytesWritten += sprintf(*buf + *bytesWritten, "%lf, ", vi);
		}
	}

    GrB_free(&it);

	// write closing bracket
	if(dim > 0) {
		// remove trailing comma and space
		*bytesWritten -= 2;
	}
	*bytesWritten += sprintf(*buf + *bytesWritten, "]");
}

void SIVector_Free
(
	SIValue vector // vector to free
) {
	ASSERT(SI_TYPE(vector) & T_VECTOR);

	GrB_Vector v = vector.vector;
	GrB_Info info = GrB_Vector_free(&v);
	ASSERT(info == GrB_SUCCESS);
}

