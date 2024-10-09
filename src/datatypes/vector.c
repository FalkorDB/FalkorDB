/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "vector.h"
#include "xxhash.h"
#include "../util/rmalloc.h"

#include <math.h>

// vector struct
typedef struct {
	uint32_t dim;     // vector's dimension
	char elements[];  // vector's elements
} SIVector;

// creates a new float32 vector
SIValue SIVectorf32_New
(
	uint32_t dim  // vector's dimension
) {
	SIVector *v = rm_calloc(1, sizeof(SIVector) + dim * sizeof(float));
	v->dim = dim;

	return (SIValue) {
		.type       = T_VECTOR_F32,
		.ptrval     = (void*)v,
		.allocation = M_SELF
	};
}

// clones vector
SIValue SIVector_Clone
(
	SIValue vector // vector to clone
) {
	SIType t = SI_TYPE(vector);
	ASSERT(t & T_VECTOR);
	ASSERT(vector.ptrval != NULL);

	//--------------------------------------------------------------------------
	// create a new vector
	//--------------------------------------------------------------------------

	// determine number of bytes to allocate
	size_t elem_sz = sizeof(float);
	size_t n = sizeof(SIVector) + SIVector_Dim(vector) * elem_sz;

	SIVector *v = rm_malloc(n);

	// clone vector
	memcpy(v, vector.ptrval, n);

	return (SIValue) {
		.type       = t,
		.ptrval     = (void*)v,
		.allocation = M_SELF
	};
}

// compares two vectors
// return values:
// 0 - vectors are equal
// >0 - a > b
// <0 - a < b
int SIVector_Compare
(
	SIValue a, // first vector to compare
	SIValue b  // second vector to compare
) {
	ASSERT(SI_TYPE(a) & T_VECTOR);
	ASSERT(SI_TYPE(b) & T_VECTOR);
	ASSERT(SI_TYPE(a) == SI_TYPE(b));

	SIVector *va = (SIVector*)a.ptrval;
	SIVector *vb = (SIVector*)b.ptrval;

	// compare vectors' dimensions
	uint32_t dim_a = va->dim;
	uint32_t dim_b = vb->dim;

	if(dim_a != dim_b) return dim_a - dim_b;

	// compare vectors' elements
	float *elements_a = (float*)va->elements;
	float *elements_b = (float*)vb->elements;

	for(uint32_t i = 0; i < dim_a; i++) {
		if(elements_a[i] != elements_b[i]) {
			return elements_a[i] - elements_b[i];
		}
	}

	return 0;
}

// compute vector hashcode
XXH64_hash_t SIVector_HashCode
(
	SIValue v  // vector to compute hashcode for
) {
	SIType t = SI_TYPE(v);

	ASSERT(t & T_VECTOR);
	ASSERT(v.ptrval != NULL);

	SIVector *vector = (SIVector*)v.ptrval;
	XXH64_hash_t hashCode = XXH64(&t, sizeof(t), 0);
	size_t elem_size = sizeof(float);

	hashCode = hashCode * 31 +
		XXH64(vector->elements, vector->dim * elem_size, 0);

	return hashCode;
}

// returns vector's elements
void *SIVector_Elements
(
	SIValue vector // vector to get elements of
) {
	ASSERT(SI_TYPE(vector) & T_VECTOR);
	ASSERT(vector.ptrval != NULL);

	SIVector *v = (SIVector*)vector.ptrval;

	return (void*)v->elements;
}

// returns vector's dimension
uint32_t SIVector_Dim
(
	SIValue vector  // vector to get dimension of
) {
	ASSERT(SI_TYPE(vector) & T_VECTOR);
	ASSERT(vector.ptrval != NULL);

	SIVector *v = (SIVector*)vector.ptrval;

	return v->dim;
}

// returns number of bytes used to represent vector's elements
// for vector32f this is 4 * vector's dimension
// for vector64f this is 8 * vector's dimension
size_t SIVector_ElementsByteSize
(
	SIValue vector // vector to get binary size of
) {
	return SIVector_Dim(vector) * sizeof(float);
}

// computes the euclidean distance between two vectors
// distance = sqrt(sum((a[i] - b[i])^2))
float SIVector_EuclideanDistance
(
	SIValue a,  // first vector
	SIValue b   // second vector
) {
	// euclideanDistance(vecf32([0.2, 0.12, 0.3178]), vecf32([0.1, 0.2, 0.3]))
	ASSERT(SI_TYPE(a) & T_VECTOR);
	ASSERT(SI_TYPE(b) & T_VECTOR);

	// validate input vectors are of the same length
	uint32_t n1 = SIVector_Dim(a);
	uint32_t n2 = SIVector_Dim(b);
	ASSERT(n1 == n2);

	// compute the euclidean distance between the two vectors
	float sum        = 0;
	float *elements1 = (float*)SIVector_Elements(a);
	float *elements2 = (float*)SIVector_Elements(b);
	for(uint32_t i = 0; i < n1; i++) {
		float diff = elements1[i] - elements2[i];
		sum += diff * diff;
	}

	return sqrtf(sum);
}

// computes the cosine distance between two vectors
// distance = 1 - dot(a, b) / (||a|| * ||b||)
float SIVector_CosineDistance
(
	SIValue a,  // first vector
	SIValue b   // second vector
) {
	// cosineDistance(vecf32([0.2, 0.12, 0.3178]), vecf32([0.1, 0.2, 0.3]))
	ASSERT(SI_TYPE(a) & T_VECTOR);
	ASSERT(SI_TYPE(b) & T_VECTOR);

	// validate input vectors are of the same length
	uint32_t n1 = SIVector_Dim(a);
	uint32_t n2 = SIVector_Dim(b);
	ASSERT(n1 == n2);

	// compute the cosine distance between the two vectors
	float dot        = 0;
	float *elements1 = (float*)SIVector_Elements(a);
	float *elements2 = (float*)SIVector_Elements(b);
	float norm_a     = 0;
	float norm_b     = 0;
	for(uint32_t i = 0; i < n1; i++) {
		dot    += elements1[i] * elements2[i];
		norm_a += elements1[i] * elements1[i];
		norm_b += elements2[i] * elements2[i];
	}

	return 1 - dot / (sqrtf(norm_a) * sqrtf(norm_b));
}

// write a string representation of vector to buf
void SIVector_ToString
(
	SIValue vector,       // vector to convert to string
	char **buf,           // output buffer
	size_t *bufferLen,    // output buffer length
	size_t *bytesWritten  // output bytes written
) {
	ASSERT(buf           != NULL);
	ASSERT(bufferLen     != NULL);
	ASSERT(vector.ptrval != NULL);
	ASSERT(bytesWritten  != NULL);
	ASSERT(SI_TYPE(vector) & T_VECTOR);

	// compute required buffer size
	// each element is represented by 24 characters
	uint64_t dim          = SIVector_Dim(vector);
	size_t   availableLen = *bufferLen -  *bytesWritten;
	size_t   requiredLen  = dim * 26 + 3;

	// make sure buffer is large enough
	if(requiredLen > availableLen) {
		*bufferLen += requiredLen - availableLen;
		*buf = rm_realloc(*buf, *bufferLen);
	}

	// write opening bracket
	*bytesWritten += sprintf(*buf + *bytesWritten, "<");

	SIVector *v = (SIVector*)vector.ptrval;

	// write vector's elements
	float *elements = (float*)v->elements;
	for(uint32_t i = 0; i < dim; i++) {
		// get the entry v(i)
		float vi = elements[i];

		// write current element to buffer
		*bytesWritten += sprintf(*buf + *bytesWritten, "%f, ", vi);
	}

	// write closing bracket
	if(dim > 0) {
		// remove trailing comma and space
		*bytesWritten -= 2;
	}
	*bytesWritten += sprintf(*buf + *bytesWritten, ">");
}

void SIVector_Free
(
	SIValue vector // vector to free
) {
	ASSERT(SI_TYPE(vector) & T_VECTOR);
	ASSERT(vector.ptrval != NULL);

	rm_free(vector.ptrval);
}

