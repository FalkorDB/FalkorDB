/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../value.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../errors/errors.h"
#include "../../datatypes/array.h"
#include "../../datatypes/vector.h"

// create a vector from an array of floats
// vecf32([0.2, 0.12, 0.3178])
SIValue AR_VECTOR32F
(
	SIValue *argv,      // arguments
	int argc,           // number of arguments
	void *private_data  // private data
) {
	// expecting input to be an array of 32 bit floats
	// vector32f([0.2, 0.12, 0.3178])

	SIValue arr = argv[0];

	// return NULL if input is NULL
	SIType t = SI_TYPE(arr);
	if(t == T_NULL) {
		return SI_NullVal();
	}

	ASSERT(t == T_ARRAY);

	// validate input contains only floats
	if(SIArray_AllOfType(arr, SI_NUMERIC) == false) {
		ErrorCtx_SetError(EMSG_VECTOR_TYPE_ERROR, 32);
		return SI_NullVal();
	}

	// create a vector of the same length as the input array
	uint32_t n         = SIArray_Length(arr);
	SIValue  v         = SI_Vectorf32(n);
	float    *elements = (float*)SIVector_Elements(v);

	// save each float into the vector's internal values array
	for(uint32_t i = 0; i < n; i++) {
		// save value as float
		elements[i] = (float)SI_GET_NUMERIC(SIArray_Get(arr, i));
	}

	return v;
}

// compute the euclidean distance between two vectors
SIValue AR_EUCLIDEAN_DISTANCE
(
	SIValue *argv,      // arguments
	int argc,           // number of arguments
	void *private_data  // private data
) {
	// expecting input to be two vectors
	// euclideanDistance(vecf32([0.2, 0.12, 0.3178]), vecf32([0.1, 0.2, 0.3]))

	SIValue v1 = argv[0];
	SIValue v2 = argv[1];

	// return NULL if input is NULL
	SIType t1 = SI_TYPE(v1);
	SIType t2 = SI_TYPE(v2);
	if(t1 == T_NULL || t2 == T_NULL) {
		return SI_NullVal();
	}

	ASSERT(t1 == T_VECTOR);
	ASSERT(t2 == T_VECTOR);

	// validate input vectors are of the same length
	uint32_t n1 = SIVector_Dim(v1);
	uint32_t n2 = SIVector_Dim(v2);
	if(n1 != n2) {
		ErrorCtx_SetError(EMSG_VECTOR_DIMENSION_MISMATCH, n1, n2);
		return SI_NullVal();
	}

	// computes the euclidean distance between two vectors
	float distance = SIVector_EuclideanDistance(v1, v2);
	return SI_DoubleVal(distance);
}

// compute the cosine distance between two vectors
SIValue AR_COSINE_DISTANCE
(
	SIValue *argv,      // arguments
	int argc,           // number of arguments
	void *private_data  // private data
) {
	// expecting input to be two vectors
	// cosineDistance(vecf32([0.2, 0.12, 0.3178]), vecf32([0.1, 0.2, 0.3]))

	SIValue v1 = argv[0];
	SIValue v2 = argv[1];

	// return NULL if input is NULL
	SIType t1 = SI_TYPE(v1);
	SIType t2 = SI_TYPE(v2);
	if(t1 == T_NULL || t2 == T_NULL) {
		return SI_NullVal();
	}

	ASSERT(t1 == T_VECTOR);
	ASSERT(t2 == T_VECTOR);

	// validate input vectors are of the same length
	uint32_t n1 = SIVector_Dim(v1);
	uint32_t n2 = SIVector_Dim(v2);
	if(n1 != n2) {
		ErrorCtx_SetError(EMSG_VECTOR_DIMENSION_MISMATCH, n1, n2);
		return SI_NullVal();
	}

	// computes the cosine distance between two vectors
	float distance = SIVector_CosineDistance(v1, v2);
	return SI_DoubleVal(distance);
}

void Register_VectorFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	// create a vector from array
	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_ARRAY);
	ret_type = T_NULL | T_VECTOR;
	func_desc = AR_FuncDescNew("vecf32", AR_VECTOR32F, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	// euclidean distance between two vectors
	types = array_new(SIType, 2);
	array_append(types, T_NULL | T_VECTOR);
	array_append(types, T_NULL | T_VECTOR);
	ret_type = T_NULL | T_DOUBLE;
	func_desc = AR_FuncDescNew("vec.euclideanDistance", AR_EUCLIDEAN_DISTANCE, 2, 2, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	// cosine distance between two vectors
	types = array_new(SIType, 2);
	array_append(types, T_NULL | T_VECTOR);
	array_append(types, T_NULL | T_VECTOR);
	ret_type = T_NULL | T_DOUBLE;
	func_desc = AR_FuncDescNew("vec.cosineDistance", AR_COSINE_DISTANCE, 2, 2, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);
}

