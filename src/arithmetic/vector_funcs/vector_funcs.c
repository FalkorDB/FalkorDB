/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../../value.h"
#include "../func_desc.h"
#include "../../errors.h"
#include "../../util/arr.h"
#include "../../datatypes/array.h"

// create a vector from an array of floats
// vector32f([0.2, 0.12, 0.3178])
SIValue AR_VECTOR32F
(
	SIValue *argv,      // arguments
	int argc,           // number of arguments
	void *private_data  // private data
) {
	// expecting input to be an array of 32 bit floats
	// vector32f([0.2, 0.12, 0.3178])

	SIValue elements = argv[0];

	// return NULL if input is NULL
	if(SI_TYPE(elements) == T_NULL) {
		return SI_NullVal();
	}

	ASSERT(t == T_ARRAY);

	// validate input contains only floats
	// save each float into the vector's internal values array
	uint32_t n = SIArray_Length(elements);
	float *values = rm_malloc(sizeof(float) * n);

	for(uint32_t i = 0; i < n; i++) {
		SIValue v = SIArray_Get(elements, i);

		// encountered a non-double value
		if(SI_TYPE(v) != T_DOUBLE) {
			rm_free(values);
			ErrorCtx_RaiseRuntimeException("vector32f expects an array of doubles");
			return SI_NullVal();
		}

		// save double value as float
		values[i] = (float)v.doubleval;
	}

	// all elements are doubles
	// create vector
	GrB_Info   info;
	GrB_Vector v;

	info = GrB_Vector_new(&v, GrB_FP32, n);
	ASSERT(info == GrB_SUCCESS);

	// set vector sparsity to FULL
	info = GxB_Vector_Option_set(v, GxB_SPARSITY_CONTROL, GxB_FULL);
	ASSERT(info == GrB_SUCCESS);

	// populate vector from array
	info = GxB_Vector_pack_Full(v, (void**)&values, sizeof(float) * n, false, NULL);
	ASSERT(info == GrB_SUCCESS);

	return SI_Vector(v);
}

void Register_VectorFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_ARRAY);
	ret_type = T_NULL | T_VECTOR;
	func_desc = AR_FuncDescNew("vector32f", AR_VECTOR32F, 1, 1, types, ret_type, false, true);
	AR_RegFunc(func_desc);

	//types = array_new(SIType, 1);
	//array_append(types, T_NULL | T_ARRAY);
	//ret_type = T_NULL | T_VECTOR;
	//func_desc = AR_FuncDescNew("vector64f", AR_DISTANCE64F, 1, 1, types, ret_type, false, true);
	//AR_RegFunc(func_desc);
}
