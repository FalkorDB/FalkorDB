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
// vector32f([0.2, 0.12, 0.3178])
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
		ErrorCtx_RaiseRuntimeException(EMSG_VECTOR_TYPE_ERROR, 32);
		return SI_NullVal();
	}

	// create a vector of the same length as the input array
	uint32_t n         = SIArray_Length(arr);
	SIValue  v         = SI_Vector32f(n);
	float    *elements = (float*)SIVector_Elements(v);

	// save each float into the vector's internal values array
	for(uint32_t i = 0; i < n; i++) {
		// save value as float
		elements[i] = (float)SI_GET_NUMERIC(SIArray_Get(arr, i));
	}

	return v;
}

void Register_VectorFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_ARRAY);
	ret_type = T_NULL | T_VECTOR;
	func_desc = AR_FuncDescNew("vector32f", AR_VECTOR32F, 1, 1, types, ret_type,
			false, true);
	AR_RegFunc(func_desc);
}

