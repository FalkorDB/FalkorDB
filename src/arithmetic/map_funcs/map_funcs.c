/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "map_funcs.h"
#include "RG.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../datatypes/map.h"
#include "../../errors/errors.h"
#include "../../datatypes/array.h"
#include "../../graph/entities/graph_entity.h"

// create a new SIMap object
// expecting an even number of arguments
// argv[even] = key
// argv[odd]  = value
SIValue AR_TOMAP
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	// validate number of arguments
	if(argc % 2 != 0) {
		ErrorCtx_RaiseRuntimeException("map expects even number of elements");
	}

	SIValue map = SI_Map(argc / 2);

	for(int i = 0; i < argc; i += 2) {
		SIValue key = argv[i];
		SIValue val = argv[i + 1];

		// make sure key is a string
		if(!(SI_TYPE(key) & T_STRING)) {
			Error_SITypeMismatch(key, T_STRING);
			break;
		}

		Map_Add(&map, key, val);
	}

	return map;
}

SIValue AR_KEYS(SIValue *argv, int argc, void *private_data) {
	ASSERT(argc == 1);
	switch(SI_TYPE(argv[0])) {
		case T_NULL:
			return SI_NullVal();
		case T_NODE:
		case T_EDGE:
			return GraphEntity_Keys(argv[0].ptrval);
		case T_MAP:
			return Map_Keys(argv[0]);
		default:
			ASSERT(false);
	}
	return SI_NullVal();
}

SIValue AR_PROPERTIES
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	ASSERT(argc == 1);

	switch(SI_TYPE(argv[0])) {
		case T_NULL:
			return SI_NullVal();
		case T_NODE:
		case T_EDGE:
			return GraphEntity_Properties(argv[0].ptrval);
		case T_MAP:
			return SI_CloneValue(argv[0]);  // clone the map
		default:
			ASSERT(false);
	}
	return SI_NullVal();
}

// receives two maps and merges them
SIValue AR_MERGEMAP
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	ASSERT(argc == 2);

	SIValue map0 = argv[0];
	SIValue map1 = argv[1];

	// both map0 and map1 are NULL then return NULL
	if(SIValue_IsNull(map0) && SIValue_IsNull(map1)) {
		return SI_NullVal();
	} else if (SIValue_IsNull(map0)) {
		// only map0 is null, return a clone of map1
		return Map_Clone(map1);
	} else if (SI_TYPE(map1) & T_NULL) {
		// only map1 is null, return a clone of map0
		return Map_Clone(map0);
	}

	//--------------------------------------------------------------------------
	// merge maps
	//--------------------------------------------------------------------------

	// clone map1
	SIValue map = Map_Clone(map1);

	// add each key in map0 to the new clone
	uint n = Map_KeyCount(map0);

	for(int i = 0; i < n; i++) {
		SIValue key;
		SIValue value;

		Map_GetIdx(map0, i, &key, &value);

		// clones the key & value
		Map_Add(&map, key, value);
	}

	return map;
}

void Register_MapFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 1);
	array_append(types, SI_ALL);
	ret_type = T_MAP;
	func_desc = AR_FuncDescNew("tomap", AR_TOMAP, 0, VAR_ARG_LEN, types,
			ret_type, true, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_MAP | T_NODE | T_EDGE);
	ret_type = T_NULL | T_ARRAY;
	func_desc = AR_FuncDescNew("keys", AR_KEYS, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_NULL | T_MAP | T_NODE | T_EDGE);
	ret_type = T_NULL | T_MAP;
	func_desc = AR_FuncDescNew("properties", AR_PROPERTIES, 1, 1, types,
			ret_type, false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, T_NULL | T_MAP);
	array_append(types, T_NULL | T_MAP);
	ret_type = T_NULL | T_MAP;
	func_desc = AR_FuncDescNew("merge_maps", AR_MERGEMAP, 2, 2, types, ret_type,
			true, true, true);
	AR_RegFunc(func_desc);
}

