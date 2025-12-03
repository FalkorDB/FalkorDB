/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "boolean_funcs.h"
#include "../func_desc.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../errors/errors.h"
#include "../../datatypes/map.h"
#include "../../datatypes/array.h"

#define CONTAINS_NULL 2 // Macro used for efficiently evaluating 3-valued truth table

SIValue AR_AND(SIValue *argv, int argc, void *private_data) {
	// false AND null evaluates to false ; all other null comparisons evaluate to null

	SIValue a = argv[0];
	SIValue b = argv[1];

	if(SIValue_IsNull(a)) {
		// If one argument is null and the other is false, returns false.
		if(SI_TYPE(b) == T_BOOL && b.longval == false) return SI_BoolVal(false);
		// Else if an argument is null, returns null.
		return SI_NullVal();
	} else if(SIValue_IsNull(b)) {
		// If one argument is null and the other is false, returns false.
		if(SI_TYPE(a) == T_BOOL && a.longval == false) return SI_BoolVal(false);
		// Else if an argument is null, returns null.
		return SI_NullVal();
	}

	return SI_BoolVal(a.longval & b.longval); // Return the logical AND.
}

SIValue AR_OR(SIValue *argv, int argc, void *private_data) {
	// true OR null evaluates to true; all other null comparisons evaluate to null

	SIValue a = argv[0];
	SIValue b = argv[1];

	int a_val = SIValue_IsNull(a) ? CONTAINS_NULL : a.longval;
	int b_val = SIValue_IsNull(b) ? CONTAINS_NULL : b.longval;
	int val = a_val | b_val;

	if(val & 1) return SI_BoolVal(true);         // If at least one argument is true, returns true.
	if(val & CONTAINS_NULL) return SI_NullVal(); // Else if an argument is null, returns null.
	return SI_BoolVal(false);                    // If both arguments are false, returns false
}

SIValue AR_XOR(SIValue *argv, int argc, void *private_data) {
	SIValue a = argv[0];
	SIValue b = argv[1];

	if(SIValue_IsNull(a)) return SI_NullVal();
	if(SIValue_IsNull(b)) return SI_NullVal();

	bool res = a.longval != b.longval;
	return SI_BoolVal(res);
}

SIValue AR_NOT(SIValue *argv, int argc, void *private_data) {
	SIValue a = argv[0];
	if(SIValue_IsNull(a)) return SI_NullVal();

	if(SI_TYPE(a) & (SI_NUMERIC | T_BOOL)) return SI_BoolVal(!SI_GET_NUMERIC(a));
	// String, Node, Edge, Ptr all evaluate to true.
	return SI_BoolVal(false);
}

SIValue AR_GT(SIValue *argv, int argc, void *private_data) {
	SIValue a = argv[0];
	SIValue b = argv[1];

	int disjointOrNull = 0;
	int res = SIValue_Compare(a, b, &disjointOrNull);
	if(disjointOrNull == COMPARED_NULL || disjointOrNull == DISJOINT) {
		// comparisons with NULL or different types values always return NULL
		return SI_NullVal();
	}
	if(disjointOrNull == COMPARED_NAN) {
		// comparisons with NaN values always return false
		return SI_BoolVal(false);
	}

	return SI_BoolVal(res > 0);
}

SIValue AR_GE(SIValue *argv, int argc, void *private_data) {
	SIValue a = argv[0];
	SIValue b = argv[1];

	int disjointOrNull = 0;
	int res = SIValue_Compare(a, b, &disjointOrNull);
	if(disjointOrNull == COMPARED_NULL) {
		// Comparisons with NULL values always return NULL.
		return SI_NullVal();
	}
	if(disjointOrNull == COMPARED_NAN) return SI_BoolVal(false);
	if(disjointOrNull == DISJOINT) return SI_NullVal();

	return SI_BoolVal(res >= 0);
}

SIValue AR_LT(SIValue *argv, int argc, void *private_data) {
	SIValue a = argv[0];
	SIValue b = argv[1];

	int disjointOrNull = 0;
	int res = SIValue_Compare(a, b, &disjointOrNull);
	if(disjointOrNull == COMPARED_NULL) {
		// Comparisons with NULL values always return NULL.
		return SI_NullVal();
	}
	if(disjointOrNull == COMPARED_NAN) return SI_BoolVal(false);
	if(disjointOrNull == DISJOINT) return SI_NullVal();

	return SI_BoolVal(res < 0);
}

SIValue AR_LE(SIValue *argv, int argc, void *private_data) {
	SIValue a = argv[0];
	SIValue b = argv[1];

	int disjointOrNull = 0;
	int res = SIValue_Compare(a, b, &disjointOrNull);
	if(disjointOrNull == COMPARED_NULL) {
		// Comparisons with NULL values always return NULL.
		return SI_NullVal();
	}
	if(disjointOrNull == COMPARED_NAN) return SI_BoolVal(false);
	if(disjointOrNull == DISJOINT) return SI_NullVal();

	return SI_BoolVal(res <= 0);
}

SIValue AR_EQ(SIValue *argv, int argc, void *private_data) {
	SIValue a = argv[0];
	SIValue b = argv[1];

	int disjointOrNull = 0;
	int res = SIValue_Compare(a, b, &disjointOrNull);
	if(disjointOrNull == COMPARED_NULL) return SI_NullVal();
	if(disjointOrNull == COMPARED_NAN) return SI_BoolVal(false);
	if(disjointOrNull == DISJOINT) return SI_BoolVal(false);
	// Disjoint comparison is allowed on EQ and NE operators, since they impose no order.
	return SI_BoolVal(res == 0);
}

SIValue AR_NE(SIValue *argv, int argc, void *private_data) {
	SIValue a = argv[0];
	SIValue b = argv[1];

	int disjointOrNull = 0;
	int res = SIValue_Compare(a, b, &disjointOrNull);
	if(disjointOrNull == COMPARED_NULL) return SI_NullVal();
	if(disjointOrNull == COMPARED_NAN) return SI_BoolVal(true);
	if(disjointOrNull == DISJOINT) return SI_BoolVal(true);
	// Disjoint comparison is allowed on EQ and NE operators, since they impose no order.
	return SI_BoolVal(res != 0);
}

// Returns true if argv[0] is null.
SIValue AR_IS_NULL(SIValue *argv, int argc, void *private_data) {
	SIValue v = argv[0];
	return SI_BoolVal(v.type == T_NULL);
}

// Returns true if argv[0] is not null.
SIValue AR_IS_NOT_NULL(SIValue *argv, int argc, void *private_data) {
	SIValue v = argv[0];
	return SI_BoolVal(v.type != T_NULL);
}


// Coverts argv[0] to boolean
// Integer input will return false if the value is 0, true otherwise
// String input will return true if value is "true", false if value is "false" (case insensitive), null otherwise
// Null logic - returns null
SIValue AR_TO_BOOLEAN(SIValue *argv, int argc, void *private_data) {
	SIValue v = argv[0];
	switch (SI_TYPE(v)) {
		case T_INT64:
			return SI_BoolVal(v.longval);
		case T_INTERN_STRING:
		case T_STRING: {
			if(!strcasecmp("true", v.stringval)) return SI_BoolVal(true);
			else if(!strcasecmp("false", v.stringval)) return SI_BoolVal(false);
			return SI_NullVal();
		}
		case T_BOOL:
			return v;
		default:
			return SI_NullVal();
	}
}

SIValue AR_ISEMPTY(SIValue *argv, int argc, void *private_data) {
	ASSERT(argc == 1);
	switch(SI_TYPE(argv[0])) {
		case T_NULL:
			return SI_NullVal();
		case T_ARRAY:
			if(SIArray_Length(argv[0]) == 0) return SI_BoolVal(true);
			break;
		case T_MAP:
			if(array_len(argv[0].map) == 0) return SI_BoolVal(true);
			break;
		case T_STRING:
		case T_INTERN_STRING:
			if(strlen(argv[0].stringval) == 0) return SI_BoolVal(true);
			break;
		default:
			ASSERT(false);
	}
	return SI_BoolVal(false);
}

void Register_BooleanFuncs() {
	SIType *types;
	SIType ret_type = T_BOOL | T_NULL;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 2);
	array_append(types, T_BOOL | T_NULL);
	array_append(types, T_BOOL | T_NULL);
	func_desc = AR_FuncDescNew("and", AR_AND, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, T_BOOL | T_NULL);
	array_append(types, T_BOOL | T_NULL);
	func_desc = AR_FuncDescNew("or", AR_OR, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, T_BOOL | T_NULL);
	array_append(types, T_BOOL | T_NULL);
	func_desc = AR_FuncDescNew("xor", AR_XOR, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_BOOL | T_NULL);
	func_desc = AR_FuncDescNew("not", AR_NOT, 1, 1, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, SI_ALL);
	array_append(types, SI_ALL);
	func_desc = AR_FuncDescNew("gt", AR_GT, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, SI_ALL);
	array_append(types, SI_ALL);
	func_desc = AR_FuncDescNew("ge", AR_GE, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, SI_ALL);
	array_append(types, SI_ALL);
	func_desc = AR_FuncDescNew("lt", AR_LT, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, SI_ALL);
	array_append(types, SI_ALL);
	func_desc = AR_FuncDescNew("le", AR_LE, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, SI_ALL);
	array_append(types, SI_ALL);
	func_desc = AR_FuncDescNew("eq", AR_EQ, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, SI_ALL);
	array_append(types, SI_ALL);
	func_desc = AR_FuncDescNew("neq", AR_NE, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, SI_ALL);
	func_desc = AR_FuncDescNew("is null", AR_IS_NULL, 1, 1, types, ret_type,
			true, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, SI_ALL);
	func_desc = AR_FuncDescNew("is not null", AR_IS_NOT_NULL, 1, 1, types,
			ret_type, true, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_BOOL | T_INT64 | T_STRING | T_NULL);
	func_desc = AR_FuncDescNew("toBoolean", AR_TO_BOOLEAN, 1, 1, types,
			ret_type, false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, SI_ALL);
	func_desc = AR_FuncDescNew("toBooleanOrNull", AR_TO_BOOLEAN, 1, 1, types,
			ret_type, false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, T_ARRAY | T_MAP | T_NULL | T_STRING);
	ret_type = T_BOOL | T_NULL;
	func_desc = AR_FuncDescNew("isempty", AR_ISEMPTY, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);
}

