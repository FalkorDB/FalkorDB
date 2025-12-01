/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../func_desc.h"
#include "numeric_funcs.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"
#include "../../errors/errors.h"
#include "../temporal_arithmetic/temporal_arithmetic.h"

#include <math.h>
#include <errno.h>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif /* M_PI */

// the '+' operator is overloaded to perform string concatenation
// as well as arithmetic addition
SIValue AR_ADD
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	return SIValue_Add(argv[0], argv[1]);
}

// returns the subtracting given values
SIValue AR_SUB
(
	SIValue *argv,
	int argc,
	void *private_data
) {
	return SIValue_Subtract(argv[0], argv[1]);
}

/* returns the multiplication of given values. */
SIValue AR_MUL(SIValue *argv, int argc, void *private_data) {
	if(SIValue_IsNull(argv[0]) || SIValue_IsNull(argv[1])) return SI_NullVal();
	return SIValue_Multiply(argv[0], argv[1]);
}

/* returns the division of given values. */
SIValue AR_DIV(SIValue *argv, int argc, void *private_data) {
	if(SIValue_IsNull(argv[0]) || SIValue_IsNull(argv[1])) return SI_NullVal();

	// check for division by zero e.g. n / 0
	if((SI_TYPE(argv[0]) & SI_TYPE(argv[1]) & T_INT64) && SI_GET_NUMERIC(argv[1]) == 0) {
		Error_DivisionByZero();
		return SI_NullVal();
	}
	return SIValue_Divide(argv[0], argv[1]);
}

SIValue AR_MODULO(SIValue *argv, int argc, void *private_data) {
	if(SIValue_IsNull(argv[0]) || SIValue_IsNull(argv[1])) return SI_NullVal();

	// check for modulo by zero e.g. n % 0
	if((SI_TYPE(argv[0]) & SI_TYPE(argv[1]) & T_INT64) && SI_GET_NUMERIC(argv[1]) == 0) {
		Error_DivisionByZero();
		return SI_NullVal();
	}
	return SIValue_Modulo(argv[0], argv[1]);
}

/* TODO All AR_* functions need to emit appropriate failures when provided
 * with arguments of invalid types and handle multiple arguments. */
/* returns the absolute value of the given number. */
SIValue AR_ABS(SIValue *argv, int argc, void *private_data) {
	SIValue result = argv[0];
	if(SIValue_IsNull(result)) return SI_NullVal();
	if(SI_GET_NUMERIC(argv[0]) < 0) return SIValue_Multiply(argv[0], SI_LongVal(-1));
	return argv[0];
}

/* returns the smallest floating point number that is greater than or equal to the given number and equal to a mathematical integer. */
SIValue AR_CEIL(SIValue *argv, int argc, void *private_data) {
	SIValue result = argv[0];
	if(SIValue_IsNull(result)) return SI_NullVal();
	// No modification is required for non-decimal values
	if(SI_TYPE(result) == T_DOUBLE) result.doubleval = ceil(result.doubleval);

	return result;
}

/* returns the largest floating point number that is less than or equal to the given number and equal to a mathematical integer. */
SIValue AR_FLOOR(SIValue *argv, int argc, void *private_data) {
	SIValue result = argv[0];
	if(SIValue_IsNull(result)) return SI_NullVal();
	// No modification is required for non-decimal values
	if(SI_TYPE(result) == T_DOUBLE) result.doubleval = floor(result.doubleval);

	return result;
}

/* returns a random floating point number in the range from 0 to 1; i.e. [0,1]. The numbers returned follow an approximate uniform distribution. */
SIValue AR_RAND(SIValue *argv, int argc, void *private_data) {
	return SI_DoubleVal((double)rand() / (double)RAND_MAX);
}

/* returns the value of the given number rounded to the nearest integer. */
SIValue AR_ROUND(SIValue *argv, int argc, void *private_data) {
	SIValue result = argv[0];
	if(SIValue_IsNull(result)) return SI_NullVal();
	// No modification is required for non-decimal values
	if(SI_TYPE(result) == T_DOUBLE) result.doubleval = round(result.doubleval);

	return result;
}

/* returns the signum of the given number: 0 if the number is 0, -1 for any negative number, and 1 for any positive number. */
SIValue AR_SIGN(SIValue *argv, int argc, void *private_data) {
	if(SIValue_IsNull(argv[0])) return SI_NullVal();
	int64_t sign = SIGN(SI_GET_NUMERIC(argv[0]));
	return SI_LongVal(sign);
}

// tries to convert input to integer
SIValue AR_TOINTEGER(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	char *sEnd = NULL;

	switch(SI_TYPE(arg)) {
	case T_INT64:
		return arg;
	case T_DOUBLE:
		// Remove floating point.
		return SI_LongVal(floor(arg.doubleval));
	case T_BOOL:
		if(SIValue_IsTrue(arg)) {
			return SI_LongVal(1);
		}
		return SI_LongVal(0);
	case T_STRING:
	case T_INTERN_STRING:
		if(strlen(arg.stringval) == 0) return SI_NullVal();
		errno = 0;
		if(strchr(arg.stringval, '.') == NULL) {
			int64_t parsedval = strtoll(arg.stringval, &sEnd, 10);
			if(sEnd[0] != '\0' || errno == ERANGE) return SI_NullVal();
			return SI_LongVal(parsedval);
		}
		double parsedval = strtod(arg.stringval, &sEnd);
		/* The input was not a complete number or represented a number that
		 * cannot be represented as a double. */
		if(sEnd[0] != '\0' || errno == ERANGE) return SI_NullVal();
		// Remove floating point.
		return SI_LongVal(floor(parsedval));
	default:
		return SI_NullVal();
	}
}

// tries to convert input to float
SIValue AR_TOFLOAT(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	char *sEnd = NULL;

	switch(SI_TYPE(arg)) {
	case T_INT64:
		return SI_DoubleVal(arg.longval);
	case T_DOUBLE:
		return arg;
	case T_STRING:
	case T_INTERN_STRING:
		if(strlen(arg.stringval) == 0) return SI_NullVal();
		errno = 0;
		double parsedval = strtof(arg.stringval, &sEnd);
		/* The input was not a complete number or represented a number that
		 * cannot be represented as a double. */
		if(sEnd[0] != '\0' || errno == ERANGE) return SI_NullVal();
		return SI_DoubleVal(parsedval);
	default:
		return SI_NullVal();
	}
}

// returns the square root of a number
SIValue AR_SQRT(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	// return NULL if input is none numeric
	if(SIValue_IsNull(arg)) return SI_NullVal();

	double value = SI_GET_NUMERIC(arg);

	// return sqrt of input
	return SI_DoubleVal(sqrt(value));
}

// returns base^exponent
SIValue AR_POW(SIValue *argv, int argc, void *private_data) {
	SIValue base = argv[0];
	SIValue exp = argv[1];
	// return NULL if input is NULL
	if(SIValue_IsNull(base) || SIValue_IsNull(exp)) return SI_NullVal();

	// return base^exponent
	return SI_DoubleVal(pow(SI_GET_NUMERIC(base), SI_GET_NUMERIC(exp)));
}

// returns the base of the natural logarithm, e.
SIValue AR_E(SIValue *argv, int argc, void *private_data) {
	// base of the natural logarithm (e) == e^1
	return SI_DoubleVal(exp(1));
}

// returns e^value
SIValue AR_EXP(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	// return NULL if input is none numeric
	if(SIValue_IsNull(arg)) return SI_NullVal();

	double value = SI_GET_NUMERIC(arg);

	// return e^value
	return SI_DoubleVal(exp(value));
}

// returns ln(val)
SIValue AR_LOG(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	// return NULL if input is none numeric
	if(SIValue_IsNull(arg)) return SI_NullVal();

	double value = SI_GET_NUMERIC(arg);

	// return ln of input
	return SI_DoubleVal(log(value));
}

// returns log10(val)
SIValue AR_LOG10(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	// return NULL if input is none numeric
	if(SIValue_IsNull(arg)) return SI_NullVal();

	double value = SI_GET_NUMERIC(arg);

	// return log10 of input
	return SI_DoubleVal(log10(value));
}

// returns sin(x)
SIValue AR_SIN(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	// return NULL if input is none numeric
	if(SIValue_IsNull(arg)) return SI_NullVal();

	double value = SI_GET_NUMERIC(arg);

	// return sin(x) of input
	return SI_DoubleVal(sin(value));
}

// returns cos(x)
SIValue AR_COS(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	// return NULL if input is none numeric
	if(SIValue_IsNull(arg)) return SI_NullVal();

	double value = SI_GET_NUMERIC(arg);

	// return cos(x) of input
	return SI_DoubleVal(cos(value));
}

// returns tan(x)
SIValue AR_TAN(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	// return NULL if input is none numeric
	if(SIValue_IsNull(arg)) return SI_NullVal();

	double value = SI_GET_NUMERIC(arg);

	// return tan(x) of input
	return SI_DoubleVal(tan(value));
}

// returns cot(x) = cos(x)/sin(x)
SIValue AR_COT(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	// return NULL if input is none numeric
	if(SIValue_IsNull(arg)) return SI_NullVal();
	
	double value = SI_GET_NUMERIC(arg);

	// return cot(x) of input
	return SI_DoubleVal(cos(value)/sin(value));
}

// returns asin(x)
SIValue AR_ASIN(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	// return NULL if input is none numeric
	if(SIValue_IsNull(arg)) return SI_NullVal();
	
	double value = SI_GET_NUMERIC(arg);
	
	// return asin(x) of input
	return SI_DoubleVal(asin(value));
}

// returns acos(x)
SIValue AR_ACOS(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	// return NULL if input is none numeric
	if(SIValue_IsNull(arg)) return SI_NullVal();
	
	double value = SI_GET_NUMERIC(arg);

	// return acos(x) of input
	return SI_DoubleVal(acos(value));
}

// returns atan(x)
SIValue AR_ATAN(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];
	// return NULL if input is none numeric
	if(SIValue_IsNull(arg)) return SI_NullVal();
	
	double value = SI_GET_NUMERIC(arg);

	// return atan(x) of input
	return SI_DoubleVal(atan(value));
}

// returns atan2(y, x)
SIValue AR_ATAN2(SIValue *argv, int argc, void *private_data) {
	SIValue argY = argv[0];
	SIValue argX = argv[1];

	// return NULL if input is NULL
	if(SIValue_IsNull(argY) || SIValue_IsNull(argX)) return SI_NullVal();
	
	double y = SI_GET_NUMERIC(argY);
	double x = SI_GET_NUMERIC(argX);
	
	// return atan2(y, x) of input
	return SI_DoubleVal(atan2(y, x));
}

// returns degrees(x)
SIValue AR_DEGREES(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];

	// return NULL if input is NULL
	if(SIValue_IsNull(arg)) return SI_NullVal();
	
	double value = SI_GET_NUMERIC(arg);
	
	// return degrees(value) of input
	return SI_DoubleVal(value * (180/M_PI));
}

// returns radians(x)
SIValue AR_RADIANS(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];

	// return NULL if input is NULL
	if(SIValue_IsNull(arg)) return SI_NullVal();
	
	double value = SI_GET_NUMERIC(arg);
	
	// return radians(value) of input
	return SI_DoubleVal( value * (M_PI / 180.0));
}

// returns radians(x)
SIValue AR_PI(SIValue *argv, int argc, void *private_data) {	
	// return pi() of input
	return SI_DoubleVal(M_PI);
}

// returns haversin(x)
SIValue AR_HAVERSIN(SIValue *argv, int argc, void *private_data) {
	SIValue arg = argv[0];

	// return NULL if input is NULL
	if(SIValue_IsNull(arg)) return SI_NullVal();
	
	double value = SI_GET_NUMERIC(arg);
	
	// return haversin(y, x) of input
	return SI_DoubleVal((1-cos(value))/2);
}

void Register_NumericFuncs() {
	SIType *types;
	SIType ret_type;
	AR_FuncDesc *func_desc;

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_STRING | T_ARRAY | T_BOOL | T_MAP | SI_TEMPORAL | T_NULL));
	array_append(types, (SI_NUMERIC | T_STRING | T_ARRAY | T_BOOL | T_MAP | SI_TEMPORAL | T_NULL));
	ret_type = SI_NUMERIC | T_STRING | T_ARRAY | T_BOOL | T_MAP | SI_TEMPORAL | T_NULL;
	func_desc = AR_FuncDescNew("add", AR_ADD, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | SI_TEMPORAL | T_NULL));
	array_append(types, (SI_NUMERIC | T_DURATION  | T_NULL));
	ret_type = SI_NUMERIC | SI_TEMPORAL | T_NULL;
	func_desc = AR_FuncDescNew("sub", AR_SUB, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = SI_NUMERIC | T_NULL;
	func_desc = AR_FuncDescNew("mul", AR_MUL, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = SI_NUMERIC | T_NULL;
	func_desc = AR_FuncDescNew("div", AR_DIV, 2, 2, types, ret_type, true, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = SI_NUMERIC | T_NULL;
	func_desc = AR_FuncDescNew("mod", AR_MODULO, 2, 2, types, ret_type, true,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = SI_NUMERIC | T_NULL;
	func_desc = AR_FuncDescNew("abs", AR_ABS, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = SI_NUMERIC | T_NULL;
	func_desc = AR_FuncDescNew("ceil", AR_CEIL, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = SI_NUMERIC | T_NULL;
	func_desc = AR_FuncDescNew("floor", AR_FLOOR, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 0);
	ret_type = T_DOUBLE;
	func_desc = AR_FuncDescNew("rand", AR_RAND, 0, 0, types, ret_type, false,
			false, false);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = SI_NUMERIC | T_NULL;
	func_desc = AR_FuncDescNew("round", AR_ROUND, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_INT64 | T_NULL;
	func_desc = AR_FuncDescNew("sign", AR_SIGN, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_STRING | T_NULL | T_BOOL));
	ret_type = T_INT64 | T_NULL;
	func_desc = AR_FuncDescNew("tointeger", AR_TOINTEGER, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);
	
	types = array_new(SIType, 1);
	array_append(types, SI_ALL);
	ret_type = T_INT64 | T_NULL;
	func_desc = AR_FuncDescNew("tointegerornull", AR_TOINTEGER, 1, 1, types,
			ret_type, false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_STRING | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("tofloat", AR_TOFLOAT, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, SI_ALL);
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("tofloatornull", AR_TOFLOAT, 1, 1, types,
			ret_type, false, true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("sqrt", AR_SQRT, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("pow", AR_POW, 2, 2, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("exp", AR_EXP, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 0);
	ret_type = T_DOUBLE;
	func_desc = AR_FuncDescNew("e", AR_E, 0, 0, types, ret_type, false, true,
			true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("log", AR_LOG, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("log10", AR_LOG10, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("sin", AR_SIN, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("cos", AR_COS, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("tan", AR_TAN, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("cot", AR_COT, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("asin", AR_ASIN, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("acos", AR_ACOS, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("atan", AR_ATAN, 1, 1, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 2);
	array_append(types, (SI_NUMERIC | T_NULL));
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("atan2", AR_ATAN2, 2, 2, types, ret_type, false,
			true, true);
	AR_RegFunc(func_desc);

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("degrees", AR_DEGREES, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);	

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("radians", AR_RADIANS, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);	

	types = array_new(SIType, 0);
	ret_type = T_DOUBLE;
	func_desc = AR_FuncDescNew("pi", AR_PI, 0, 0, types, ret_type, false, true,
			true);
	AR_RegFunc(func_desc);	

	types = array_new(SIType, 1);
	array_append(types, (SI_NUMERIC | T_NULL));
	ret_type = T_DOUBLE | T_NULL;
	func_desc = AR_FuncDescNew("haversin", AR_HAVERSIN, 1, 1, types, ret_type,
			false, true, true);
	AR_RegFunc(func_desc);
}

