/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../value.h"

// creates an array from its binary representation
// this is the reverse of SIArray_ToBinary
// x = SIArray_FromBinary(SIArray_ToBinary(y));
// x == y
static SIValue SIArray_FromBinary
(
	FILE *stream  // stream containing binary representation of an array
) {
	// read number of elements
	uint32_t n;
	fread_assert(&n, sizeof(uint32_t), stream);

	SIValue arr = SI_Array(n);

	for(uint32_t i = 0; i < n; i++) {
		array_append(arr.array, SIValue_FromBinary(stream));
	}

	return arr;
}

// creates a vector from its binary representation
static SIValue SIVector_FromBinary
(
	FILE *stream, // binary stream
	SIType t      // vector type
) {
	// format:
	// number of elements
	// elements

	ASSERT(stream != NULL);
	ASSERT(t & T_VECTOR);

	// read vector dimension from stream
	uint32_t dim;
	fread_assert(&dim, sizeof(uint32_t), stream);

	// create vector
	SIValue v = SIVectorf32_New(dim);
	size_t elem_size = sizeof(float);

	// set vector's elements
	if(dim > 0) {
		fread_assert(SIVector_Elements(v), dim * elem_size, stream);
	}

	return v;
}

// create map from binary stream
static SIValue Map_FromBinary
(
	FILE *stream  // binary stream
) {
	// format:
	// key count
	// key:value

	ASSERT(stream != NULL);

	// read number of keys in map
	uint32_t n;
	fread_assert(&n, sizeof(uint32_t), stream);

	SIValue map = Map_New(n);

	for(uint32_t i = 0; i < n; i++) {
		// read string length from stream
		size_t len;
		fread_assert(&len, sizeof(len), stream);

		// read string from stream
		char *key = rm_malloc(sizeof(char) * len);
		fread_assert(key, sizeof(char) * len, stream);

		Pair p = {
			.key = key,
			.val = SIValue_FromBinary(stream)
		};

		array_append(map.map, p);
	}

	return map;
}

// reads an SIValue from a its binary representation
SIValue EffectsBuffer_ReadSIValue
(
    FILE *stream  // stream to read value from
) {
	ASSERT(stream != NULL);

	// read value type
	SIType t;
	SIValue v;
	size_t len;  // string length

	bool     b;
	int64_t  i;
	double   d;
	Point    p;
	char    *s;
	struct SIValue *array;

	fread_assert(&t, sizeof(SIType), stream);
	switch(t) {
		case T_POINT:
			// read point from stream
			fread_assert(&p, sizeof(v.point), stream);
			v = SI_Point(p.latitude, p.longitude);
			break;
		case T_ARRAY:
			// read array from stream
			v = SIArray_FromBinary(stream);
			break;
		case T_STRING:
			// read string length from stream
			fread_assert(&len, sizeof(len), stream);
			s = rm_malloc(sizeof(char) * len);
			// read string from stream
			fread_assert(s, sizeof(char) * len, stream);
			v = SI_TransferStringVal(s);
			break;
		case T_BOOL:
			// read bool from stream
			fread_assert(&b, sizeof(b), stream);
			v = SI_BoolVal(b);
			break;
		case T_INT64:
			// read int from stream
			fread_assert(&i, sizeof(i), stream);
			v = SI_LongVal(i);
			break;
		case T_DOUBLE:
			// read double from stream
			fread_assert(&d, sizeof(d), stream);
			v = SI_DoubleVal(d);
			break;
		case T_VECTOR_F32:
			v = SIVector_FromBinary(stream, t);
			break;
		case T_MAP:
			v = Map_FromBinary(stream);
			break;
		case T_NULL:
			v = SI_NullVal();
			break;
		default:
			assert(false && "unknown SIValue type");
	}

	return v;
}

