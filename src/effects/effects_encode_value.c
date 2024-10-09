/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// forward declarations

// writes a binary representation of v into Effect-Buffer
static void EffectsBuffer_WriteSIValue
(
	const SIValue *v,
	EffectsBuffer *buff
);

// writes a binary representation of arr into Effect-Buffer
static void EffectsBuffer_WriteSIArray
(
	const SIValue *arr,  // array
	EffectsBuffer *buff  // effect buffer
) {
	// format:
	// number of elements
	// elements

	SIValue *elements = arr->array;
	uint32_t len = array_len(elements);

	// write number of elements
	EffectsBuffer_WriteBytes(&len, sizeof(uint32_t), buff);

	// write each element
	for (uint32_t i = 0; i < len; i++) {
		EffectsBuffer_WriteSIValue(elements + i, buff);
	}
}

// write vector to effects buffer
static void EffectsBuffer_WriteSIVector
(
	const SIValue *v,    // vector
	EffectsBuffer *buff  // effect buffer
) {
	// format:
	// number of elements
	// elements

	// write vector dimension
	uint32_t dim = SIVector_Dim(*v);
	EffectsBuffer_WriteBytes(&dim, sizeof(uint32_t), buff);

	// write vector elements
	void *elements   = SIVector_Elements(*v);
	size_t elem_size = sizeof(float);
	size_t n = dim * elem_size;

	if(n > 0) {
		EffectsBuffer_WriteBytes(elements, n, buff);
	}
}

// writes a binary representation of map into Effect-Buffer
static void EffectsBuffer_WriteMap
(
	const SIValue *map,  // map
	EffectsBuffer *buff  // effects buffer
) {
	// format:
	// key count
	// key:value

	// write number of keys in map
	uint32_t n = Map_KeyCount(*map);
	EffectsBuffer_WriteBytes(&n, sizeof(uint32_t), buff);

	// write each key:value pair to buffer
	for(uint32_t i = 0; i < n; i++) {
		SIValue value;
		const char *key;
		Map_GetIdx(*map, i, &key, &value);

		ASSERT(key != NULL);
		EffectsBuffer_WriteString(key, buff);
		EffectsBuffer_WriteSIValue(&value, buff);
	}
}

// writes a binary representation of v into Effect-Buffer
static void EffectsBuffer_WriteSIValue
(
	const SIValue *v,
	EffectsBuffer *buff
) {
	ASSERT(v != NULL);
	ASSERT(buff != NULL);

	// format:
	//    type
	//    value
	bool b;
	size_t len = 0;

	SIType t = v->type;

	// write type
	EffectsBuffer_WriteBytes(&t, sizeof(SIType), buff);

	// write value
	switch(t) {
		case T_POINT:
			// write value to stream
			EffectsBuffer_WriteBytes(&v->point, sizeof(Point), buff);
			break;
		case T_ARRAY:
			// write array to stream
			EffectsBuffer_WriteSIArray(v, buff);
			break;
		case T_STRING:
			EffectsBuffer_WriteString(v->stringval, buff);
			break;
		case T_BOOL:
			// write bool to stream
			b = SIValue_IsTrue(*v);
			EffectsBuffer_WriteBytes(&b, sizeof(bool), buff);
			break;
		case T_INT64:
			// write int to stream
			EffectsBuffer_WriteBytes(&v->longval, sizeof(v->longval), buff);
			break;
		case T_DOUBLE:
			// write double to stream
			EffectsBuffer_WriteBytes(&v->doubleval, sizeof(v->doubleval), buff);
			break;
		case T_NULL:
			// no additional data is required to represent NULL
			break;
		case T_VECTOR_F32:
			EffectsBuffer_WriteSIVector(v, buff);
			break;
		case T_MAP:
			EffectsBuffer_WriteMap(v, buff);
			break;
		default:
			assert(false && "unknown SIValue type");
	}
}

