/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../value.h"
#include "effects.h"

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

