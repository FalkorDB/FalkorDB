/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/graph/entities/attribute_set.h"

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

// test access to an empty (NULL) attribute-set
void test_nullAttributeSet() {
	SIValue v;
	AttributeID attr_id;
	AttributeSet set = NULL;

	// NULL attribute-set isn't READONLY
	bool read_only = ATTRIBUTE_SET_IS_READONLY(set);
	TEST_ASSERT(read_only  == false);

	// NULL attribute-set doesn't contains any attributes
	uint16_t attr_count = AttributeSet_Count(set);
	TEST_ASSERT(attr_count == 0);

	// NULL attribute-set doesn't contains any attributes
	bool attr_found = AttributeSet_Get(set, 0, &v);
	TEST_ASSERT(attr_found        == false);
	TEST_ASSERT(SIValue_IsNull(v) == true);

	attr_found = AttributeSet_GetIdx(set, 1, &attr_id, &v);
	TEST_ASSERT(attr_found        == false);
	TEST_ASSERT(SIValue_IsNull(v) == true);

	attr_found = AttributeSet_Contains(set, 0);
	TEST_ASSERT(attr_found == false);

	char *attributes = AttributeSet_Attributes(set);
	TEST_ASSERT(attributes == NULL);
}

// test adding attributes to set
void test_addAttributeSet() {
	AttributeSet set = NULL;

	// add the first attribute:
	// attribute ID 0
	// attribute value 1
	AttributeID attr_id = 0;
	SIValue v = SI_LongVal(1);
	AttributeSet_Add(&set, attr_id, v);

	uint16_t attr_count = AttributeSet_Count(set);
	TEST_ASSERT(attr_count == 1);

	bool found = AttributeSet_GetIdx(set, 0, &attr_id, &v);
	TEST_ASSERT(found             == true);
	TEST_ASSERT(attr_id           == 0);
	TEST_ASSERT(SI_GET_NUMERIC(v) == 1);

	// add the second attribute:
	// attribute ID 1
	// attribute value 2.1

	attr_id = 1;
	v = SI_DoubleVal(2.1);
	AttributeSet_Add(&set, attr_id, v);

	attr_count = AttributeSet_Count(set);
	TEST_ASSERT(attr_count == 2);

	// make sure both attributes can be retrieved

	// get first attribute
	found = AttributeSet_GetIdx(set, 0, &attr_id, &v);
	TEST_ASSERT(found             == true);
	TEST_ASSERT(attr_id           == 0);
	TEST_ASSERT(SI_GET_NUMERIC(v) == 1);

	found = AttributeSet_GetIdx(set, 1, &attr_id, &v);
	TEST_ASSERT(found             == true);
	TEST_ASSERT(attr_id           == 1);
	TEST_ASSERT(SI_GET_NUMERIC(v) == 2.1);

	// get attributes via attribute ID
	found = AttributeSet_Get(set, 1, &v);
	TEST_ASSERT(found             == true);
	TEST_ASSERT(SI_GET_NUMERIC(v) == 2.1);

	found = AttributeSet_Get(set, 0, &v);
	TEST_ASSERT(found             == true);
	TEST_ASSERT(SI_GET_NUMERIC(v) == 1);

	// clean up
	AttributeSet_Free(&set);
	TEST_ASSERT(set == NULL);
}

// assert attribute id
#define ASSERT_ATTR_ID(buff, id)                                \
{                                                               \
	AttributeID _attr_id = *((AttributeID*) (buff));            \
	buff += sizeof(AttributeID);                                \
	TEST_ASSERT(_attr_id == id);                                \
}

// assert attribute type
#define ASSERT_ATTR_TYPE(buff, t)                               \
{                                                               \
	char _t = *((char*) (buff));                                \
	buff += sizeof(char);                                       \
	TEST_ASSERT(_t == t);                                       \
}

#define GET_ATTR_VALUE_AS(buff, t) (*((t*) (buff))); buff += sizeof(t)

// assert attribute's value
#define ASSERT_ATTR_VALUE_AS(buff, t, val)                      \
{                                                               \
	t _val = *((t*) (buff));                                    \
	buff += sizeof(t);                                          \
	TEST_ASSERT(_val == val);                                   \
}

// test value representation
void test_representation() {
	SIValue v;
	AttributeID attr_id;
	AttributeSet set = NULL;

	// ATTR_TYPE_INT8
	attr_id = 0;
	v = SI_LongVal(-100);
	AttributeSet_Add(&set, attr_id, v);

	// ATTR_TYPE_INT16
	attr_id = 1;
	v = SI_LongVal(32700);
	AttributeSet_Add(&set, attr_id, v);

	// ATTR_TYPE_INT32
	attr_id = 2;
	v = SI_LongVal(-2147483600);
	AttributeSet_Add(&set, attr_id, v);

	// ATTR_TYPE_INT64
	attr_id = 3;
	v = SI_LongVal(9223372036854775000);
	AttributeSet_Add(&set, attr_id, v);

	// ATTR_TYPE_BOOL_TRUE
	attr_id = 4;
	v = SI_BoolVal(true);
	AttributeSet_Add(&set, attr_id, v);

	// ATTR_TYPE_BOOL_FALSE
	attr_id = 5;
	v = SI_BoolVal(false);
	AttributeSet_Add(&set, attr_id, v);

	// ATTR_TYPE_FLOAT
	attr_id = 6;
	v = SI_DoubleVal(-2.0);
	AttributeSet_Add(&set, attr_id, v);

	// ATTR_TYPE_DOUBLE
	attr_id = 7;
	v = SI_DoubleVal(54321.1234566789);
	AttributeSet_Add(&set, attr_id, v);

	// ATTR_TYPE_STRING
	attr_id = 8;
	const char *str = "HellO!";
	v = SI_DuplicateStringVal(str);
	str = v.stringval;
	AttributeSet_AddNoClone(&set, &attr_id, &v, 1, false);

	// ATTR_TYPE_NULL
	attr_id = 9;
	v = SI_NullVal();
	AttributeSet_AddNoClone(&set, &attr_id, &v, 1, true);

	// ATTR_TYPE_POINT
	attr_id = 10;
	SIValue p = SI_Point(1.2, 3.4);
	AttributeSet_Add(&set, attr_id, p);

	// ATTR_TYPE_VECTOR_F32
	attr_id = 11;
	SIValue vec = SI_Vectorf32(32);
	AttributeSet_AddNoClone(&set, &attr_id, &vec, 1, false);

	// ATTR_TYPE_ARRAY
	attr_id = 12;
	v = SI_Array(4);
	void *arr = v.ptrval;
	AttributeSet_AddNoClone(&set, &attr_id, &v, 1, false);

	// ATTR_TYPE_MAP
	// attr_id = 13;
	// v = SI_Map(6);
	// AttributeSet_Add(&set, attr_id, v);

	//--------------------------------------------------------------------------
	// validate attributes structure
	//--------------------------------------------------------------------------

	const char *attributes = AttributeSet_Attributes(set);
	TEST_ASSERT(attributes != NULL);

	// ATTR_TYPE_INT8
	ASSERT_ATTR_ID(attributes, 0)
	ASSERT_ATTR_TYPE(attributes, 0)
	ASSERT_ATTR_VALUE_AS(attributes, int8_t, -100)

	// ATTR_TYPE_INT16
	ASSERT_ATTR_ID(attributes, 1)
	ASSERT_ATTR_TYPE(attributes, 1)
	ASSERT_ATTR_VALUE_AS(attributes, int16_t, 32700)

	// ATTR_TYPE_INT32
	ASSERT_ATTR_ID(attributes, 2)
	ASSERT_ATTR_TYPE(attributes, 2)
	ASSERT_ATTR_VALUE_AS(attributes, int32_t, -2147483600)

	// ATTR_TYPE_INT64
	ASSERT_ATTR_ID(attributes, 3)
	ASSERT_ATTR_TYPE(attributes, 3)
	ASSERT_ATTR_VALUE_AS(attributes, int64_t, 9223372036854775000)

	// ATTR_TYPE_BOOL_TRUE
	ASSERT_ATTR_ID(attributes, 4)
	ASSERT_ATTR_TYPE(attributes, 4)

	// ATTR_TYPE_BOOL_FALSE
	ASSERT_ATTR_ID(attributes, 5)
	ASSERT_ATTR_TYPE(attributes, 5)

	// ATTR_TYPE_FLOAT
	ASSERT_ATTR_ID(attributes, 6)
	ASSERT_ATTR_TYPE(attributes, 6)
	ASSERT_ATTR_VALUE_AS(attributes, float, -2.0)

	// ATTR_TYPE_DOUBLE
	ASSERT_ATTR_ID(attributes, 7)
	ASSERT_ATTR_TYPE(attributes, 7)
	ASSERT_ATTR_VALUE_AS(attributes, double, 54321.1234566789)

	// ATTR_TYPE_STRING
	ASSERT_ATTR_ID(attributes, 8)
	ASSERT_ATTR_TYPE(attributes, 8)
	ASSERT_ATTR_VALUE_AS(attributes, char*, str);

	// ATTR_TYPE_NULL
	ASSERT_ATTR_ID(attributes, 9)
	ASSERT_ATTR_TYPE(attributes, 9)

	// ATTR_TYPE_POINT
	ASSERT_ATTR_ID(attributes, 10)
	ASSERT_ATTR_TYPE(attributes, 10)
	Point actual = GET_ATTR_VALUE_AS(attributes, Point);
	TEST_ASSERT(actual.latitude == p.point.latitude);
	TEST_ASSERT(actual.longitude == p.point.longitude);

	// ATTR_TYPE_VECTOR_F32
	ASSERT_ATTR_ID(attributes, 11)
	ASSERT_ATTR_TYPE(attributes, 11)
	ASSERT_ATTR_VALUE_AS(attributes, void*, vec.ptrval);

	// ATTR_TYPE_ARRAY
	ASSERT_ATTR_ID(attributes, 12)
	ASSERT_ATTR_TYPE(attributes, 12)
	ASSERT_ATTR_VALUE_AS(attributes, void*, arr);

	// clean up
	AttributeSet_Free(&set);
	TEST_ASSERT(set == NULL);
}

TEST_LIST = {
	{"nullAttributeSet", test_nullAttributeSet},
	{"addAttributeSet", test_addAttributeSet},
	{"representation", test_representation},
	{NULL, NULL}
};

