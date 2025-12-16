/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/datatypes/datatypes.h"
#include "src/graph/entities/attribute_set.h"

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

void test_null_attributeset() {
	SIValue v ;
	uint16_t idx ;
	AttributeSet set = NULL ;
	AttributeID attr_id = 0 ;

	TEST_ASSERT (AttributeSet_Count (set) == 0) ;

	TEST_ASSERT (AttributeSet_Contains (set, attr_id, &idx) == false) ;

	TEST_ASSERT (AttributeSet_Remove (&set, attr_id) == false ) ;

	TEST_ASSERT (AttributeSet_Get (set, attr_id, &v) == false) ;

	TEST_ASSERT (AttributeSet_ShallowClone (set) == NULL) ;

	TEST_ASSERT (AttributeSet_memoryUsage (set) == 0) ;

	// try to free a NULL set
	AttributeSet_Free (&set) ;
}

void test_attributeset_add() {
	uint16_t idx ;
	SIValue      x       = SI_NullVal () ;
	SIValue      v       = SI_DoubleVal (123) ;
	AttributeSet set     = NULL ;
	AttributeID  attr_id = 0 ;

	//--------------------------------------------------------------------------
	// add and validate a single attribute
	//--------------------------------------------------------------------------

	AttributeSet_Add (&set, &attr_id, &v, 1, false) ;
	TEST_ASSERT (set != NULL) ;

	TEST_ASSERT (AttributeSet_Count(set) == 1) ;

	TEST_ASSERT (AttributeSet_Contains (set, 12, &idx) == false) ;

	TEST_ASSERT (AttributeSet_Contains (set, attr_id, &idx) == true) ;
	TEST_ASSERT (idx == 0) ;
	
	AttributeSet_Get (set, attr_id, &x) ;
	TEST_ASSERT (SIValue_Compare (v, x, NULL) == 0) ;

	AttributeSet_GetIdx (set, 0, &attr_id, &x) ;
	TEST_ASSERT (attr_id == 0) ;
	TEST_ASSERT (SIValue_Compare (v, x, NULL) == 0) ;

	TEST_ASSERT (AttributeSet_GetKey (set, 0) == attr_id);

	AttributeSet_Free (&set) ;
	TEST_ASSERT (set == NULL) ;

	//--------------------------------------------------------------------------
	// add and validate multiple attributes
	//--------------------------------------------------------------------------

	AttributeID attr_ids[3] = {4, 6, 1} ;

	SIValue values[3] = {
		SI_ConstStringVal ("AbC!"),
		SI_BoolVal (true),
		SI_Point(21, 12.5) 
	} ;

	AttributeSet_Add (&set, attr_ids, values, 3, true) ;
	TEST_ASSERT (set != NULL) ;

	TEST_ASSERT (AttributeSet_Count(set) == 3) ;

	for (int i = 0; i < 3; i++) {
		TEST_ASSERT (AttributeSet_Contains (set, attr_ids[i], &idx) == true) ;
		TEST_ASSERT (idx == i) ;
		
		AttributeSet_Get (set, attr_ids[i], &x) ;
		TEST_ASSERT (SIValue_Compare (values[i], x, NULL) == 0) ;

		AttributeSet_GetIdx (set, i, &attr_id, &x) ;
		TEST_ASSERT (attr_id == attr_ids[i]) ;
		TEST_ASSERT (SIValue_Compare (values[i], x, NULL) == 0) ;

		TEST_ASSERT (AttributeSet_GetKey (set, i) == attr_ids[i]);
	}

	// attribute-set should duplicate heap allocated values
	AttributeSet_Get (set, attr_ids[0], &x) ;  // retrieve string attribute
	TEST_ASSERT (SI_ALLOCATION (&x) == M_VOLATILE) ;
	TEST_ASSERT (x.stringval != values[0].stringval) ;

	AttributeSet_GetIdx (set, 0, &attr_id, &x) ;
	TEST_ASSERT (SI_ALLOCATION(&x) == M_VOLATILE) ;
	TEST_ASSERT (x.stringval != values[0].stringval) ;

	AttributeSet_Free (&set) ;
	TEST_ASSERT (set == NULL) ;

	//--------------------------------------------------------------------------
	// add without cloning
	//--------------------------------------------------------------------------

	attr_id = 8 ;
	x = SI_DuplicateStringVal ("no-clone") ;
	AttributeSet_Add (&set, &attr_id, &x, 1, false) ;

	// attribute-set should NOT duplicate heap allocated values
	AttributeSet_Get (set, attr_id, &v) ;
	TEST_ASSERT (SI_ALLOCATION (&v) == M_VOLATILE) ;
	TEST_ASSERT (x.stringval == v.stringval) ;

	AttributeSet_GetIdx (set, 0, &attr_id, &v) ;
	TEST_ASSERT (SI_ALLOCATION(&v) == M_VOLATILE) ;
	TEST_ASSERT (x.stringval == v.stringval) ;

	AttributeSet_Free (&set) ;
	TEST_ASSERT (set == NULL) ;
}

void test_attributeset_update() {
	SIValue v ;
	AttributeSet set = NULL ;

	//--------------------------------------------------------------------------
	// build attribute-set
	//--------------------------------------------------------------------------

	AttributeID attr_ids[3] = {0, 1, 2} ;

	SIValue values[3] = {
		SI_ConstStringVal ("AbC!"),
		SI_BoolVal (true),
		SI_Point(21, 12.5) 
	} ;

	AttributeSet_Add (&set, attr_ids, values, 3, true) ;

	//--------------------------------------------------------------------------
	// update boolean attribute
	//--------------------------------------------------------------------------

	AttributeSet_Get (set, 1, &v) ;
	TEST_ASSERT (SIValue_IsFalse (v) == false) ;

	// update boolean attribute
	AttributeID attr_id = 1 ;
	SIValue attr_val = SI_BoolVal (false) ;
	AttributeSet_Update (NULL, &set, &attr_id, &attr_val, 1, true) ;

	AttributeSet_Get (set, 1, &v) ;
	TEST_ASSERT (SIValue_IsFalse (v) == true) ;

	//--------------------------------------------------------------------------
	// update string attribute
	//--------------------------------------------------------------------------

	AttributeSet_Get (set, 0, &v) ;
	TEST_ASSERT (SIValue_Compare (v, values[0], NULL) == 0) ;

	// update string attribute
	attr_id = 0 ;
	SIValue x = SI_DuplicateStringVal ("E4, D6") ;
	AttributeSet_Update (NULL, &set, &attr_id, &x, 1, false) ;

	AttributeSet_Get (set, 0, &v) ;
	TEST_ASSERT (SIValue_Compare (v, values[0], NULL) != 0) ;
	TEST_ASSERT (SIValue_Compare (v, x, NULL) == 0) ;

	//--------------------------------------------------------------------------
	// update non existing attribute
	//--------------------------------------------------------------------------

	TEST_ASSERT (AttributeSet_Count (set) == 3) ;

	x = SI_LongVal (99) ;
	attr_id = 3 ;

	// update should add missing attribute
	AttributeSet_Update (NULL, &set, &attr_id, &x, 1, false) ;

	TEST_ASSERT (AttributeSet_Count (set) == 4) ;

	TEST_ASSERT (AttributeSet_Get (set, attr_id, &v) == true) ;
	TEST_ASSERT (SIValue_Compare (v, x, NULL) == 0) ;

	AttributeSet_GetIdx (set, AttributeSet_Count (set) - 1, &attr_id, &v) ;
	TEST_ASSERT (attr_id == 3) ;
	TEST_ASSERT (SIValue_Compare (v, x, NULL) == 0) ;

	TEST_ASSERT (AttributeSet_GetKey
			(set, AttributeSet_Count (set) -1) == attr_id) ;

	AttributeSet_Free (&set) ;
	TEST_ASSERT (set == NULL) ;
}

void test_attributeset_remove() {
	AttributeSet set = NULL ;

	//SIValue m = SI_Map       (2);
	SIValue t = SI_Time      (time (NULL)) ;
	SIValue a = SI_Array     (4);
	SIValue p = SI_Point     (12.2, 21.1) ;
	SIValue l = SI_LongVal   (414);
	SIValue d = SI_DoubleVal (859);
	SIValue v = SI_Vectorf32 (12);

	//Map_Add (&m, SI_ConstStringVal ("k0"), SI_ConstStringVal ("v")) ;
	//Map_Add (&m, SI_ConstStringVal ("k1"), SI_LongVal (1)) ;

	SIArray_Append (&a, SI_ConstStringVal ("A")) ;
	SIArray_Append (&a, SI_ConstStringVal ("B")) ;
	SIArray_Append (&a, SI_LongVal (-1)) ;
	SIArray_Append (&a, SI_DoubleVal (-2)) ;

	float *vec_elements = (float*)SIVector_Elements (v) ;
	for(uint32_t i = 0; i < 4; i++) {
		vec_elements[i] = i ;
	}

	uint16_t n = 6 ;
	SIValue     values[6]   = {t, a, p, l, d, v} ;
	AttributeID attr_ids[6] = {0, 1, 2, 3, 4, 5} ;

	AttributeSet_Add (&set, attr_ids, values, n, true) ;
	TEST_ASSERT (AttributeSet_Count (set) == n) ;

	for (uint16_t i = 0; i < n; i++) {
		SIValue attr ;
		AttributeID attr_id ;

		TEST_ASSERT (AttributeSet_Get (set, attr_ids[i], &attr) == true) ;
		TEST_ASSERT (SIValue_Compare (values[i], attr, NULL) == 0) ;

		AttributeSet_GetIdx (set, i, &attr_id, &attr) ;
		TEST_ASSERT (attr_id == attr_ids[i]) ;
		TEST_ASSERT (SIValue_Compare (values[i], attr, NULL) == 0) ;
	}

	for (uint16_t i = 0; i < n; i++) {
		TEST_ASSERT (AttributeSet_Remove (&set, attr_ids[i])) ;
		TEST_ASSERT (AttributeSet_Count (set) == n - i - 1) ;

		for (uint16_t j = i+1; j < n; j++) {
			SIValue attr ;
			AttributeID attr_id ;

			TEST_ASSERT (AttributeSet_Get (set, attr_ids[j], &attr) == true) ;
			TEST_ASSERT (SIValue_Compare (values[j], attr, NULL) == 0) ;
		}
	}
	TEST_ASSERT (set == NULL) ;

	//--------------------------------------------------------------------------
	// remove elements in reverse order
	//--------------------------------------------------------------------------

	AttributeSet_Add (&set, attr_ids, values, n, true) ;
	TEST_ASSERT (AttributeSet_Count (set) == n) ;

	for (int16_t i = n-1; i >= 0; i--) {
		TEST_ASSERT (AttributeSet_Remove (&set, attr_ids[i])) ;
		TEST_ASSERT (AttributeSet_Count (set) == i) ;

		for (int16_t j = i-1; j >= 0; j--) {
			SIValue attr ;
			AttributeID attr_id ;

			TEST_ASSERT (AttributeSet_Get (set, attr_ids[j], &attr) == true) ;
			TEST_ASSERT (SIValue_Compare (values[j], attr, NULL) == 0) ;
		}
	}
	TEST_ASSERT (set == NULL) ;

	//--------------------------------------------------------------------------
	// remove elements via AttributeSet_Update
	//--------------------------------------------------------------------------

	AttributeSet_Add (&set, attr_ids, values, n, true) ;
	TEST_ASSERT (AttributeSet_Count (set) == n) ;

	for (uint16_t i = 0; i < n; i++) {
		SIValue v = SI_NullVal () ;
		AttributeSetChangeType change ;
		AttributeSet_Update (&change, &set, attr_ids + i, &v, 1, false) ;
		TEST_ASSERT (change == CT_DEL) ;
		TEST_ASSERT (AttributeSet_Count (set) == n - i - 1) ;

		for (uint16_t j = i+1; j < n; j++) {
			SIValue attr ;
			AttributeID attr_id ;

			TEST_ASSERT (AttributeSet_Get (set, attr_ids[j], &attr) == true) ;
			TEST_ASSERT (SIValue_Compare (values[j], attr, NULL) == 0) ;
		}
	}
	TEST_ASSERT (set == NULL) ;

	// free values
	for (uint16_t i = 0; i < n; i++) {
		SIValue_Free (values[i]) ;
	}
}

void test_attributeset_shallowClone() {
	AttributeSet set = NULL ;

	uint16_t n = 2 ;
	SIValue str = SI_ConstStringVal ("abc") ;
	SIValue num = SI_LongVal (23) ;

	SIValue     values[2] = {str, num} ;
	AttributeID ids   [2] = {0, 1} ;

	AttributeSet_Add (&set, ids, values, n, true) ;
	AttributeSet clone = AttributeSet_ShallowClone (set) ;

	TEST_ASSERT (clone != set) ;

	for (uint16_t i = 0; i < AttributeSet_Count (set); i++) {
		SIValue v ;
		SIValue clone_v ;
		AttributeID attr_id ;
		AttributeID clone_attr_id ;

		AttributeSet_GetIdx (set, i, &attr_id, &v) ;
		AttributeSet_GetIdx (clone, i, &clone_attr_id, &clone_v) ;

		TEST_ASSERT (attr_id == clone_attr_id) ;
		TEST_ASSERT (SIValue_Compare (v, clone_v, NULL) == 0) ;

		TEST_ASSERT (SI_ALLOCATION (&v) == M_VOLATILE ||
					 SI_ALLOCATION (&v) == M_NONE) ;

		TEST_ASSERT (SI_ALLOCATION (&clone_v) == M_VOLATILE ||
					 SI_ALLOCATION (&clone_v) == M_NONE) ;

		TEST_ASSERT (v.ptrval == clone_v.ptrval) ;
	}

	for (uint16_t i = 0; i < n; i++) {
		SIValue_Free (values[i]) ;
	}

	AttributeSet_Free (&set) ;
	AttributeSet_Free (&clone) ;
}

TEST_LIST = {
	{ "null_attributeset", test_null_attributeset},
	{ "attributeset_add", test_attributeset_add},
	{ "attributeset_update", test_attributeset_update},
	{ "attributeset_remove", test_attributeset_remove},
	{ "attributeset_shallowClone", test_attributeset_shallowClone},
	{ NULL, NULL }
};

