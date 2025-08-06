/*
 * Test for Record_Get bounds checking fix
 * This test verifies that Record_Get handles out-of-bounds access gracefully
 */

#include "src/value.h"
#include "src/util/rmalloc.h"
#include "src/execution_plan/record.h"

#include <stdio.h>

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

void test_record_get_bounds_checking() {
	rax *_rax = raxNew();

	// Create a record with 3 entries
	for(int i = 0; i < 3; i++) {
		char buf[2] = {(char)i, '\0'};
		raxInsert(_rax, (unsigned char *)buf, 2, NULL, NULL);
	}

	Record r = Record_New(_rax);
	
	// Add some test data
	SIValue v_string = SI_ConstStringVal("test");
	SIValue v_int = SI_LongVal(42);
	SIValue v_double = SI_DoubleVal(3.14);

	Record_AddScalar(r, 0, v_string);
	Record_AddScalar(r, 1, v_int);
	Record_AddScalar(r, 2, v_double);

	// Test normal access (should work)
	SIValue result0 = Record_Get(r, 0);
	TEST_ASSERT(SIValue_IsValid(result0));
	TEST_ASSERT(SI_TYPE(result0) == T_STRING);

	SIValue result1 = Record_Get(r, 1);
	TEST_ASSERT(SIValue_IsValid(result1));
	TEST_ASSERT(SI_TYPE(result1) == T_INT64);

	SIValue result2 = Record_Get(r, 2);
	TEST_ASSERT(SIValue_IsValid(result2));
	TEST_ASSERT(SI_TYPE(result2) == T_DOUBLE);

	// Test out-of-bounds access (should return NULL gracefully, not crash)
	SIValue result_oob = Record_Get(r, 5);  // Index 5 is out of bounds
	TEST_ASSERT(SIValue_IsNull(result_oob));

	// Test with very large index
	SIValue result_large = Record_Get(r, 10000);
	TEST_ASSERT(SIValue_IsNull(result_large));

	// Test with NULL record (should return NULL gracefully)
	SIValue result_null_record = Record_Get(NULL, 0);
	TEST_ASSERT(SIValue_IsNull(result_null_record));

	Record_Free(r);
	raxFree(_rax);
}

void test_record_helper_functions_bounds_checking() {
	rax *_rax = raxNew();

	// Create a record with 2 entries
	for(int i = 0; i < 2; i++) {
		char buf[2] = {(char)i, '\0'};
		raxInsert(_rax, (unsigned char *)buf, 2, NULL, NULL);
	}

	Record r = Record_New(_rax);
	
	// Add test data
	SIValue v_string = SI_ConstStringVal("test");
	Record_AddScalar(r, 0, v_string);

	// Test Record_GetType bounds checking
	TEST_ASSERT(Record_GetType(r, 0) == REC_TYPE_SCALAR);
	TEST_ASSERT(Record_GetType(r, 1) == REC_TYPE_UNKNOWN);
	TEST_ASSERT(Record_GetType(r, 10) == REC_TYPE_UNKNOWN);  // Out of bounds
	TEST_ASSERT(Record_GetType(NULL, 0) == REC_TYPE_UNKNOWN);  // NULL record

	// Test Record_ContainsEntry bounds checking
	TEST_ASSERT(Record_ContainsEntry(r, 0) == true);
	TEST_ASSERT(Record_ContainsEntry(r, 1) == false);
	TEST_ASSERT(Record_ContainsEntry(r, 10) == false);  // Out of bounds
	TEST_ASSERT(Record_ContainsEntry(NULL, 0) == false);  // NULL record

	// Test Record_GetNode bounds checking
	TEST_ASSERT(Record_GetNode(r, 0) == NULL);  // Not a node
	TEST_ASSERT(Record_GetNode(r, 10) == NULL);  // Out of bounds
	TEST_ASSERT(Record_GetNode(NULL, 0) == NULL);  // NULL record

	// Test Record_GetEdge bounds checking
	TEST_ASSERT(Record_GetEdge(r, 0) == NULL);  // Not an edge
	TEST_ASSERT(Record_GetEdge(r, 10) == NULL);  // Out of bounds
	TEST_ASSERT(Record_GetEdge(NULL, 0) == NULL);  // NULL record

	// Test Record_GetGraphEntity bounds checking
	TEST_ASSERT(Record_GetGraphEntity(r, 10) == NULL);  // Out of bounds
	TEST_ASSERT(Record_GetGraphEntity(NULL, 0) == NULL);  // NULL record

	Record_Free(r);
	raxFree(_rax);
}

void test_record_modification_bounds_checking() {
	rax *_rax = raxNew();

	// Create a record with 2 entries
	for(int i = 0; i < 2; i++) {
		char buf[2] = {(char)i, '\0'};
		raxInsert(_rax, (unsigned char *)buf, 2, NULL, NULL);
	}

	Record r = Record_New(_rax);

	// Test Record_AddScalar bounds checking
	SIValue v_string = SI_ConstStringVal("test");
	SIValue *result = Record_AddScalar(r, 0, v_string);
	TEST_ASSERT(result != NULL);
	
	// Out of bounds add should return NULL and not crash
	SIValue *result_oob = Record_AddScalar(r, 10, v_string);
	TEST_ASSERT(result_oob == NULL);

	// NULL record should return NULL and not crash
	SIValue *result_null = Record_AddScalar(NULL, 0, v_string);
	TEST_ASSERT(result_null == NULL);

	// Test Record_Remove bounds checking - should not crash
	Record_Remove(r, 0);  // Valid
	Record_Remove(r, 10);  // Out of bounds - should not crash
	Record_Remove(NULL, 0);  // NULL record - should not crash

	// Test Record_FreeEntry bounds checking - should not crash
	Record_FreeEntry(r, 0);  // Valid
	Record_FreeEntry(r, 10);  // Out of bounds - should not crash
	Record_FreeEntry(NULL, 0);  // NULL record - should not crash

	Record_Free(r);
	raxFree(_rax);
}

TEST_LIST = {
	{ "record_get_bounds_checking", test_record_get_bounds_checking },
	{ "record_helper_functions_bounds_checking", test_record_helper_functions_bounds_checking },
	{ "record_modification_bounds_checking", test_record_modification_bounds_checking },
	{ NULL, NULL }
};