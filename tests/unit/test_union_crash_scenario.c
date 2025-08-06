/*
 * Integration test for UNION ALL crash scenario
 * This test simulates the problematic query pattern to verify the fix
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

void test_union_all_record_scenario() {
	// This test simulates the scenario where UNION ALL processing
	// might create records with mismatched indices
	
	rax *mapping1 = raxNew();
	rax *mapping2 = raxNew();
	
	// Create records with different mappings (simulating UNION branches)
	// First branch: has "cdo" at index 0
	raxInsert(mapping1, (unsigned char *)"cdo", 3, (void*)0, NULL);
	
	// Second branch: might have different structure due to processing order
	raxInsert(mapping2, (unsigned char *)"cdo", 3, (void*)0, NULL);
	raxInsert(mapping2, (unsigned char *)"temp", 4, (void*)1, NULL);  // Extra entry
	
	Record r1 = Record_New(mapping1);
	Record r2 = Record_New(mapping2);
	
	// Add data to first record (single entry)
	SIValue node_val = SI_ConstStringVal("node_from_C");
	Record_AddScalar(r1, 0, node_val);
	
	// Add data to second record (two entries)
	SIValue node_val2 = SI_ConstStringVal("node_from_E");
	SIValue temp_val = SI_LongVal(42);
	Record_AddScalar(r2, 0, node_val2);
	Record_AddScalar(r2, 1, temp_val);
	
	// Test that we can access valid indices
	SIValue result1 = Record_Get(r1, 0);
	TEST_ASSERT(!SIValue_IsNull(result1));
	
	SIValue result2 = Record_Get(r2, 0);
	TEST_ASSERT(!SIValue_IsNull(result2));
	
	SIValue result3 = Record_Get(r2, 1);
	TEST_ASSERT(!SIValue_IsNull(result3));
	
	// Test that accessing out-of-bounds indices doesn't crash
	// This simulates the crash scenario where UNION processing
	// might try to access an index that doesn't exist in a record
	SIValue result_oob1 = Record_Get(r1, 1);  // r1 only has index 0
	TEST_ASSERT(SIValue_IsNull(result_oob1));
	
	SIValue result_oob2 = Record_Get(r1, 2);  // Way out of bounds
	TEST_ASSERT(SIValue_IsNull(result_oob2));
	
	SIValue result_oob3 = Record_Get(r2, 5);  // r2 has indices 0,1 but not 5
	TEST_ASSERT(SIValue_IsNull(result_oob3));
	
	// Test with very large indices (potential integer overflow scenarios)
	SIValue result_huge = Record_Get(r1, UINT32_MAX);
	TEST_ASSERT(SIValue_IsNull(result_huge));
	
	// Clean up
	Record_Free(r1);
	Record_Free(r2);
	raxFree(mapping1);
	raxFree(mapping2);
}

void test_record_merge_scenario() {
	// This test simulates record merging issues that might occur
	// during UNION ALL processing with ORDER BY
	
	rax *mapping = raxNew();
	
	// Create a mapping for merged records
	raxInsert(mapping, (unsigned char *)"cdo", 3, (void*)0, NULL);
	raxInsert(mapping, (unsigned char *)"order_key", 9, (void*)1, NULL);
	
	Record r1 = Record_New(mapping);
	Record r2 = Record_New(mapping);
	
	// Simulate first branch data (C nodes)
	SIValue c_node = SI_ConstStringVal("C_node");
	SIValue c_time = SI_LongVal(100);
	Record_AddScalar(r1, 0, c_node);
	Record_AddScalar(r1, 1, c_time);
	
	// Simulate second branch data (E nodes)  
	SIValue e_node = SI_ConstStringVal("E_node");
	SIValue e_time = SI_LongVal(200);
	Record_AddScalar(r2, 0, e_node);
	Record_AddScalar(r2, 1, e_time);
	
	// Test normal access
	TEST_ASSERT(!SIValue_IsNull(Record_Get(r1, 0)));
	TEST_ASSERT(!SIValue_IsNull(Record_Get(r1, 1)));
	TEST_ASSERT(!SIValue_IsNull(Record_Get(r2, 0)));
	TEST_ASSERT(!SIValue_IsNull(Record_Get(r2, 1)));
	
	// Test that out-of-bounds access doesn't crash
	// This would previously cause the assertion failure
	TEST_ASSERT(SIValue_IsNull(Record_Get(r1, 2)));
	TEST_ASSERT(SIValue_IsNull(Record_Get(r2, 2)));
	TEST_ASSERT(SIValue_IsNull(Record_Get(r1, 10)));
	TEST_ASSERT(SIValue_IsNull(Record_Get(r2, 10)));
	
	// Clean up
	Record_Free(r1);
	Record_Free(r2);
	raxFree(mapping);
}

TEST_LIST = {
	{ "union_all_record_scenario", test_union_all_record_scenario },
	{ "record_merge_scenario", test_record_merge_scenario },
	{ NULL, NULL }
};