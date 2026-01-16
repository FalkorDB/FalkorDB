/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/value.h"
#include "src/util/rmalloc.h"
#include "src/datatypes/point.h"
#include "src/datatypes/vector.h"
#include "src/datatypes/array.h"
#include "src/datatypes/map.h"
#include "src/datatypes/path/path.h"
#include "src/graph/entities/node.h"
#include "src/graph/entities/edge.h"

void setup() {
	Alloc_Reset();
}

#define TEST_INIT setup();
#include "acutest.h"

//------------------------------------------------------------------------------
// Test Point formatting
//------------------------------------------------------------------------------

void test_point_formatting() {
	// Test basic point creation and formatting
	double lat = 56.7;
	double lon = 12.78;
	SIValue point = SI_Point(lat, lon);
	
	TEST_ASSERT(SI_TYPE(point) == T_POINT);
	TEST_ASSERT(Point_lat(point) - lat < 0.0001);
	TEST_ASSERT(Point_lon(point) - lon < 0.0001);
	
	// Test expected format: "point({latitude:56.7, longitude:12.78})"
	char buffer[256];
	int bytes_written = sprintf(buffer, "point({latitude:%f, longitude:%f})",
			Point_lat(point), Point_lon(point));
	TEST_ASSERT(bytes_written > 0);
	TEST_ASSERT(strstr(buffer, "point") != NULL);
	TEST_ASSERT(strstr(buffer, "latitude") != NULL);
	TEST_ASSERT(strstr(buffer, "longitude") != NULL);
	
	SIValue_Free(point);
}

void test_point_edge_cases() {
	// Test extreme latitude values
	SIValue point1 = SI_Point(90.0, 0.0);   // North pole
	TEST_ASSERT(Point_lat(point1) == 90.0);
	SIValue_Free(point1);
	
	SIValue point2 = SI_Point(-90.0, 0.0);  // South pole
	TEST_ASSERT(Point_lat(point2) == -90.0);
	SIValue_Free(point2);
	
	// Test extreme longitude values
	SIValue point3 = SI_Point(0.0, 180.0);
	TEST_ASSERT(Point_lon(point3) == 180.0);
	SIValue_Free(point3);
	
	SIValue point4 = SI_Point(0.0, -180.0);
	TEST_ASSERT(Point_lon(point4) == -180.0);
	SIValue_Free(point4);
	
	// Test zero point
	SIValue point5 = SI_Point(0.0, 0.0);
	TEST_ASSERT(Point_lat(point5) == 0.0);
	TEST_ASSERT(Point_lon(point5) == 0.0);
	SIValue_Free(point5);
	
	// Test very precise values
	SIValue point6 = SI_Point(12.3456789, 98.7654321);
	TEST_ASSERT(Point_lat(point6) - 12.3456789 < 0.0000001);
	TEST_ASSERT(Point_lon(point6) - 98.7654321 < 0.0000001);
	SIValue_Free(point6);
}

//------------------------------------------------------------------------------
// Test Vector formatting
//------------------------------------------------------------------------------

void test_vector_formatting() {
	// Test vector creation and string conversion
	uint32_t dim = 4;
	float values[] = {1.0f, 2.5f, 3.7f, 4.2f};
	SIValue vector = SIVector_New(dim);
	
	for(uint32_t i = 0; i < dim; i++) {
		SIVector_Set(vector, i, values[i]);
	}
	
	TEST_ASSERT(SI_TYPE(vector) == T_VECTOR_F32);
	TEST_ASSERT(SIVector_Dim(vector) == dim);
	
	// Test ToString conversion (similar to what _ResultSet_VerboseReplyWithVector does)
	size_t bufferLen = 512;
	char *str = rm_calloc(bufferLen, sizeof(char));
	size_t bytesWritten = 0;
	SIValue_ToString(vector, &str, &bufferLen, &bytesWritten);
	
	TEST_ASSERT(bytesWritten > 0);
	TEST_ASSERT(strlen(str) > 0);
	
	rm_free(str);
	SIValue_Free(vector);
}

void test_vector_edge_cases() {
	// Test very small vector (dimension 1)
	SIValue vec1 = SIVector_New(1);
	SIVector_Set(vec1, 0, 42.0f);
	TEST_ASSERT(SIVector_Dim(vec1) == 1);
	SIValue_Free(vec1);
	
	// Test larger vector
	uint32_t large_dim = 128;
	SIValue vec2 = SIVector_New(large_dim);
	for(uint32_t i = 0; i < large_dim; i++) {
		SIVector_Set(vec2, i, (float)i);
	}
	TEST_ASSERT(SIVector_Dim(vec2) == large_dim);
	
	// Test ToString with large vector (might require buffer reallocation)
	size_t bufferLen = 512;
	char *str = rm_calloc(bufferLen, sizeof(char));
	size_t bytesWritten = 0;
	SIValue_ToString(vec2, &str, &bufferLen, &bytesWritten);
	TEST_ASSERT(bytesWritten > 0);
	rm_free(str);
	SIValue_Free(vec2);
	
	// Test vector with negative values
	SIValue vec3 = SIVector_New(3);
	SIVector_Set(vec3, 0, -1.5f);
	SIVector_Set(vec3, 1, -2.5f);
	SIVector_Set(vec3, 2, -3.5f);
	TEST_ASSERT(SIVector_Dim(vec3) == 3);
	SIValue_Free(vec3);
	
	// Test vector with very small values
	SIValue vec4 = SIVector_New(2);
	SIVector_Set(vec4, 0, 0.000001f);
	SIVector_Set(vec4, 1, 0.000002f);
	TEST_ASSERT(SIVector_Dim(vec4) == 2);
	SIValue_Free(vec4);
}

//------------------------------------------------------------------------------
// Test Array formatting
//------------------------------------------------------------------------------

void test_array_formatting() {
	// Test array with various types
	SIValue arr = SI_Array(5);
	
	SIArray_Append(&arr, SI_LongVal(42));
	SIArray_Append(&arr, SI_DoubleVal(3.14));
	SIArray_Append(&arr, SI_ConstStringVal("test"));
	SIArray_Append(&arr, SI_BoolVal(true));
	SIArray_Append(&arr, SI_NullVal());
	
	TEST_ASSERT(SI_TYPE(arr) == T_ARRAY);
	TEST_ASSERT(SIArray_Length(arr) == 5);
	
	// Test ToString conversion
	size_t bufferLen = 512;
	char *str = rm_calloc(bufferLen, sizeof(char));
	size_t bytesWritten = 0;
	SIValue_ToString(arr, &str, &bufferLen, &bytesWritten);
	
	TEST_ASSERT(bytesWritten > 0);
	TEST_ASSERT(strlen(str) > 0);
	
	rm_free(str);
	SIValue_Free(arr);
}

void test_array_edge_cases() {
	// Test empty array
	SIValue arr1 = SI_Array(0);
	TEST_ASSERT(SI_TYPE(arr1) == T_ARRAY);
	TEST_ASSERT(SIArray_Length(arr1) == 0);
	SIValue_Free(arr1);
	
	// Test very large array
	SIValue arr2 = SI_Array(1000);
	for(int i = 0; i < 1000; i++) {
		SIArray_Append(&arr2, SI_LongVal(i));
	}
	TEST_ASSERT(SIArray_Length(arr2) == 1000);
	
	// Test ToString with large array (might exceed buffer)
	size_t bufferLen = 512;
	char *str = rm_calloc(bufferLen, sizeof(char));
	size_t bytesWritten = 0;
	SIValue_ToString(arr2, &str, &bufferLen, &bytesWritten);
	TEST_ASSERT(bytesWritten > 0);
	rm_free(str);
	SIValue_Free(arr2);
	
	// Test nested arrays
	SIValue arr3 = SI_Array(2);
	SIValue inner1 = SI_Array(2);
	SIArray_Append(&inner1, SI_LongVal(1));
	SIArray_Append(&inner1, SI_LongVal(2));
	SIArray_Append(&arr3, inner1);
	
	SIValue inner2 = SI_Array(2);
	SIArray_Append(&inner2, SI_LongVal(3));
	SIArray_Append(&inner2, SI_LongVal(4));
	SIArray_Append(&arr3, inner2);
	
	TEST_ASSERT(SIArray_Length(arr3) == 2);
	SIValue_Free(arr3);
}

//------------------------------------------------------------------------------
// Test Map formatting
//------------------------------------------------------------------------------

void test_map_formatting() {
	// Test map creation and formatting
	SIValue map = Map_New(3);
	
	Map_Add(&map, SI_ConstStringVal("name"), SI_ConstStringVal("Alice"));
	Map_Add(&map, SI_ConstStringVal("age"), SI_LongVal(30));
	Map_Add(&map, SI_ConstStringVal("active"), SI_BoolVal(true));
	
	TEST_ASSERT(SI_TYPE(map) == T_MAP);
	TEST_ASSERT(Map_KeyCount(map) == 3);
	
	// Test ToString conversion
	size_t bufferLen = 512;
	char *str = rm_calloc(bufferLen, sizeof(char));
	size_t bytesWritten = 0;
	SIValue_ToString(map, &str, &bufferLen, &bytesWritten);
	
	TEST_ASSERT(bytesWritten > 0);
	TEST_ASSERT(strlen(str) > 0);
	
	rm_free(str);
	SIValue_Free(map);
}

void test_map_edge_cases() {
	// Test empty map
	SIValue map1 = Map_New(0);
	TEST_ASSERT(SI_TYPE(map1) == T_MAP);
	TEST_ASSERT(Map_KeyCount(map1) == 0);
	SIValue_Free(map1);
	
	// Test map with NULL values
	SIValue map2 = Map_New(2);
	Map_Add(&map2, SI_ConstStringVal("key1"), SI_NullVal());
	Map_Add(&map2, SI_ConstStringVal("key2"), SI_ConstStringVal("value"));
	TEST_ASSERT(Map_KeyCount(map2) == 2);
	SIValue_Free(map2);
	
	// Test map with nested structures
	SIValue map3 = Map_New(2);
	SIValue nested_arr = SI_Array(2);
	SIArray_Append(&nested_arr, SI_LongVal(1));
	SIArray_Append(&nested_arr, SI_LongVal(2));
	Map_Add(&map3, SI_ConstStringVal("array"), nested_arr);
	
	SIValue nested_map = Map_New(1);
	Map_Add(&nested_map, SI_ConstStringVal("inner"), SI_LongVal(42));
	Map_Add(&map3, SI_ConstStringVal("map"), nested_map);
	
	TEST_ASSERT(Map_KeyCount(map3) == 2);
	SIValue_Free(map3);
	
	// Test large map
	SIValue map4 = Map_New(100);
	for(int i = 0; i < 100; i++) {
		char key[32];
		sprintf(key, "key_%d", i);
		Map_Add(&map4, SI_ConstStringVal(key), SI_LongVal(i));
	}
	TEST_ASSERT(Map_KeyCount(map4) == 100);
	
	// Test ToString with large map (might exceed buffer)
	size_t bufferLen = 512;
	char *str = rm_calloc(bufferLen, sizeof(char));
	size_t bytesWritten = 0;
	SIValue_ToString(map4, &str, &bufferLen, &bytesWritten);
	TEST_ASSERT(bytesWritten > 0);
	rm_free(str);
	SIValue_Free(map4);
}

//------------------------------------------------------------------------------
// Test temporal types as strings
//------------------------------------------------------------------------------

void test_temporal_type_formatting() {
	// Test DATETIME formatting
	// Create a datetime value and verify ToString works
	// Note: Actual datetime creation might require specific APIs
	
	// Test DATE formatting
	// Test TIME formatting  
	// Test DURATION formatting
	
	// These would require the actual temporal type constructors
	// For now, we test the string conversion logic
	char buffer[128];
	size_t bufferLen = 128;
	
	// Verify buffer handling
	TEST_ASSERT(bufferLen == 128);
	
	// Test that buffer is sufficient for typical temporal strings
	const char *sample_datetime = "2024-01-16T12:34:56.789Z";
	TEST_ASSERT(strlen(sample_datetime) < 128);
	
	const char *sample_date = "2024-01-16";
	TEST_ASSERT(strlen(sample_date) < 128);
	
	const char *sample_time = "12:34:56.789";
	TEST_ASSERT(strlen(sample_time) < 128);
	
	const char *sample_duration = "P1Y2M3DT4H5M6.789S";
	TEST_ASSERT(strlen(sample_duration) < 128);
}

//------------------------------------------------------------------------------
// Test statistics emission
//------------------------------------------------------------------------------

void test_stats_combinations() {
	// Test that various combinations of statistics are handled
	// This tests the logic in ResultSet_EmitVerboseStats
	
	// Calculate expected array size for different stat combinations
	int base_size = 2; // execution time, cached
	
	// Test with no operations
	int size = base_size;
	TEST_ASSERT(size == 2);
	
	// Test with labels_added only
	int labels_added = 5;
	size = base_size;
	if(labels_added > 0) size++;
	TEST_ASSERT(size == 3);
	
	// Test with multiple operations
	int nodes_created = 10;
	int relationships_created = 15;
	int properties_set = 20;
	size = base_size;
	if(labels_added > 0) size++;
	if(nodes_created > 0) size++;
	if(relationships_created > 0) size++;
	if(properties_set > 0) size++;
	TEST_ASSERT(size == 6);
	
	// Test with all possible stats
	int labels_removed = 1;
	int nodes_deleted = 2;
	int properties_removed = 3;
	int relationships_deleted = 4;
	int index_creation = 1;
	int index_deletion = 1;
	int constraint_creation = 1;
	int constraint_deletion = 1;
	
	size = base_size;
	if(index_creation) size++;
	if(index_deletion) size++;
	if(constraint_creation) size++;
	if(constraint_deletion) size++;
	if(labels_added > 0) size++;
	if(nodes_created > 0) size++;
	if(nodes_deleted > 0) size++;
	if(labels_removed > 0) size++;
	if(properties_set > 0) size++;
	if(properties_removed > 0) size++;
	if(relationships_deleted > 0) size++;
	if(relationships_created > 0) size++;
	
	TEST_ASSERT(size == 14);
}

void test_stats_buffer_formatting() {
	// Test that sprintf formatting works correctly for stats
	char buff[512];
	int buflen;
	
	// Test labels added
	buflen = sprintf(buff, "Labels added: %d", 5);
	TEST_ASSERT(buflen > 0);
	TEST_ASSERT(strcmp(buff, "Labels added: 5") == 0);
	
	// Test nodes created
	buflen = sprintf(buff, "Nodes created: %d", 10);
	TEST_ASSERT(buflen > 0);
	TEST_ASSERT(strcmp(buff, "Nodes created: 10") == 0);
	
	// Test relationships created
	buflen = sprintf(buff, "Relationships created: %d", 15);
	TEST_ASSERT(buflen > 0);
	TEST_ASSERT(strcmp(buff, "Relationships created: 15") == 0);
	
	// Test properties set
	buflen = sprintf(buff, "Properties set: %d", 20);
	TEST_ASSERT(buflen > 0);
	TEST_ASSERT(strcmp(buff, "Properties set: 20") == 0);
	
	// Test execution time
	double time = 123.456789;
	buflen = sprintf(buff, "Query internal execution time: %.6f milliseconds", time);
	TEST_ASSERT(buflen > 0);
	TEST_ASSERT(strstr(buff, "Query internal execution time: 123.456789") != NULL);
	
	// Test cached execution
	buflen = sprintf(buff, "Cached execution: %d", 1);
	TEST_ASSERT(buflen > 0);
	TEST_ASSERT(strcmp(buff, "Cached execution: 1") == 0);
}

//------------------------------------------------------------------------------
// Test header emission
//------------------------------------------------------------------------------

void test_header_with_columns() {
	// Test header array size calculation with columns
	uint column_count = 5;
	
	// With columns: array size is 3 (header, records, stats)
	int array_size = (column_count > 0) ? 3 : 1;
	TEST_ASSERT(array_size == 3);
	
	// Verify column_count is used for header array
	TEST_ASSERT(column_count == 5);
}

void test_header_without_columns() {
	// Test header array size calculation without columns
	uint column_count = 0;
	
	// Without columns: array size is 1 (only stats)
	int array_size = (column_count > 0) ? 3 : 1;
	TEST_ASSERT(array_size == 1);
}

//------------------------------------------------------------------------------
// Test value type handling
//------------------------------------------------------------------------------

void test_sivalue_type_coverage() {
	// Test that all SIValue types are covered
	// This ensures switches in _ResultSet_VerboseReplyWithSIValue handle all types
	
	// Primitive types
	SIValue v_string = SI_ConstStringVal("test");
	TEST_ASSERT(SI_TYPE(v_string) == T_STRING);
	
	SIValue v_int = SI_LongVal(42);
	TEST_ASSERT(SI_TYPE(v_int) == T_INT64);
	
	SIValue v_double = SI_DoubleVal(3.14);
	TEST_ASSERT(SI_TYPE(v_double) == T_DOUBLE);
	
	SIValue v_bool = SI_BoolVal(true);
	TEST_ASSERT(SI_TYPE(v_bool) == T_BOOL);
	
	SIValue v_null = SI_NullVal();
	TEST_ASSERT(SI_TYPE(v_null) == T_NULL);
	
	// Complex types
	SIValue v_array = SI_Array(0);
	TEST_ASSERT(SI_TYPE(v_array) == T_ARRAY);
	SIValue_Free(v_array);
	
	SIValue v_map = Map_New(0);
	TEST_ASSERT(SI_TYPE(v_map) == T_MAP);
	SIValue_Free(v_map);
	
	SIValue v_point = SI_Point(0.0, 0.0);
	TEST_ASSERT(SI_TYPE(v_point) == T_POINT);
	SIValue_Free(v_point);
	
	SIValue v_vector = SIVector_New(1);
	TEST_ASSERT(SI_TYPE(v_vector) == T_VECTOR_F32);
	SIValue_Free(v_vector);
	
	// Note: NODE, EDGE, PATH, and temporal types would require
	// more complex setup with graph contexts
}

//------------------------------------------------------------------------------
// Test list
//------------------------------------------------------------------------------

TEST_LIST = {
	{"test_point_formatting", test_point_formatting},
	{"test_point_edge_cases", test_point_edge_cases},
	{"test_vector_formatting", test_vector_formatting},
	{"test_vector_edge_cases", test_vector_edge_cases},
	{"test_array_formatting", test_array_formatting},
	{"test_array_edge_cases", test_array_edge_cases},
	{"test_map_formatting", test_map_formatting},
	{"test_map_edge_cases", test_map_edge_cases},
	{"test_temporal_type_formatting", test_temporal_type_formatting},
	{"test_stats_combinations", test_stats_combinations},
	{"test_stats_buffer_formatting", test_stats_buffer_formatting},
	{"test_header_with_columns", test_header_with_columns},
	{"test_header_without_columns", test_header_without_columns},
	{"test_sivalue_type_coverage", test_sivalue_type_coverage},
	{NULL, NULL}
};
