/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "mock_log.h"
#include "src/value.h"
#include "src/util/rmalloc.h"
#include "src/datatypes/vector.h"

void setup() {
	Alloc_Reset () ;
	Logging_Reset () ;
}

#define TEST_INIT setup();
#include "acutest.h"

void test_empty_vector(void) {
	SIValue v = SIVectorf32_New(0);
	TEST_ASSERT(SI_TYPE(v) == T_VECTOR_F32);
	TEST_ASSERT(SI_ALLOCATION(&v) == M_SELF);
	TEST_ASSERT(SIVector_Dim(v) == 0);
	SIVector_Free(v);
}

void test_vector_elements(void) {
	float *f_elements;
	double *d_elements;

	//--------------------------------------------------------------------------
	// test vectorf32
	//--------------------------------------------------------------------------

	SIValue v = SIVectorf32_New(3);
	
	f_elements = (float*)SIVector_Elements(v);
	for(int i = 0; i < 3; i++) {
		TEST_ASSERT(f_elements[i] == 0);
		f_elements[i] = i;
	}

	f_elements = (float*)SIVector_Elements(v);
	for(int i = 0; i < 3; i++) {
		TEST_ASSERT(f_elements[i] == i);
	}

	f_elements = NULL;
	SIVector_Free(v);
}

void test_vector_clone(void) {
	SIValue v;
	SIValue clone;
	float  *f_elements       = NULL;
	double *d_elements       = NULL;
	float  *clone_f_elements = NULL;
	double *clone_d_elements = NULL;

	//--------------------------------------------------------------------------
	// test vectorf32
	//--------------------------------------------------------------------------

	v = SIVectorf32_New(3);
	f_elements = (float*)SIVector_Elements(v);

	for(int i = 0; i < 3; i++) {
		f_elements[i] = i;
	}

	clone = SIVector_Clone(v);

	TEST_ASSERT(SI_TYPE(clone)        == SI_TYPE(v));
	TEST_ASSERT(SIVector_Dim(clone)   == SIVector_Dim(v));
	TEST_ASSERT(SI_ALLOCATION(&clone) == M_SELF);

	clone_f_elements = (float*)SIVector_Elements(clone);
	for(int i = 0; i < 3; i++) {
		TEST_ASSERT(clone_f_elements[i] == f_elements[i]);
	}

	SIVector_Free(v);
	SIVector_Free(clone);
}

void test_vector_compare(void) {
	SIValue a;
	SIValue b;
	float  *a_f_elements = NULL;
	double *a_d_elements = NULL;
	float  *b_f_elements = NULL;
	double *b_d_elements = NULL;

	//--------------------------------------------------------------------------
	// test vectorf32
	//--------------------------------------------------------------------------

	a = SIVectorf32_New(3);
	b = SIVectorf32_New(3);

	a_f_elements = (float*)SIVector_Elements(a);
	b_f_elements = (float*)SIVector_Elements(b);

	for(int i = 0; i < 3; i++) {
		a_f_elements[i] = i;
		b_f_elements[i] = i;
	}

	TEST_ASSERT(SIVector_Compare(a, b) == 0);

	a_f_elements[0] = 1;
	TEST_ASSERT(SIVector_Compare(a, b) > 0);

	a_f_elements[0] = 0;
	b_f_elements[0] = 1;
	TEST_ASSERT(SIVector_Compare(a, b) < 0);

	SIVector_Free(a);
	SIVector_Free(b);
}

void test_vector_tostring(void) {
	SIValue v;
	char *str;
	size_t bufferLen;
	size_t bytesWritten;

	//--------------------------------------------------------------------------
	// test vectorf32
	//--------------------------------------------------------------------------

	v = SIVectorf32_New(3);

	bufferLen = 1;
	bytesWritten = 0;
	str = malloc(sizeof(char) * bufferLen);

	SIValue_ToString(v, &str, &bufferLen, &bytesWritten);
	TEST_ASSERT(strcasecmp(str, "<0.000000, 0.000000, 0.000000>") == 0);

	free(str);
	SIVector_Free(v);

	//--------------------------------------------------------------------------

	v = SIVectorf32_New(3);

	float *f_elements = (float*)SIVector_Elements(v);
	for(int i = 0; i < 3; i++) {
		f_elements[i] = i;
	}

	bufferLen = 1;
	bytesWritten = 0;
	str = malloc(sizeof(char) * bufferLen);

	SIValue_ToString(v, &str, &bufferLen, &bytesWritten);
	TEST_ASSERT(strcasecmp(str, "<0.000000, 1.000000, 2.000000>") == 0);

	free(str);
	SIVector_Free(v);
}

TEST_LIST = {
	{"empty_vector", test_empty_vector},
	{"vector_elements", test_vector_elements},
	{"vector_clone", test_vector_clone},
	{"vector_compare", test_vector_compare},
	{"vector_tostring", test_vector_tostring},
	{NULL, NULL}
};

