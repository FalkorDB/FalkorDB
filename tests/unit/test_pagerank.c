/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/algorithms/pagerank.h"

#include <stdio.h>
#include <stdlib.h>

void setup() {
	Alloc_Reset();
	GrB_init(GrB_NONBLOCKING);
}

void tearDown() {
	GrB_finalize();
}

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

void test_pagerank() {
	GrB_Matrix A;
	GrB_Info info;
	double tol = 1e-4 ;
	int iters, itermax = 100 ;
	LAGraph_PageRank *ranking = NULL;

	/* Graph on the cover of the book, 'Graph Algorithms in the language of linear algebra'.
	A = [
	    0 0 0 1 0 0 0
	    1 0 0 0 0 0 0
	    0 0 0 1 0 1 1
	    1 0 0 0 0 0 1
	    0 1 0 0 0 0 1
	    0 0 1 0 1 0 0
	    0 1 0 0 0 0 0 ] ;
	*/

	info = GrB_Matrix_new(&A, GrB_BOOL, 7, 7);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_setElement_BOOL(A, true, 3, 0);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_setElement_BOOL(A, true, 0, 1);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_setElement_BOOL(A, true, 3, 2);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_setElement_BOOL(A, true, 5, 2);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_setElement_BOOL(A, true, 6, 2);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_setElement_BOOL(A, true, 0, 3);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_setElement_BOOL(A, true, 6, 3);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_setElement_BOOL(A, true, 1, 4);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_setElement_BOOL(A, true, 6, 4);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_setElement_BOOL(A, true, 2, 5);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_setElement_BOOL(A, true, 4, 5);
	TEST_ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_setElement_BOOL(A, true, 1, 6);
	TEST_ASSERT(info == GrB_SUCCESS);

	Pagerank(&ranking, A, itermax, tol, &iters);

	// Page:5, pagerank:0.392289
	// Page:2, pagerank:0.387241
	// Page:3, pagerank:0.050505
	// Page:4, pagerank:0.049130
	// Page:0, pagerank:0.042893
	// Page:1, pagerank:0.039658
	// Page:6, pagerank:0.038283

	LAGraph_PageRank expectations[7] = {
		{5, 0.392289},
		{2, 0.387241},
		{3, 0.050505},
		{4, 0.049130},
		{0, 0.042893},
		{1, 0.039658},
		{6, 0.038283},
	};

	for(int i = 0; i < 7; i++) {
		TEST_ASSERT(fabs((double)ranking[i].page - expectations[i].page) < 0.000001);
		TEST_ASSERT(fabs((double)ranking[i].pagerank - expectations[i].pagerank) < 0.000001);
	}
    
	rm_free(ranking);
	GrB_free(&A);
}

TEST_LIST = {
	{"pagerank", test_pagerank},
	{NULL, NULL}
};
