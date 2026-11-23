/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "src/util/rmalloc.h"
#include "src/graph/graph.h"
#include "src/serializers/serializer_io.h"
#include "src/serializers/encoder/v20/encode_v20.h"
#include "GraphBLAS/Include/GraphBLAS.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

void setup() {
	// use the malloc family for allocations
	Alloc_Reset();

	// initialize GraphBLAS
	GrB_init(GrB_NONBLOCKING);

	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW); // all matrices in CSR format
	GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER); // never hypersparse
}

void tearDown() {
	GrB_finalize();
}

// encode g's relation matrices, returns the encoded bytes
static unsigned char *_encode_relations
(
	Graph *g,    // graph
	size_t *len  // [output] number of bytes
) {
	FILE *f = tmpfile();
	TEST_ASSERT(f != NULL);

	SerializerIO rdb = SerializerIO_FromStream(f, true);
	RdbSaveRelationMatrices_v20(rdb, g);
	SerializerIO_Free(&rdb);

	long n = ftell(f);
	TEST_ASSERT(n > 0);
	*len = (size_t)n;

	unsigned char *buf = malloc(*len);
	rewind(f);
	TEST_ASSERT(fread(buf, 1, *len, f) == *len);
	fclose(f);

	return buf;
}

// a relation flagged multi-edge (its matrix entry count disagrees with its
// edge count statistic) that holds no tensor must encode its tensor section
// as a single 0, exactly like a relation without tensors; encoders up to
// 4.22.3 wrote three zeros, which made the RDB unloadable
void test_empty_tensor_section(void) {
	Graph *g = Graph_New(16, 16);
	RelationID r = Graph_AddRelationType(g);

	Node n;
	for(int i = 0; i < 2; i++) {
		n = GE_NEW_NODE();
		Graph_CreateNode(g, &n, NULL, 0);
	}

	Edge e;
	Graph_CreateEdge(g, 0, 1, r, &e);
	TEST_ASSERT(!Graph_RelationshipContainsMultiEdge(g, r));

	size_t expected_len;
	unsigned char *expected = _encode_relations(g, &expected_len);

	// skew the edge count statistic, the relation is now flagged multi-edge
	// though its matrix holds no tensor
	GraphStatistics_IncEdgeCount(&g->stats, r, 1);
	TEST_ASSERT(Graph_RelationshipContainsMultiEdge(g, r));

	size_t len;
	unsigned char *actual = _encode_relations(g, &len);

	// the matrix encodes to the same size either way (its container holds
	// pointers, so the bytes differ); only the tensor section could grow
	TEST_CHECK(len == expected_len);
	TEST_MSG("expected %zu bytes, got %zu", expected_len, len);

	// both end with the lone "no tensors" 0
	uint64_t zero = 0;
	TEST_ASSERT(len >= sizeof(zero));
	TEST_CHECK(memcmp(actual + len - sizeof(zero), &zero, sizeof(zero)) == 0);
	TEST_CHECK(memcmp(expected + expected_len - sizeof(zero), &zero,
				sizeof(zero)) == 0);

	free(actual);
	free(expected);
	Graph_Free(g);
}

TEST_LIST = {
	{"empty_tensor_section", test_empty_tensor_section},
	{NULL, NULL}
};
