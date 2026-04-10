/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// tests for RdbLoadDeletedNodes / RdbLoadDeletedEdges buffer validation
// verifies that a misaligned buffer does not cause use-after-free / double-free

#include "src/util/rmalloc.h"
#include "src/graph/graph.h"
#include "src/serializers/graph_extensions.h"
#include "src/serializers/decoders/current/v19/decode_v19.h"
#include "src/serializers/serializer_io.h"
#include "GraphBLAS/Include/GraphBLAS.h"

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

#define NODE_CAP 16384
#define EDGE_CAP 16384

//------------------------------------------------------------------------------
// helpers
//------------------------------------------------------------------------------

// create a pipe-based writer/reader pair
static void _create_pipe_io
(
	int pipefd[2],
	FILE **fs_write,
	FILE **fs_read,
	SerializerIO *writer,
	SerializerIO *reader
) {
	TEST_ASSERT(pipe(pipefd) != -1);
	*fs_write = fdopen(pipefd[1], "w");
	*fs_read  = fdopen(pipefd[0], "r");
	TEST_ASSERT(*fs_write != NULL);
	TEST_ASSERT(*fs_read  != NULL);
	*writer = SerializerIO_FromStream(*fs_write, true);
	*reader = SerializerIO_FromStream(*fs_read, false);
	TEST_ASSERT(*writer != NULL);
	TEST_ASSERT(*reader != NULL);
}

// close writer end of the pipe
static void _close_writer
(
	int pipefd[2],
	FILE *fs_write,
	SerializerIO *writer
) {
	fclose(fs_write);
	close(pipefd[1]);
	SerializerIO_Free(writer);
}

// close reader end of the pipe
static void _close_reader
(
	int pipefd[2],
	FILE *fs_read,
	SerializerIO *reader
) {
	fclose(fs_read);
	close(pipefd[0]);
	SerializerIO_Free(reader);
}

//------------------------------------------------------------------------------
// setup / teardown
//------------------------------------------------------------------------------

void setup() {
	Alloc_Reset();
	GrB_init(GrB_NONBLOCKING);
	GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
	GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER);
}

void tearDown() {
	GrB_finalize();
}

//------------------------------------------------------------------------------
// test: misaligned buffer in RdbLoadDeletedNodes_v19
//------------------------------------------------------------------------------

void test_deletedNodes_misaligned_buffer(void) {
	// write a buffer whose byte length (7) is not a multiple of
	// sizeof(NodeID) (8) — this must not cause a use-after-free or
	// double-free in release builds

	int pipefd[2];
	FILE *fs_write, *fs_read;
	SerializerIO writer, reader;
	_create_pipe_io(pipefd, &fs_write, &fs_read, &writer, &reader);

	// 7 bytes — misaligned for uint64_t
	uint8_t bad_data[7] = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03};
	SerializerIO_WriteBuffer(writer, bad_data, sizeof(bad_data));

	_close_writer(pipefd, fs_write, &writer);

	Graph *g = Graph_New(NODE_CAP, EDGE_CAP);
	Graph_AcquireWriteLock(g);

	// should detect misalignment, free the buffer, and return immediately
	// without touching the freed memory
	RdbLoadDeletedNodes_v19(reader, g, 0);

	// no nodes should have been marked as deleted
	TEST_ASSERT(Graph_DeletedNodeCount(g) == 0);

	Graph_ReleaseLock(g);
	Graph_Free(g);

	_close_reader(pipefd, fs_read, &reader);
}

//------------------------------------------------------------------------------
// test: misaligned buffer in RdbLoadDeletedEdges_v19
//------------------------------------------------------------------------------

void test_deletedEdges_misaligned_buffer(void) {
	// same idea — 7 bytes is not aligned to sizeof(EdgeID)

	int pipefd[2];
	FILE *fs_write, *fs_read;
	SerializerIO writer, reader;
	_create_pipe_io(pipefd, &fs_write, &fs_read, &writer, &reader);

	uint8_t bad_data[7] = {0xCA, 0xFE, 0xBA, 0xBE, 0x04, 0x05, 0x06};
	SerializerIO_WriteBuffer(writer, bad_data, sizeof(bad_data));

	_close_writer(pipefd, fs_write, &writer);

	Graph *g = Graph_New(NODE_CAP, EDGE_CAP);
	Graph_AcquireWriteLock(g);

	RdbLoadDeletedEdges_v19(reader, g, 0);

	TEST_ASSERT(Graph_DeletedEdgeCount(g) == 0);

	Graph_ReleaseLock(g);
	Graph_Free(g);

	_close_reader(pipefd, fs_read, &reader);
}

//------------------------------------------------------------------------------
// test: aligned buffer in RdbLoadDeletedNodes_v19 (happy path)
//------------------------------------------------------------------------------

void test_deletedNodes_aligned_buffer(void) {
	// write a properly aligned buffer with 2 node IDs and verify they
	// are recorded as deleted

	int pipefd[2];
	FILE *fs_write, *fs_read;
	SerializerIO writer, reader;
	_create_pipe_io(pipefd, &fs_write, &fs_read, &writer, &reader);

	// two valid NodeIDs
	NodeID ids[2] = {0, 1};
	SerializerIO_WriteBuffer(writer, ids, sizeof(ids));

	_close_writer(pipefd, fs_write, &writer);

	Graph *g = Graph_New(NODE_CAP, EDGE_CAP);
	Graph_AcquireWriteLock(g);

	// pre-allocate the node slots so MarkDeleted has something to mark
	Node n;
	for (int i = 0; i < 2; i++) {
		n = GE_NEW_NODE();
		Graph_CreateNode(g, &n, NULL, 0);
	}

	TEST_ASSERT(Graph_DeletedNodeCount(g) == 0);

	RdbLoadDeletedNodes_v19(reader, g, 2);

	TEST_ASSERT(Graph_DeletedNodeCount(g) == 2);

	Graph_ReleaseLock(g);
	Graph_Free(g);

	_close_reader(pipefd, fs_read, &reader);
}

//------------------------------------------------------------------------------
// test: aligned buffer in RdbLoadDeletedEdges_v19 (happy path)
//------------------------------------------------------------------------------

void test_deletedEdges_aligned_buffer(void) {
	int pipefd[2];
	FILE *fs_write, *fs_read;
	SerializerIO writer, reader;
	_create_pipe_io(pipefd, &fs_write, &fs_read, &writer, &reader);

	// one valid EdgeID
	EdgeID ids[1] = {0};
	SerializerIO_WriteBuffer(writer, ids, sizeof(ids));

	_close_writer(pipefd, fs_write, &writer);

	Graph *g = Graph_New(NODE_CAP, EDGE_CAP);
	Graph_AcquireWriteLock(g);

	// create 2 nodes and 1 edge so edge slot 0 exists
	Node n;
	Edge e;
	for (int i = 0; i < 2; i++) {
		n = GE_NEW_NODE();
		Graph_CreateNode(g, &n, NULL, 0);
	}
	int r = Graph_AddRelationType(g);
	Graph_CreateEdge(g, 0, 1, r, &e);

	TEST_ASSERT(Graph_DeletedEdgeCount(g) == 0);

	RdbLoadDeletedEdges_v19(reader, g, 1);

	TEST_ASSERT(Graph_DeletedEdgeCount(g) == 1);

	Graph_ReleaseLock(g);
	Graph_Free(g);

	_close_reader(pipefd, fs_read, &reader);
}

//------------------------------------------------------------------------------
// test: single-byte buffer (edge case)
//------------------------------------------------------------------------------

void test_deletedNodes_single_byte_buffer(void) {
	// a 1-byte buffer is the smallest misalignment case

	int pipefd[2];
	FILE *fs_write, *fs_read;
	SerializerIO writer, reader;
	_create_pipe_io(pipefd, &fs_write, &fs_read, &writer, &reader);

	uint8_t one = 0xFF;
	SerializerIO_WriteBuffer(writer, &one, 1);

	_close_writer(pipefd, fs_write, &writer);

	Graph *g = Graph_New(NODE_CAP, EDGE_CAP);
	Graph_AcquireWriteLock(g);

	RdbLoadDeletedNodes_v19(reader, g, 0);

	TEST_ASSERT(Graph_DeletedNodeCount(g) == 0);

	Graph_ReleaseLock(g);
	Graph_Free(g);

	_close_reader(pipefd, fs_read, &reader);
}

//------------------------------------------------------------------------------
// test list
//------------------------------------------------------------------------------

TEST_LIST = {
	{"deletedNodes_misaligned_buffer", test_deletedNodes_misaligned_buffer},
	{"deletedEdges_misaligned_buffer", test_deletedEdges_misaligned_buffer},
	{"deletedNodes_aligned_buffer",    test_deletedNodes_aligned_buffer},
	{"deletedEdges_aligned_buffer",    test_deletedEdges_aligned_buffer},
	{"deletedNodes_single_byte",       test_deletedNodes_single_byte_buffer},
	{NULL, NULL}
};
