/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// Tests for RdbLoadDeletedNodes / RdbLoadDeletedEdges buffer validation.
// Malformed buffers (misaligned or count-mismatch) must cause the process
// to exit via RedisModule_Assert.  We verify this with fork-based death tests.

#include "src/RG.h"
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
#include <sys/wait.h>

void setup();
void tearDown();

#define TEST_INIT setup();
#define TEST_FINI tearDown();
#include "acutest.h"

#define NODE_CAP 16384
#define EDGE_CAP 16384

//------------------------------------------------------------------------------
// mock Redis module functions for death-test child processes
//------------------------------------------------------------------------------

static void _mock_log
(
	RedisModuleCtx *ctx,
	const char *level,
	const char *fmt,
	...
) {
	(void)ctx; (void)level; (void)fmt;
}

static void _mock_assert
(
	const char *estr,
	const char *file,
	int line
) {
	(void)estr; (void)file; (void)line;
	// use _exit to avoid atexit handlers (ASAN cleanup can hang after fork)
	_exit(1);
}

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

// Death-test helper: write data to a pipe, fork a child that calls the loader,
// assert the child exits abnormally (non-zero exit or signal).
typedef void (*DeletedEntityLoader)(SerializerIO, Graph *, uint64_t);

static void _assert_loader_exits
(
	const void *data,
	size_t data_len,
	uint64_t expected_count,
	DeletedEntityLoader loader
) {
	// create pipe and write data before fork
	int pipefd[2];
	TEST_ASSERT(pipe(pipefd) != -1);

	FILE *fs_write = fdopen(pipefd[1], "w");
	TEST_ASSERT(fs_write != NULL);
	SerializerIO writer = SerializerIO_FromStream(fs_write, true);
	SerializerIO_WriteBuffer(writer, data, data_len);
	fclose(fs_write);
	close(pipefd[1]);
	SerializerIO_Free(&writer);

	// create graph before fork to avoid GrB thread-pool issues
	Graph *g = Graph_New(NODE_CAP, EDGE_CAP);
	Graph_AcquireWriteLock(g);

	pid_t pid = fork();
	TEST_ASSERT(pid >= 0);

	if (pid == 0) {
		// child: install mock Redis module functions
		RedisModule_Log    = (void (*)(RedisModuleCtx *, const char *,
			const char *, ...))_mock_log;
		RedisModule__Assert = _mock_assert;

		FILE *fs_read = fdopen(pipefd[0], "r");
		SerializerIO reader = SerializerIO_FromStream(fs_read, false);

		// should detect malformed buffer and exit(1)
		loader(reader, g, expected_count);

		// must not be reached
		_exit(0);
	}

	// parent: close read end (child owns it) and wait
	close(pipefd[0]);

	int status;
	waitpid(pid, &status, 0);

	Graph_ReleaseLock(g);
	Graph_Free(g);

	bool died = false;
	if(WIFEXITED(status))        died = (WEXITSTATUS(status) != 0);
	else if(WIFSIGNALED(status)) died = true;
	TEST_ASSERT(died);
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
// death tests: misaligned buffer
//------------------------------------------------------------------------------

void test_deletedNodes_misaligned_buffer(void) {
	uint8_t bad[7] = {0xDE, 0xAD, 0xBE, 0xEF, 0x01, 0x02, 0x03};
	_assert_loader_exits(bad, sizeof(bad), 0,
		(DeletedEntityLoader)RdbLoadDeletedNodes_v19);
}

void test_deletedEdges_misaligned_buffer(void) {
	uint8_t bad[7] = {0xCA, 0xFE, 0xBA, 0xBE, 0x04, 0x05, 0x06};
	_assert_loader_exits(bad, sizeof(bad), 0,
		(DeletedEntityLoader)RdbLoadDeletedEdges_v19);
}

void test_deletedNodes_single_byte_buffer(void) {
	uint8_t one = 0xFF;
	_assert_loader_exits(&one, 1, 0,
		(DeletedEntityLoader)RdbLoadDeletedNodes_v19);
}

//------------------------------------------------------------------------------
// death tests: count mismatch (aligned buffer, wrong count)
//------------------------------------------------------------------------------

void test_deletedNodes_count_mismatch(void) {
	// 2 valid NodeIDs (16 bytes) but claim 3 — count mismatch
	NodeID ids[2] = {0, 1};
	_assert_loader_exits(ids, sizeof(ids), 3,
		(DeletedEntityLoader)RdbLoadDeletedNodes_v19);
}

void test_deletedEdges_count_mismatch(void) {
	// 1 valid EdgeID (8 bytes) but claim 2 — count mismatch
	EdgeID ids[1] = {0};
	_assert_loader_exits(ids, sizeof(ids), 2,
		(DeletedEntityLoader)RdbLoadDeletedEdges_v19);
}

//------------------------------------------------------------------------------
// happy path: aligned buffer with correct count
//------------------------------------------------------------------------------

void test_deletedNodes_aligned_buffer(void) {
	int pipefd[2];
	FILE *fs_write, *fs_read;
	SerializerIO writer, reader;
	_create_pipe_io(pipefd, &fs_write, &fs_read, &writer, &reader);

	NodeID ids[2] = {0, 1};
	SerializerIO_WriteBuffer(writer, ids, sizeof(ids));
	_close_writer(pipefd, fs_write, &writer);

	Graph *g = Graph_New(NODE_CAP, EDGE_CAP);
	Graph_AcquireWriteLock(g);

	// pre-allocate node slots so MarkDeleted has something to mark
	Node n;
	for(int i = 0; i < 2; i++) {
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

void test_deletedEdges_aligned_buffer(void) {
	int pipefd[2];
	FILE *fs_write, *fs_read;
	SerializerIO writer, reader;
	_create_pipe_io(pipefd, &fs_write, &fs_read, &writer, &reader);

	EdgeID ids[1] = {0};
	SerializerIO_WriteBuffer(writer, ids, sizeof(ids));
	_close_writer(pipefd, fs_write, &writer);

	Graph *g = Graph_New(NODE_CAP, EDGE_CAP);
	Graph_AcquireWriteLock(g);

	// create 2 nodes and 1 edge so edge slot 0 exists
	Node n;
	Edge e;
	for(int i = 0; i < 2; i++) {
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
// test list
//------------------------------------------------------------------------------

TEST_LIST = {
	{"deletedNodes_misaligned_buffer",  test_deletedNodes_misaligned_buffer},
	{"deletedEdges_misaligned_buffer",  test_deletedEdges_misaligned_buffer},
	{"deletedNodes_single_byte",        test_deletedNodes_single_byte_buffer},
	{"deletedNodes_count_mismatch",     test_deletedNodes_count_mismatch},
	{"deletedEdges_count_mismatch",     test_deletedEdges_count_mismatch},
	{"deletedNodes_aligned_buffer",     test_deletedNodes_aligned_buffer},
	{"deletedEdges_aligned_buffer",     test_deletedEdges_aligned_buffer},
	{NULL, NULL}
};
