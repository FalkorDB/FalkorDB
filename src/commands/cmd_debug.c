/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../redismodule.h"
#include "../util/rmalloc.h"
#include "../util/thpool/pools.h"
#include "../module_event_handlers.h"

#include <string.h>
#include <sys/mman.h>

void ModuleEventHandler_AUXAfterKeyspaceEvent(void);
void ModuleEventHandler_AUXBeforeKeyspaceEvent(void);

extern uint aux_field_counter;

static void Debug_AUX
(
	RedisModuleString **argv,
	int argc
) {
	if(argc < 2) return;

	const char *arg = RedisModule_StringPtrLen(argv[1], NULL);

	if(strcmp(arg, "START") == 0) {
		ModuleEventHandler_AUXBeforeKeyspaceEvent();
	} else if(strcmp(arg, "END") == 0) {
		ModuleEventHandler_AUXAfterKeyspaceEvent();
	}
}

// crash the server simulating an out-of-memory error
static void _Debug_OOM
(
	void *args  // unused
) {
	void *ptr = rm_malloc(SIZE_MAX/2); // should trigger an out of memory
	rm_free(ptr);
}

// crash the server with sigsegv
static void _Debug_SegFault
(
	void *args  // unused
) {
	// compiler gives warnings about writing to a random address
	// e.g "*((char*)-1) = 'x';"
	// as a workaround, we map a read-only area
	// and try to write there to trigger segmentation fault
	char* p = mmap(NULL, 4096, PROT_READ, MAP_PRIVATE | MAP_ANON, -1, 0);
	*p = 'x';
}

// crash by assertion failed
static void _Debug_ASSERT
(
	void *args  // unused
) {
	RedisModule_Assert(false && "DEBUG ASSERT");
}

int Graph_Debug
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT(ctx != NULL);

	if (argc < 2) {
		return RedisModule_WrongArity(ctx);
	}

	const char *sub_cmd = RedisModule_StringPtrLen(argv[1], NULL);

	if (!strcasecmp(sub_cmd, "AUX")) {
		Debug_AUX(argv + 1, argc - 1);
		RedisModule_ReplyWithLongLong(ctx, aux_field_counter);
		RedisModule_ReplicateVerbatim(ctx);
	}

	else if (!strcasecmp(sub_cmd, "OOM")) {
		// crash the server simulating an out-of-memory error
		ThreadPools_AddWorkReader(_Debug_OOM, NULL, true);
		RedisModule_ReplyWithCString(ctx, "OK");
	}

	else if (!strcasecmp(sub_cmd, "ASSERT")) {
		// crash by assertion failed
		ThreadPools_AddWorkReader(_Debug_ASSERT, NULL, true);
		RedisModule_ReplyWithCString(ctx, "OK");
	}

	else if (!strcasecmp(sub_cmd, "SEGFAULT")) {
		// crash the server with sigsegv
		ThreadPools_AddWorkReader(_Debug_SegFault, NULL, true);
		RedisModule_ReplyWithCString(ctx, "OK");
	}

	else {
		RedisModule_ReplyWithError(ctx, "ERR unknown subcommand or wrong number of arguments");
	}

	return REDISMODULE_OK;
}

