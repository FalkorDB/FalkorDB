/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "../redismodule.h"
#include "../module_event_handlers.h"

#include <string.h>

void ModuleEventHandler_AUXBeforeKeyspaceEvent(void);
void ModuleEventHandler_AUXAfterKeyspaceEvent(void);

extern uint aux_field_counter;

static void Debug_AUX(RedisModuleString **argv, int argc) {
	if(argc < 2) return;

	const char *arg = RedisModule_StringPtrLen(argv[1], NULL);

	if(strcmp(arg, "START") == 0) {
		ModuleEventHandler_AUXBeforeKeyspaceEvent();
	} else if(strcmp(arg, "END") == 0) {
		ModuleEventHandler_AUXAfterKeyspaceEvent();
	}
}

int Graph_Debug(RedisModuleCtx *ctx, RedisModuleString **argv, int argc) {
	ASSERT(ctx != NULL);
	RedisModule_ReplicateVerbatim(ctx);

	if(strcmp(RedisModule_StringPtrLen(argv[1], NULL), "AUX") == 0) {
		Debug_AUX(argv + 1, argc - 1);
	}

	RedisModule_ReplyWithLongLong(ctx, aux_field_counter);

	return REDISMODULE_OK;
}

