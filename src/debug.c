/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <stdio.h>
#include <signal.h>
#include <pthread.h>
#include <sys/types.h>
#include "RG.h"
#include "globals.h"
#include "util/thpool/pool.h"
#include "commands/cmd_context.h"

void InfoFunc
(
	RedisModuleInfoCtx *ctx,
	int for_crash_report
) {
	// make sure information is requested for crash report
	if(!for_crash_report) return;

	// #workers + Redis main thread
	uint32_t n = ThreadPool_ThreadCount() + 1;
	CommandCtx* commands[n];
	Globals_GetCommandCtxs(commands, &n);

	RedisModule_InfoAddSection(ctx, "executing commands");

	for(int i = 0; i < n; i++) {
		CommandCtx *cmd = commands[i];
		ASSERT(cmd != NULL);

		int rc __attribute__((unused));
		char *command_desc = NULL;
		rc = asprintf(&command_desc, "%s %s", cmd->command_name, cmd->query);
		RedisModule_InfoAddFieldCString(ctx, "command", command_desc);

		free(command_desc);
		CommandCtx_Free(cmd);
	}
}

void setupCrashHandlers
(
	RedisModuleCtx *ctx
) {
	int registered = RedisModule_RegisterInfoFunc(ctx, InfoFunc);
	ASSERT(registered == REDISMODULE_OK);
}

