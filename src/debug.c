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

static struct sigaction old_act;

static void startCrashReport(void) {
	RedisModule_Log(NULL, "warning", "=== REDISGRAPH BUG REPORT START: ===");
}

static void endCrashReport(void) {
	RedisModule_Log(NULL, "warning", "=== REDISGRAPH BUG REPORT END. ===");
}

static void logCommands(void) {
	// #readers + Redis main thread
	uint32_t n = ThreadPool_ThreadCount() + 1;
	CommandCtx* commands[n];
	Globals_GetCommandCtxs(commands, &n);

	for(uint32_t i = 0; i < n; i++) {
		CommandCtx *cmd = commands[i];
		ASSERT(cmd != NULL);

		RedisModule_Log(NULL, "warning", "%s %s", cmd->command_name,
				cmd->query);

		CommandCtx_Free(cmd);
	}
}

void InfoFunc
(
	RedisModuleInfoCtx *ctx,
	int for_crash_report
) {
	// make sure information is requested for crash report
	if(!for_crash_report) return;

	// pause all working threads
	// NOTE: pausing is not an atomic action;
	// other threads can potentially change states before being interrupted.
	ThreadPool_Pause();

	// #readers + Redis main thread
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

void crashHandler
(
	int sig,
	siginfo_t *info,
	void *ucontext
) {
	// pause all working threads
	// NOTE: pausing is an async operation
	ThreadPool_Pause();

	startCrashReport();

	// log currently executing GRAPH commands
	logCommands();

	endCrashReport();

	// call previous (Redis original) handler
	(*old_act.sa_sigaction)(sig, info, ucontext);
}

void setupCrashHandlers
(
	RedisModuleCtx *ctx
) {
	// if RedisModule_RegisterInfoFunc is available use it
	// to report RedisGraph additional information in case of a crash
	// otherwise overwrite Redis signal handler

	if(RedisModule_RegisterInfoFunc) {
		int registered = RedisModule_RegisterInfoFunc(ctx, InfoFunc);
		ASSERT(registered == REDISMODULE_OK);
	} else {
		// RegisterInfoFunc is not available, replace redis
		// SIGSEGV signal handler

		struct sigaction act;

		sigemptyset(&act.sa_mask);
		act.sa_flags = SA_NODEFER | SA_RESETHAND | SA_SIGINFO;
		act.sa_sigaction = crashHandler;

		sigaction(SIGSEGV, &act, &old_act);
	}
}

