/*
 * Copyright Redis Ltd. 2018 - present
 * Copyright FalkorDB Ltd. 2024 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../query_ctx.h"
#include "execution_ctx.h"
#include "cmd_bulk_insert.h"

//------------------------------------------------------------------------------
// Module Commands
//------------------------------------------------------------------------------

typedef enum {
	CMD_UNKNOWN     = 0,
	CMD_QUERY       = 1,
	CMD_RO_QUERY    = 2,
	CMD_DELETE      = 3,
	CMD_CONFIG      = 4,
	CMD_EXPLAIN     = 5,
	CMD_PROFILE     = 6,
	CMD_BULK_INSERT = 7,
	CMD_SLOWLOG     = 8,
	CMD_LIST        = 9,
	CMD_DEBUG       = 10,
	CMD_INFO        = 11,
	CMD_EFFECT      = 12,
	CMD_COPY        = 13,
	CMD_RESTORE     = 14,
	CMD_ACL         = 15,
	CMD_PASSWORD    = 16,
	CMD_MEMORY      = 17,
	CMD_EVAL        = 18,
	CMD_UDF         = 19
} GRAPH_Commands;

//------------------------------------------------------------------------------
// graph commands
//------------------------------------------------------------------------------

GRAPH_Commands CommandFromString(const char *cmd_name);

void Graph_Query(void *args);
void Graph_Profile(void *args);
void Graph_Explain(void *args);

int Graph_List        (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_Info        (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_Copy        (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_Eval        (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_Debug       (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_Delete      (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_Effect      (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_Config      (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_Restore     (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_Slowlog     (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_Memory      (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int CommandDispatch   (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_Constraint  (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int graph_acl_cmd     (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_SetPassword (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
int Graph_RegisterUDF (RedisModuleCtx *ctx, RedisModuleString **argv, int argc);
