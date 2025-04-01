/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "../../redismodule.h"

extern char *ADMIN_USER;

typedef int (*RedisCommandAsUserFunc)(RedisModuleCtx *ctx,
	RedisModuleString **argv, int argc, const char *username, void *privdata);

// initialize the impersonate mechanism
// this function will be called once at the beginning of the module
// it will read the environment variable GRAPH_ADMIN_USER
// and set the ADMIN_USER variable to the value of the environment variable
// if the environment variable is not set, the default value will be 'default'
void init_run_cmd_as
(
	RedisModuleCtx *ctx  // redis module context
);

// run the given command as the given user
// the function will switch the user to the given username
// and run the command with the given arguments
// the function will switch back to the original user after the 
// command is executed
// the function will return REDISMODULE_OK on success
// and REDISMODULE_ERR on failure
int run_redis_command_as
(
	RedisModuleCtx *ctx,         // redis module context
	RedisModuleString **argv,    // the arguments to call
	int argc,                    // the number of arguments
	RedisCommandAsUserFunc cmd,  // the command to call
	const char *username,        // the username to switch to
	void *privdata               // optional private data
);

// free the impersonate mechanism
// this function will be called once at the end of the module
// it will free the ADMIN_USER variable
// the function should be called only once
void free_run_cmd_as(void); 