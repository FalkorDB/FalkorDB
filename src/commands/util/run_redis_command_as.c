/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include <string.h>
#include <stdlib.h>
#include "run_redis_command_as.h"


char *ADMIN_USER = NULL;

// initialize the impersonate mechanism
// this function will be called once at the beginning of the module
// it will read the environment variable GRAPH_ADMIN_USER
// and set the ADMIN_USER variable to the value of the environment variable
// if the environment variable is not set, the default value will be 'default'
void init_run_cmd_as
(
	RedisModuleCtx *ctx  // redis module context
) {
	ASSERT(ctx       != NULL);
	ASSERT(ADMIN_USER == NULL);

	const char *admin_user = getenv("GRAPH_ADMIN_USER");
	if (admin_user != NULL) {
		ADMIN_USER = strdup(admin_user);
	}else{
		ADMIN_USER = strdup("default");
	}
}



// switch the user to the given username
// the function will authenticate the user with the given username
// and return the client id of the new user
// the function will return REDISMODULE_OK on success
// and REDISMODULE_ERR on failure
static int _switch_user
(
	RedisModuleCtx *ctx,   // redis module context
	const char *username,  // the username to switch to 
	uint64_t *client_id    // the client id of the new user, output parameter
) {
	size_t username_len = strlen(username);
	if(RedisModule_AuthenticateClientWithACLUser(ctx, username, username_len,
		NULL, ctx, client_id) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "error", "Failed to authenticate as %s", username);
		return REDISMODULE_ERR;
	}
	return REDISMODULE_OK;
}

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
) {

	ASSERT(ctx      != NULL);
	ASSERT(cmd      != NULL);
	ASSERT(argc     > 0);
	ASSERT(argv     != NULL);
	ASSERT(username != NULL);

	RedisModuleString *_redis_current_user_name = 
		RedisModule_GetCurrentUserName(ctx);
	const char *redis_current_user_name = 
		RedisModule_StringPtrLen(_redis_current_user_name, NULL);
	
	uint64_t client_id = 0;
	if(_switch_user(ctx, username, &client_id) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "error", "Failed to authenticate as user %s", username);
		RedisModule_ReplyWithError(ctx, "FAILED");
		RedisModule_FreeString(ctx, _redis_current_user_name);
		return REDISMODULE_ERR;
    }
	
	int res = cmd(ctx, argv, argc, redis_current_user_name, privdata);

	if(_switch_user(ctx, redis_current_user_name, NULL) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "error", "Failed to authenticate back as user %s", redis_current_user_name);
		RedisModule_DeauthenticateAndCloseClient(ctx, client_id);
		RedisModule_FreeString(ctx, _redis_current_user_name);
		return REDISMODULE_ERR;
	}

	RedisModule_FreeString(ctx, _redis_current_user_name);
	return res;
}

// free the impersonate mechanism
// this function will be called once at the end of the module
// it will free the ADMIN_USER variable
// the function should be called only once
void free_run_cmd_as(void) {
	ASSERT(ADMIN_USER != NULL);
	free(ADMIN_USER);
	ADMIN_USER = NULL;
}