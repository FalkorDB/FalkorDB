/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include <string.h>
#include <stdlib.h>
#include "run_redis_command_as.h"


char *ADMIN_USER = NULL;

// let user ovveride the default admin user
void init_run_cmd_as
(
	RedisModuleCtx *ctx
) {
	const char *admin_user = getenv("GRAPH_ADMIN_USER");
	if (admin_user != NULL) {
		ADMIN_USER = strdup(admin_user);
	}else{
		ADMIN_USER = strdup("default");
	}
}

void free_run_cmd_as
(
) {
	ASSERT(ADMIN_USER != NULL);
	free(ADMIN_USER);
}

// authenticate as a 'username' user
static int _switch_user
(
	RedisModuleCtx *ctx,
	const char *username,
	uint64_t *client_id
) {
	size_t username_len = strlen(username);
	if (RedisModule_AuthenticateClientWithACLUser(ctx, username, username_len,
		NULL, ctx, client_id) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "error", "Failed to authenticate as %s", username);
		return REDISMODULE_ERR;
	}
	return REDISMODULE_OK;
}

int run_redis_command_as(RedisModuleCtx *ctx, RedisModuleString **argv,
	int argc, RedisCommandAsUserFunc cmd, const char *username, void *privdata){
	RedisModuleString *_redis_current_user_name = 
		RedisModule_GetCurrentUserName(ctx);
	const char *redis_current_user_name = 
		RedisModule_StringPtrLen(_redis_current_user_name, NULL);
	
	uint64_t client_id = 0;
	if (_switch_user(ctx, username, &client_id) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "error", "Failed to authenticate as user %s", username);
		RedisModule_ReplyWithError(ctx, "FAILED");
		RedisModule_FreeString(ctx, _redis_current_user_name);
		return REDISMODULE_ERR;
    }
	
	int res = cmd(ctx, argv, argc, redis_current_user_name, privdata);

	if (_switch_user(ctx, redis_current_user_name, NULL) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "error", "Failed to authenticate back as user %s", redis_current_user_name);
		RedisModule_DeauthenticateAndCloseClient(ctx, client_id);
		RedisModule_FreeString(ctx, _redis_current_user_name);
		return REDISMODULE_ERR;
	}

	RedisModule_FreeString(ctx, _redis_current_user_name);
	return res;
}

