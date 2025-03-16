/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/


#include <string.h>
#include "run_redis_command_as.h"

const char *ADMIN_USER = "default";
//todo maybe scan for a user with @admin or @all using redis acl

// authenticate as a 'username' user
static int _switch_user
(
	RedisModuleCtx *ctx,
	const char *username
) {
    size_t username_len = strlen(username);
	if (RedisModule_AuthenticateClientWithACLUser(ctx, username, username_len,
		NULL, ctx, NULL) != REDISMODULE_OK) {
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
	
	if (_switch_user(ctx, username) != REDISMODULE_OK) {
        RedisModule_Log(ctx, "error", "Failed to authenticate as user %s", username);
		RedisModule_ReplyWithError(ctx, "FAILED");
        return REDISMODULE_ERR;
    }
	
	int res = cmd(ctx, argv, argc, redis_current_user_name, privdata);

	if (_switch_user(ctx, redis_current_user_name) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "error", "Failed to authenticate back as user %s", redis_current_user_name);
		//@todo how do we close the connection the user is admin now. disconnect the client
		return REDISMODULE_ERR;
	}
	return res;
}

