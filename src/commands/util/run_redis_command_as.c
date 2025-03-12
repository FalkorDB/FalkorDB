/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/


#include "RG.h"
#include <string.h>
#include "run_redis_command_as.h"

const char *INTERNAL_ADMIN_NAME = "GRAPH.INTERNAL.ADMIN";

static RedisModuleUser *_create_internal_admin
(
	RedisModuleCtx *ctx
) {
	ASSERT(ctx != NULL);

	RedisModuleUser *internal_user = RedisModule_CreateModuleUser(INTERNAL_ADMIN_NAME);
	
	if (internal_user == NULL) {
		RedisModule_Log(ctx, "error", "Failed to create internal user %s",
			INTERNAL_ADMIN_NAME);
		return NULL;
	}

	// set the user as admin
	if (RedisModule_SetModuleUserACL(internal_user, "@all") != REDISMODULE_OK) {
		RedisModule_Log(ctx, "error", "Failed to set user %s as admin",
			INTERNAL_ADMIN_NAME);
			RedisModule_FreeModuleUser(internal_user);
		return NULL;
	}

	return internal_user;
}


// authenticate as a 'username' user
static int _switch_user
(
	RedisModuleCtx *ctx,
	const char *username,
	uint64_t *client_id
) {
	ASSERT(ctx != NULL);
	ASSERT(username != NULL);

    size_t username_len = strlen(username);
	if (RedisModule_AuthenticateClientWithACLUser(ctx, username, username_len,
		NULL, ctx, client_id) != REDISMODULE_OK) {
        RedisModule_Log(ctx, "error", "Failed to authenticate as %s", username);
        return REDISMODULE_ERR;
    }
	return REDISMODULE_OK;
}

int run_redis_command_as_graph_internal_admin
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc,
	RedisCommandAsUserFunc cmd,
	void *privdata
) {
	ASSERT(ctx != NULL);
	ASSERT(argv != NULL);
	ASSERT(argc > 0);
	ASSERT(cmd != NULL);

	RedisModuleString *_redis_current_user_name = 
		RedisModule_GetCurrentUserName(ctx);

	const char *redis_current_user_name = 
		RedisModule_StringPtrLen(_redis_current_user_name, NULL);

	RedisModuleUser *admin_user = _create_internal_admin(ctx);
	if (admin_user == NULL) {
		RedisModule_Log(ctx, "error", "Failed to create internal admin user");
		RedisModule_ReplyWithError(ctx, "FAILED");
		return REDISMODULE_ERR;
	}

	uint64_t client_id = 0;

	if (_switch_user(ctx, INTERNAL_ADMIN_NAME, &client_id) != REDISMODULE_OK) {
        RedisModule_Log(ctx, "error", "Failed to authenticate as user %s",
			INTERNAL_ADMIN_NAME);
		RedisModule_ReplyWithError(ctx, "FAILED");
        return REDISMODULE_ERR;
    }
	
	int res = cmd(ctx, argv, argc, redis_current_user_name, privdata);

	if (_switch_user(ctx, redis_current_user_name, NULL) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "error", "Failed to authenticate back as user %s",
			redis_current_user_name);
		// try to close the connection the user is admin now.
		// disconnect the client, anyway we are freeing the admin user
		// so this should disconnect all clients still connected to it.
		// we dont have to reply because cmd should have replied
		RedisModule_DeauthenticateAndCloseClient(ctx, client_id);
		RedisModule_FreeModuleUser(admin_user);
		return REDISMODULE_ERR;
	}

	RedisModule_FreeModuleUser(admin_user);
	return res;
}
