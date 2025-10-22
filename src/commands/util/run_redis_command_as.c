/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include <string.h>
#include <stdlib.h>
#include "redismodule.h"
#include "run_redis_command_as.h"

char *ACL_ADMIN_USER = "default";  // the default username

// initialize the impersonation mechanism
// this function will be called once at the beginning of the module load
// it will read the environment variable $GRAPH_ADMIN_USER
// and set the AC_ADMIN_USER variable to the value of the environment variable
// if the environment variable is not set, the default value will be 'default'
void init_acl_admin_username
(
	RedisModuleCtx *ctx  // redis module context
) {
	ASSERT(ctx != NULL);

	// see if GRAPH_ADMIN_USER is specified in env var
	const char *admin_user = getenv("GRAPH_ADMIN_USER");
	if(admin_user != NULL) {
		// replace default ACL_ADMIN_USER
		ACL_ADMIN_USER = strdup(admin_user);
	}
}

// switch the current user to the given username
// the function will authenticate the user with the given username
// and return the client id of the new user
// the function will return REDISMODULE_OK on success
// and REDISMODULE_ERR on failure
static int _switch_user
(
	RedisModuleCtx *ctx,   // redis module context
	const char *username,  // the username to switch to 
	uint64_t *client_id    // [output] the client id of the new user
) {
	ASSERT(ctx      != NULL);
	ASSERT(username != NULL);

	size_t username_len = strlen(username);

	if(RedisModule_AuthenticateClientWithACLUser(ctx, username, username_len,
		NULL, ctx, client_id) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "error", "Failed to authenticate as %s", username);

		return REDISMODULE_ERR;
	}

	return REDISMODULE_OK;
}

int is_replica(RedisModuleCtx *ctx) {
    int flags = RedisModule_GetContextFlags(ctx);
    return (flags & REDISMODULE_CTX_FLAGS_SLAVE) != 0;
}

// run the given acl function as the given user
// the function will switch the user to the given username
// and run the given acl function with the given arguments
// the function will switch back to the original user after the 
// command is executed
// the function will return REDISMODULE_OK on success
// and REDISMODULE_ERR on failure
int run_acl_function_as
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // the arguments to call
	int argc,                  // the number of arguments
	ACLFunction cmd,           // the command to call
	const char *username,      // the username to switch to
	void *privdata             // optional private data
) {
	ASSERT(ctx      != NULL);
	ASSERT(cmd      != NULL);
	ASSERT(argc     > 0);
	ASSERT(argv     != NULL);
	ASSERT(username != NULL);

	// if running on replica, skip user switching and just execute the command
	if(is_replica(ctx)) {
		return cmd(ctx, argv, argc, privdata);
	}

	// get current user
	RedisModuleString *_redis_current_user_name = 
		RedisModule_GetCurrentUserName(ctx);

	ASSERT(_redis_current_user_name != NULL);

	const char *redis_current_user_name = 
		RedisModule_StringPtrLen(_redis_current_user_name, NULL);
		
	// try switching user
	uint64_t client_id = 0;
	if(_switch_user(ctx, username, &client_id) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "error", "Failed to authenticate as user %s",
				username);

		RedisModule_ReplyWithError(ctx, "FAILED");

		RedisModule_FreeString(ctx, _redis_current_user_name);

		return REDISMODULE_ERR;
    }
	
	// managed to swtich, run function under new user
	int res = cmd(ctx, argv, argc, privdata);

	// restore original user
	if(_switch_user(ctx, redis_current_user_name, NULL) != REDISMODULE_OK) {
		RedisModule_Log(ctx, "error", "Failed to authenticate back as user %s",
				redis_current_user_name);

		RedisModule_DeauthenticateAndCloseClient(ctx, client_id);

		RedisModule_FreeString(ctx, _redis_current_user_name);

		return REDISMODULE_ERR;
	}

	if(_redis_current_user_name != NULL) {
		RedisModule_FreeString(ctx, _redis_current_user_name);
	}

	return res;
}

// free the impersonation mechanism
// this function will be called once on module unload
// it will free the ACL_ADMIN_USER variable
// the function should be called only once
void free_run_cmd_as(void) {
	ASSERT(ACL_ADMIN_USER != NULL);

	if(strcmp(ACL_ADMIN_USER, "default") != 0) {
		free(ACL_ADMIN_USER);
	}

	ACL_ADMIN_USER = NULL;
}

