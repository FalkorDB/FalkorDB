/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "../globals.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "./util/run_redis_command_as.h"

#include <assert.h>
#include <stdbool.h>

// GRAPH.PASSWORD ADD <password>
// GRAPH.PASSWORD REMOVE <password>

// assume passwrod is prefixed with "<" or ">", 
// either add password or remove password from the current user
static int _set_password_fun
(
	RedisModuleCtx *ctx,
	const char *username,
	char *password
) {
	ASSERT(ctx      != NULL);
	ASSERT(password != NULL);
	ASSERT(username != NULL);
	ASSERT(password[0] == '<' || password[0] == '>');

	RedisModuleCallReply *reply = 
		RedisModule_Call(ctx, "ACL", "ccc", "SETUSER", username, password);
	
	int ret = REDISMODULE_OK;

    if (reply == NULL 
		|| RedisModule_CallReplyType(reply) != REDISMODULE_REPLY_STRING) {
		RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING,
			"Failed to set password for user %s.", username);

		RedisModule_ReplyWithError(ctx, "FAILED");
		
		ret = REDISMODULE_ERR;
	} else {
		RedisModuleString *reply_str =
			RedisModule_CreateStringFromCallReply(reply);
		if(reply_str == NULL) {
			RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING,
				"Failed to set password for user %s.", username);
			RedisModule_ReplyWithError(ctx, "FAILED");
			ret = REDISMODULE_ERR;
		} else {
			RedisModule_ReplyWithString(ctx, reply_str);
			RedisModule_FreeString(ctx, reply_str);
		}
	}

	RedisModule_FreeCallReply(reply);
	return ret;
}

// a helper function that get a prefix char and call the set password function
// with the password prefixed with the given char, this is the syntax for redis
// add/remove password for a user '<' used to remove password,
//  '>' to add password
static int _set_password_with_prefix
(
	RedisModuleCtx *ctx,            // redis module context
	const char *username,           // username to set password for
	const RedisModuleString *pass,  // password that should be added / removed
	const char prefix               // prefix that will be added to the password

) {
	ASSERT(ctx      != NULL);
	ASSERT(pass     != NULL);
	ASSERT(username != NULL);
	
	size_t passwordStrLen;
	const char *passwordStr = RedisModule_StringPtrLen(pass, &passwordStrLen);

	char *passwordBuff = NULL;
	int n = asprintf(&passwordBuff, "%c%s", prefix, passwordStr);
	assert (n > 0) ;

	int ret = _set_password_fun(ctx, username, passwordBuff);

	free(passwordBuff);
	return ret;
}

// call to set password command with a ">" prefix, indicating that this password
// should be added to the set of stored passwords for the current user
// implement RedisFunc
static int _add_password
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // arguments to the command (RedisFunc)
	int argc,                  // number of args (RedisFunc)
	void *privdata             // private data, the current username
) {
	ASSERT(ctx      != NULL);
	ASSERT(argv     != NULL);
	ASSERT(argc     > 0);
	ASSERT(privdata != NULL);

	const RedisModuleString *pass = (RedisModuleString *)argv[2];
	return _set_password_with_prefix(ctx, (const char*)privdata, pass, '>');
}

// call to set password command with a "<" prefix
// that will remove 'password' from the current user passwords 
// or return error if password is not found
// implement RedisFunc
static int _remove_password
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // arguments to the command (RedisFunc)
	int argc,                  // number of args (RedisFunc)
	void *privdata             // private data, the current username
) {
	ASSERT(ctx      != NULL);
	ASSERT(argv     != NULL);
	ASSERT(argc     > 0);
	ASSERT(privdata != NULL);

	const RedisModuleString *pass = (RedisModuleString *)argv[2];
	return _set_password_with_prefix(ctx, (const char*) privdata, pass, '<');
}

// add or remove password for current user
// examples:
// GRAPH.PASSWORD ADD <password>
// GRAPH.PASSWORD REMOVE <password>
int Graph_SetPassword
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command args
	int argc                   // number of args
) {
	ASSERT(ctx  != NULL);
	ASSERT(argv != NULL);
	ASSERT(argc > 0);

	// expecting 3 arguments
	if(argc != 3) {
		return RedisModule_WrongArity(ctx);
	}

	// get the current user name as C string, to pass as private data
	RedisModuleString *_redis_current_user_name = 
		RedisModule_GetCurrentUserName(ctx);

	const char *username = 
		RedisModule_StringPtrLen(_redis_current_user_name, NULL);

	// get the action ADD / REMOVE
 	const char *action = RedisModule_StringPtrLen(argv[1], NULL);
	
	ACLFunction f = NULL;
	if(strcasecmp(action, "ADD") == 0) {
		f = _add_password;
	} else if(strcasecmp(action, "REMOVE") == 0) {
		f = _remove_password;
	} else {
		RedisModule_FreeString(ctx, _redis_current_user_name);
		RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING, 
		"Unknown command: GRAPH.PASSWORD %s, passible commands are [ADD, REMOVE]",
		action);

		RedisModule_ReplyWithError(ctx, "Unknown sub-command");
		return REDISMODULE_ERR;
	}

	int ret = run_acl_function_as(ctx, argv, argc, f, ACL_ADMIN_USER,
		 (void*) username);

	RedisModule_FreeString(ctx, _redis_current_user_name);
	return ret;
}

