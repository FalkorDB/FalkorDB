/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include <stdbool.h>
#include "../globals.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "./util/run_redis_command_as.h"

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

	RedisModuleCallReply *reply = 
		RedisModule_Call(ctx, "ACL", "ccc", "SETUSER", username, password);
	
	int ret = REDISMODULE_OK;

    if (reply == NULL 
		|| RedisModule_CallReplyType(reply) != REDISMODULE_REPLY_STRING) {
		RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING,
			"Failed to set passwordfor user %s.", username);

		RedisModule_ReplyWithError(ctx, "FAILED");
		
		ret = REDISMODULE_ERR;
	} else {
		RedisModuleString *reply_str = RedisModule_CreateStringFromCallReply(reply);
		if (reply_str == NULL) {
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
// add / remove password for a user '<' used to remove password,
//  '>' to add password
static int _set_password_with_prefix
(
	RedisModuleCtx *ctx,       //
	RedisModuleString *pass,  //
	const char *username,      //
	const char prefix,         //
	void       *privdata       //
) {
	size_t passwordStrLen;
	const char *passwordStr = 
		RedisModule_StringPtrLen(pass, &passwordStrLen);
	// TODO: user redis string formatting
	char *passwordBuff = RedisModule_Alloc(passwordStrLen + 2);
	snprintf(passwordBuff, passwordStrLen + 2, "%c%s", prefix, passwordStr);

	int ret = _set_password_fun(ctx, username, passwordBuff);
	
	RedisModule_Free(passwordBuff);
	return ret;
}

// call to set password command with a ">" prefix, indicating that this password
// should be added to the set of stored passwords for the current user
static int _add_password
(
	RedisModuleCtx *ctx,       // redis module context
 	const RedisModuleString *pass,  // 
	const char *username,      // set password for this user
	void *privdata             // private data
) {
	return _set_password_with_prefix(ctx, pass, username, '>', privdata);
}

// call to set password command with a "<" prefix
// that will remove 'password' from the current user passwords 
// or return error if password is not found
static int _remove_password
(
	RedisModuleCtx *ctx,       // redis module context
 	const RedisModuleString *pass,  // 
	int argc,                  // number of arguments
	const char *username,      // remove password from user
	void *privdata             // private data
) {
	return _set_password_with_prefix(ctx, pass, username, '<', privdata);
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

	// get the action ADD / REMOVE
 	const char *action = RedisModule_StringPtrLen(argv[1], NULL);
	
	RedisFunc f = NULL;
	if(strcasecmp(action, "ADD") == 0) {
		f = _add_password;
	} else if(strcasecmp(action, "REMOVE") == 0) {
		f = _remove_password;
	} else {
		RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING, 
		"Unknown command: GRAPH.PASSWORD %s, passible commands are [ADD, REMOVE]",
		action);

		RedisModule_ReplyWithError(ctx, "Unknown sub-command");
		return REDISMODULE_ERR;
	}

	return run_redis_command_as(ctx, argv, argc, f, ADMIN_USER, NULL);
}

