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
	RedisModuleString **argv,
	int argc,
	const char *username,
	char * password
);

// call to set password with a ">" prefix, indicating that this password
// should be add to the set of stored passwords for the current user
static int _add_password_fun
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc,
	const char *username,
	void *privdata
);

// call to set password with a "<" prefix
// that will remove 'password' from the current user passwords 
// or return error if password is not found
static int _remove_password_fun
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc,
	const char *username,
	void *privdata
);

// a helper function that get a prefix char and call the set password function
// with the password prefixed with the given char, this is the syntax for redis
// add remove password for a user '<' used to remove password,
//  '>' to add password
static int _set_password_with_prefix
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc,
	const char *username,
	const char prefix,
	void *privdata
);

// add or remove password for current user
// Examples:
// GRAPH.PASSWORD ADD <password>
// GRAPH.PASSWORD REMOVE <password>
int graph_password_cmd
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT(ctx != NULL);
	ASSERT(argv != NULL);

	if(argc != 3) {
		return RedisModule_WrongArity(ctx);
	}

 	const char *command = RedisModule_StringPtrLen(argv[1], NULL);
	
	if (strcasecmp(command, "ADD") == 0) {
		return run_redis_command_as(ctx, argv, argc, _add_password_fun,
		ADMIN_USER, NULL);
	} else if (strcasecmp(command, "REMOVE") == 0) {
		return run_redis_command_as(ctx, argv, argc, _remove_password_fun,
		ADMIN_USER, NULL);
	} else {
		RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING, 
		"Unknown command: GRAPH.PASSWORD %s, passible commands are [ADD, REMOVE]",
		command);
		RedisModule_ReplyWithError(ctx, "Failed");
		return REDISMODULE_ERR;
	}
		
	return REDISMODULE_OK;
}

// call to set password with a ">" prefix, indicating that this password
// should be add to the set of stored passwords for the current user
static int _add_password_fun
(
	RedisModuleCtx *ctx,
 	RedisModuleString **argv,
	int argc,
	const char *username,
	void *privdata
) {
	return _set_password_with_prefix(ctx, argv, argc, username, '>', privdata);
}

// call to set password with a "<" prefix
// that will remove 'password' from the current user passwords 
// or return error if password is not found
static int _remove_password_fun
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc,
	const char *username,
	void *privdata
) {
	return _set_password_with_prefix(ctx, argv, argc, username, '<', privdata);
}

// a helper function that get a prefix char and call the set password function
// with the password prefixed with the given char, this is the syntax for redis
// add remove password for a user '<' used to remove password,
//  '>' to add password
static int _set_password_with_prefix
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc,
	const char *username,
	const char prefix,
	void       *privdata
) {
	RedisModuleString *password = argv[2];
	size_t passwordStrLen;
	const char *passwordStr = 
		RedisModule_StringPtrLen(password, &passwordStrLen);
	char *passwordBuff = RedisModule_Alloc(passwordStrLen + 2);
	sprintf(passwordBuff, "%c%s", prefix, passwordStr);

	int ret = _set_password_fun(ctx, argv, argc, username, passwordBuff);
	
	RedisModule_Free(passwordBuff);
	return ret;
}

// assume passwrod is prefixed with "<" or ">", 
// either add password or remove password from the current user
static int _set_password_fun
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc,
	const char *username,
	char *password
) {
	ASSERT(ctx      != NULL);
	ASSERT(argv     != NULL);
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
		RedisModule_ReplyWithString(ctx, 
			RedisModule_CreateStringFromCallReply(reply));
	}

	RedisModule_FreeCallReply(reply);
	return ret;
}

