/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include "../../redismodule.h"

// this variable will be set to the value of the environment variable
// $GRAPH_ADMIN_USER
// if the environment variable is not set, the default value will be 'default'
// this variable will be used to run the command as admin user
extern char *ACL_ADMIN_USER;

// this function will be used to run the code in the context of admin user
// the function will be called with the following parameters:
// ctx - the redis module context
// argv - the arguments to call
// argc - the number of arguments
// privdata - the private data to pass to the function, optional
typedef int (*ACLFunction)(RedisModuleCtx *ctx,
	RedisModuleString **argv, int argc, void *privdata);

// initialize the impersonation mechanism
// this function will be called once at the beginning of the module load
// it will read the environment variable $GRAPH_ADMIN_USER
// and set the AC_ADMIN_USER variable to the value of the environment variable
// if the environment variable is not set, the default value will be 'default'
void init_acl_admin_username
(
	RedisModuleCtx *ctx  // redis module context
);

// check if the current instance is a replica
// returns 1 if replica, 0 otherwise
int is_replica
(
	RedisModuleCtx *ctx  // redis module context
);

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
	ACLFunction cmd,           // the function to call
	const char *username,      // the username to switch to
	void *privdata             // optional private data
);

// free the impersonation mechanism
// this function will be called once on module unload
// it will free the ACL_ADMIN_USER variable
// the function should be called only once
void free_run_cmd_as(void); 

