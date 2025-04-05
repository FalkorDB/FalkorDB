/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#pragma once

#include <stdbool.h>
#include "../redismodule.h"

// initializes the ACL command by reading environment variables 
// 'ACL_GRAPH_READONLY_USER', 'ACL_GRAPH_ADMIN' and 'ACL_GRAPH_USER'
// and build the corrisponding CommandCategory structure for each
// the environment variables should contain space-separated lists of commands
// for example: SET ACL_GRAPH_USER = "INFO CLIENT DBSIZE PING HELLO AUTH"
// if one of the environment variables is not set or its value is "false", 
// the entire GRAPH.ACL command is disabled
//
// returns REDISMODULE_OK if the ACL initialization was successful
// indicating that the GRAPH.ACL command should be activated
int init_cmd_acl
(
	RedisModuleCtx *ctx  // redis module context
);

// free the resources consumed by the ACL command
void free_cmd_acl(void); 

