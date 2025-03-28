#pragma once
#include <stdbool.h>
#include "../redismodule.h"


int init_cmd_acl
(
	RedisModuleCtx *ctx
);

void free_cmd_acl
(

); 