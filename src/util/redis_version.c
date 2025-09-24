/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "redis_version.h"
#include "../redismodule.h"
#include <stdbool.h>

//
static Redis_Version _redis_version;
static bool _initialized = false;

// Calculates the semantic version.
static inline uint _SemanticVersion(uint major, uint minor, uint patch) {
	return major * 10000 + minor * 100 + patch;
}

Redis_Version RG_GetRedisVersion() {
	// Singleton.
	if(_initialized) return _redis_version;
	RedisModuleCtx *ctx = RedisModule_GetThreadSafeContext(NULL);
	// Check if there is an implementation for redis module api for redis 6 and up, by checking the existence pf a Redis 6 API function pointer.
	if(RedisModule_GetServerInfo) {
		// Retrive the server info.
		const char *server_version;
		RedisModuleServerInfoData *info = RedisModule_GetServerInfo(ctx, "Server");
		server_version = RedisModule_ServerInfoGetFieldC(info, "redis_version");
		sscanf(server_version, "%u.%u.%u", &_redis_version.major, &_redis_version.minor,
			   &_redis_version.patch);
		RedisModule_FreeServerInfo(ctx, info);
	} else {
		RedisModuleCallReply *reply = RedisModule_Call(ctx, "info", "c", "server");
		size_t len;
		const char *replyStr = RedisModule_CallReplyStringPtr(reply, &len);
		sscanf(replyStr, "# Server\nredis_version:%u.%u.%u", &_redis_version.major, &_redis_version.minor,
			   &_redis_version.patch);
		RedisModule_FreeCallReply(reply);
	}
	RedisModule_FreeThreadSafeContext(ctx);
	_initialized = true;
	return _redis_version;
}

inline bool Redis_Version_GreaterOrEqual(uint major, uint minor, uint patch) {
	if(!_initialized) RG_GetRedisVersion();
	return _SemanticVersion(major, minor, patch) <=
		   _SemanticVersion(_redis_version.major, _redis_version.minor, _redis_version.patch);
}
