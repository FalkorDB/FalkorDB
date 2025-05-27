/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <string.h>
#include "RG.h"
#include "configuration/config.h"

int Graph_Config
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	// GRAPH.CONFIG <GET|SET> <NAME> [value]
	if(argc < 3) {
		return RedisModule_WrongArity(ctx);
	}

	// determine action
	int step;
	const char *action = RedisModule_StringPtrLen(argv[1], NULL);
	if(strcasecmp(action, "get") == 0) {
		step = 1;
	} else if(strcasecmp(action, "set") == 0) {
		step = 2;
	} else {
		RedisModule_ReplyWithErrorFormat(ctx, "ERR unknown subcommand '%s'.",
				action);
		return REDISMODULE_OK;
	}

	// add "GRAPH." prefix to each config key
	// e.g.
	// RESULTSET_MAX_SIZE > GRAPH.RESULTSET_MAX_SIZE

	for(int i = 2; i < argc; i+=step) {
		RedisModuleString *s = RedisModule_CreateStringPrintf(ctx, "%s%s",
				"graph.", RedisModule_StringPtrLen(argv[i], NULL));

		RedisModule_FreeString(ctx, argv[i]);
		argv[i] = s;
	}

	RedisModuleCallReply *reply = RedisModule_Call(ctx, "CONFIG", "v", argv+1, argc-1);

	if (step == 1 && RedisModule_CallReplyType(reply) == REDISMODULE_REPLY_ARRAY) { // config get
		size_t len = RedisModule_CallReplyLength(reply);
		size_t num_pairs = (len == 2) ? 2 : len / 2;
		RedisModule_ReplyWithArray(ctx, num_pairs);
		for (size_t i = 0; i < len; i+=2) {
			RedisModuleCallReply *key = RedisModule_CallReplyArrayElement(reply, i);
			RedisModuleCallReply *value = RedisModule_CallReplyArrayElement(reply, i + 1);
			size_t key_len, value_len;
			const char *key_str = RedisModule_CallReplyStringPtr(key, &key_len);			
			const char *value_str = RedisModule_CallReplyStringPtr(value, &value_len);
			// remove the "graph." prefix from the key
			if (strncmp(key_str, "graph.", 6) == 0) {
				key_str += 6; // skip the "graph." prefix
				key_len -= 6;
			}

            // in case of get with one key we return array of key value, 
			// in case of multiple keys we return array of arrays.		
			if (len != 2) {
				RedisModule_ReplyWithArray(ctx, 2);
			}
			RedisModule_ReplyWithStringBuffer(ctx, key_str, key_len); 

			// if value is number convert it to int and reply with it
			// if it is 'yes' or 'no' reply with booleans 'yes' -> 1, 'no' -> 0
			// othjrwise reply with the string value  
			RedisModuleString *value_rms = RedisModule_CreateString(ctx, value_str, value_len);
			long long int_value;
			if (RedisModule_StringToLongLong(value_rms, &int_value) == REDISMODULE_OK) {
				RedisModule_ReplyWithLongLong(ctx, int_value);
			} else if ((value_len == 3 && strncasecmp(value_str, "yes", 3) == 0)) {
        		RedisModule_ReplyWithBool(ctx, 1);
			} else if ((value_len == 2 && strncasecmp(value_str, "no", 2) == 0)) {
				RedisModule_ReplyWithBool(ctx, 0);	
			} else {
				RedisModule_ReplyWithStringBuffer(ctx, value_str, value_len);
			}
    		RedisModule_FreeString(ctx, value_rms);
		}

	} else {
		RedisModule_ReplyWithCallReply(ctx, reply);
	}


	RedisModule_FreeCallReply(reply);

	return REDISMODULE_OK;
}

