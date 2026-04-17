/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <string.h>
#include <limits.h>
#include "RG.h"
#include "configuration/config.h"

static bool _Config_GetNumericValue(Config_Option_Field field, long long *value) {
	ASSERT(value != NULL);

	switch(field) {
	case Config_TIMEOUT:
	case Config_TIMEOUT_DEFAULT:
	case Config_TIMEOUT_MAX:
	case Config_CACHE_SIZE:
	case Config_OPENMP_NTHREAD:
	case Config_THREAD_POOL_SIZE:
	case Config_VKEY_MAX_ENTITY_COUNT:
	case Config_MAX_QUEUED_QUERIES:
	case Config_NODE_CREATION_BUFFER:
	case Config_CMD_INFO_MAX_QUERY_COUNT:
	case Config_EFFECTS_THRESHOLD: {
		uint64_t n = 0;
		bool res = Config_Option_get(field, &n);
		if(!res) return false;
		ASSERT(n <= (uint64_t)LLONG_MAX);
		*value = (long long)n;
		return true;
	}

	case Config_RESULTSET_MAX_SIZE: {
		uint64_t n = 0;
		bool res = Config_Option_get(field, &n);
		if(!res) return false;
		if(n == RESULTSET_SIZE_UNLIMITED) {
			*value = -1;
			return true;
		}
		ASSERT(n <= (uint64_t)LLONG_MAX);
		*value = (long long)n;
		return true;
	}

	case Config_QUERY_MEM_CAPACITY:
	case Config_DELTA_MAX_PENDING_CHANGES: {
		int64_t n = 0;
		bool res = Config_Option_get(field, &n);
		if(!res) return false;
		*value = (long long)n;
		return true;
	}

	case Config_BOLT_PORT: {
		int32_t n = 0;
		bool res = Config_Option_get(field, &n);
		if(!res) return false;
		*value = (long long)n;
		return true;
	}

	case Config_JS_HEAP_SIZE:
	case Config_JS_STACK_SIZE: {
		size_t n = 0;
		bool res = Config_Option_get(field, &n);
		if(!res) return false;
		ASSERT(n <= (size_t)LLONG_MAX);
		*value = (long long)n;
		return true;
	}

	default:
		return false;
	}
}

// emit config field name and value
static void _Emit_config
(
	RedisModuleCtx *ctx,       // redis module context
	Config_Option_Field field  // config field
) {
	bool res;
	bool bool_value;
	char *str_value;
	long long numeric_value;

	// get config name
	const char *config_name = Config_Field_name(field);
	ASSERT(config_name != NULL);

	// get config according to its type
	switch(Config_Field_type(field)) {
		case T_BOOL:
			res = Config_Option_get(field, &bool_value);
			ASSERT(res == true);

			RedisModule_ReplyWithArray(ctx, 2);
			RedisModule_ReplyWithCString(ctx, config_name);
			RedisModule_ReplyWithBool(ctx, bool_value);

			break;

		case T_INT64:
				res = _Config_GetNumericValue(field, &numeric_value);
			ASSERT(res == true);

			RedisModule_ReplyWithArray(ctx, 2);
			RedisModule_ReplyWithCString(ctx, config_name);
			RedisModule_ReplyWithLongLong(ctx, numeric_value);

			break;
		case T_STRING:
			res = Config_Option_get(field, &str_value);
			ASSERT(res == true);

			RedisModule_ReplyWithArray(ctx, 2);
			RedisModule_ReplyWithCString(ctx, config_name);
			RedisModule_ReplyWithCString(ctx, str_value);
			break;

		default:
			ASSERT(false && "unsupported config type");
	}
}

static void _Config_get_all
(
	RedisModuleCtx *ctx
) {
	uint config_count = Config_END_MARKER;
	RedisModule_ReplyWithArray(ctx, config_count);

	for(Config_Option_Field field = 0; field < Config_END_MARKER; field++) {
		_Emit_config(ctx, field);
	}
}

void _Config_get
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	// return the given config's value to the user
	Config_Option_Field config_field;
	const char *config_name = RedisModule_StringPtrLen(argv[2], NULL);

	// return entire configuration
	if(!strcmp(config_name, "*")) {
		_Config_get_all(ctx);
		return;
	}

	// return specific configuration field
	if(!Config_Contains_field(config_name, &config_field)) {
		RedisModule_ReplyWithError(ctx, "Unknown configuration field");
		return;
	} else {
		_Emit_config(ctx, config_field);
	}
}

void _Config_set
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	//--------------------------------------------------------------------------
	// validate configuration
	//--------------------------------------------------------------------------

	// dryrun configuration, make sure all configurations are valid
	for(int i = 0; i < argc; i += 2) {
		RedisModuleString *key = argv[i];
		RedisModuleString *val = argv[i+1];

		//----------------------------------------------------------------------
		// retrieve and validate config field
		//----------------------------------------------------------------------

		Config_Option_Field config_field;
		const char *config_name = RedisModule_StringPtrLen(key, NULL);
		if(!Config_Contains_field(config_name, &config_field)) {
			RedisModule_ReplyWithError(ctx, "Unknown configuration field");
			return;
		}

		// ensure field is a runtime configuration
		bool configurable_field = false;
		for(int i = 0; i < RUNTIME_CONFIG_COUNT; i++) {
			if(RUNTIME_CONFIGS[i] == config_field) {
				configurable_field = true;
				break;
			}
		}
	
		// field is not allowed to be reconfigured
		if(!configurable_field) {
			RedisModule_ReplyWithError(ctx, "This configuration parameter cannot be set at run-time");
			return;
		}

		// make sure value is valid
		char *error = NULL;
		const char *val_str = RedisModule_StringPtrLen(val, NULL);
		if(!Config_Option_dryrun(config_field, val_str, &error)) {
			if(error != NULL) {
				RedisModule_ReplyWithError(ctx, error);
			} else {
				char *errmsg;
				int rc __attribute__((unused));
				rc = asprintf(&errmsg, "Failed to set config value %s to %s", config_name, val_str);
				RedisModule_ReplyWithError(ctx, errmsg);
				free(errmsg);
			}
			return;
		}
	}

	// if we're here configuration passed all validations
	// apply configuration
	for(int i = 0; i < argc; i += 2) {
		bool               res   =  false;
		RedisModuleString  *key  =  argv[i];
		RedisModuleString  *val  =  argv[i+1];

		UNUSED(res);

		Config_Option_Field config_field;
		const char *config_name = RedisModule_StringPtrLen(key, NULL);
		res = Config_Contains_field(config_name, &config_field);
		ASSERT(res == true);

		// set configuration
		const char *val_str = RedisModule_StringPtrLen(val, NULL);
		res = Config_Option_set(config_field, val_str, NULL);
		ASSERT(res == true);
	}

	RedisModule_ReplyWithSimpleString(ctx, "OK");
}

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

	const char *action = RedisModule_StringPtrLen(argv[1], NULL);

	if(!strcasecmp(action, "GET")) {
		// GRAPH.CONFIG GET <NAME>
		if(argc != 3) {
			return RedisModule_WrongArity(ctx);
		}
		_Config_get(ctx, argv, argc);
	} else if(!strcasecmp(action, "SET")) {
		// GRAPH.CONFIG SET <NAME> [value] <NAME> [value] ...
		// emit an error if we received an odd number of arguments,
		// as this indicates an invalid configuration
		if(argc < 4 || (argc % 2) == 1) {
			return RedisModule_WrongArity(ctx);
		}
		_Config_set(ctx, argv+2, argc-2);
	} else {
		RedisModule_ReplyWithError(ctx, "Unknown subcommand for GRAPH.CONFIG");
	}

	return REDISMODULE_OK;
}

