/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <string.h>
#include "RG.h"
#include "configuration/config.h"

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
			res = Config_Option_get(field, &numeric_value);
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

