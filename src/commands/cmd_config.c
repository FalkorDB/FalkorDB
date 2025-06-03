/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include <string.h>
#include "RG.h"
#include "configuration/config.h"

bool validate_config_keys(RedisModuleCtx *ctx, RedisModuleCallReply *reply, Config_Option_Field *fields) {
	size_t len = RedisModule_CallReplyLength(reply);

	if (len == 0) {
		return false; 
	}
	
	for (size_t i = 0; i < len; i += 2) {
		RedisModuleCallReply *key = RedisModule_CallReplyArrayElement(reply, i);
		size_t key_len;
		const char *key_str = RedisModule_CallReplyStringPtr(key, &key_len);
		
		// remove the "graph." prefix from the key
		if (strncmp(key_str, "graph.", 6) == 0) {
			key_str += 6; // skip the "graph." prefix
			key_len -= 6;
		}
		
		// create null-terminated C string for validation
		char *key_cstr = RedisModule_Alloc(key_len + 1);
		memcpy(key_cstr, key_str, key_len);
		key_cstr[key_len] = '\0';
		
		Config_Option_Field field;
		bool is_valid = Config_Contains_field(key_cstr, &field);
		RedisModule_Free(key_cstr);
		
		if (!is_valid) {
			return false; // invalid key found
		}
		
		// store the field type at position i/2 in the array
		if (fields != NULL) {
			fields[i / 2] = field;
		}
	}
	
	return true; 
}

int Get_Config(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	// add greph prefix to each config key
	for(int i = 2; i < argc; i += 2) {
		RedisModuleString *s = RedisModule_CreateStringPrintf(ctx, "%s%s",
				"graph.", RedisModule_StringPtrLen(argv[i], NULL));

		RedisModule_FreeString(ctx, argv[i]);
		argv[i] = s;
	}

	// send the CONFIG GET request
	RedisModuleCallReply *reply = RedisModule_Call(ctx, "CONFIG", "v", argv+1, argc-1);

	// translate values and structure to GRAPH.CONFIG GET format
	size_t len = RedisModule_CallReplyLength(reply);
	size_t num_pairs = (len == 2) ? 2 : len / 2;
	
	// create array to store field types for each key
	Config_Option_Field *fields = RedisModule_Alloc(num_pairs * sizeof(Config_Option_Field));
	
	// validate that all keys in the reply are valid configuration fields
	if (!validate_config_keys(ctx, reply, fields)) {
		RedisModule_ReplyWithError(ctx, "Unknown configuration field");
		RedisModule_FreeCallReply(reply);
		RedisModule_Free(fields);
		return REDISMODULE_ERR;
	}
	
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

		// get the type of the value from pre-validated fields array
		Config_Option_Field field = fields[i / 2];
		
		// use the type returns by Config_Field_type(field) to convert the 
		// value: string -> string, boolean yes -> 1 no -> 0, 
		// number -> as long
		switch(Config_Field_type(field)) {
			case T_STRING: {
				// if the value is a string, return it as is
				RedisModule_ReplyWithStringBuffer(ctx, value_str, value_len);
				break;
			}
			case T_INT64: {
				long long int_value;
				RedisModuleString *value_rms = RedisModule_CreateString(ctx, value_str, value_len);
				RedisModule_StringToLongLong(value_rms, &int_value);
				RedisModule_ReplyWithLongLong(ctx, int_value);
				RedisModule_FreeString(ctx, value_rms);
				break;
			}
			case T_BOOL: {
				if ((value_len == 2 && strncasecmp(value_str, "no", 2) == 0)) {
					RedisModule_ReplyWithBool(ctx, 0);	
				} else if ((value_len == 3 && strncasecmp(value_str, "yes", 3) == 0)) {
					RedisModule_ReplyWithBool(ctx, 1);
				}
				break;
			}
			default: 
				ASSERT(false && "unsupported config type");
			}
	}

	RedisModule_FreeCallReply(reply);
	RedisModule_Free(fields);

	return REDISMODULE_OK;
} 

int Set_Config(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
    // GRAPH.CONFIG SET NAME1 value1 [NAME2 value2 ...]
	// Minimum args: GRAPH.CONFIG SET NAME value (argc == 4)
	// Multiple args must be in pairs: argc must be even and >= 4
	if(argc < 4 || argc % 2 != 0) {
		return RedisModule_WrongArity(ctx);
	}

	// Calculate number of key-value pairs
	int num_pairs = (argc - 2) / 2;
	
	// Create arguments for Redis CONFIG command: "SET" key1 value1 key2 value2 ...
	// Total args = 1 ("SET") + 2 * num_pairs (key-value pairs)
	int total_args = 1 + 2 * num_pairs;
	RedisModuleString **set_args = RedisModule_Alloc(total_args * sizeof(RedisModuleString*));
	
	// Array to store field types and track which values need to be freed
	Config_Option_Field *fields = RedisModule_Alloc(num_pairs * sizeof(Config_Option_Field));
	bool *values_to_free = RedisModule_Alloc(num_pairs * sizeof(bool));
	
	set_args[0] = argv[1];  // "SET"
	
	// Process each key-value pair
	for(int i = 0; i < num_pairs; i++) {
		int key_idx = 2 + i * 2;             // Position of key in argv
		int value_idx = key_idx + 1;         // Position of value in argv
		int set_key_idx = 1 + i * 2;         // Position of key in set_args
		int set_value_idx = set_key_idx + 1; // Position of value in set_args
		
		size_t key_len;
		const char *key_str = RedisModule_StringPtrLen(argv[key_idx], &key_len);
		
		// Validate the configuration key
		if(!Config_Contains_field(key_str, &fields[i])) {
			// Clean up allocated memory before error return
			for(int j = 0; j < i; j++) {
				RedisModule_FreeString(ctx, set_args[1 + j * 2]);     // Free prefixed keys
				if(values_to_free[j]) {
					RedisModule_FreeString(ctx, set_args[2 + j * 2]); // Free converted boolean values
				}
			}
			RedisModule_Free(set_args);
			RedisModule_Free(fields);
			RedisModule_Free(values_to_free);
			RedisModule_ReplyWithError(ctx, "Unknown configuration name");
			return REDISMODULE_ERR;
		}
		
		// Add the "graph." prefix to the key
		set_args[set_key_idx] = RedisModule_CreateStringPrintf(ctx, "%s%s", "graph.", key_str);
		
		// Initialize as false, will be set to true if we create a new string for boolean
		values_to_free[i] = false;
		
		// Process the value based on field type
		RedisModuleString *value = argv[value_idx];
		switch(Config_Field_type(fields[i])) {
			case T_STRING: {
				// String values can be used as-is
				set_args[set_value_idx] = value;
				break;
			}
			case T_INT64: {
				// Integer values can be used as-is
				set_args[set_value_idx] = value;
				break;
			}
			case T_BOOL: {
				// Boolean values need to be converted to "true"/"false"
				long long bool_value;
				if(RedisModule_StringToLongLong(value, &bool_value) != REDISMODULE_OK) {
					// Clean up allocated memory before error return
					for(int j = 0; j <= i; j++) {
						RedisModule_FreeString(ctx, set_args[1 + j * 2]); // Free prefixed keys
						if(j < i && values_to_free[j]) {
							RedisModule_FreeString(ctx, set_args[2 + j * 2]); // Free converted boolean values
						}
					}
					RedisModule_Free(set_args);
					RedisModule_Free(fields);
					RedisModule_Free(values_to_free);
					RedisModule_ReplyWithError(ctx, "Invalid boolean value");
					return REDISMODULE_ERR;
				}
				if (bool_value == 0) {
					set_args[set_value_idx] = RedisModule_CreateString(ctx, "false", 5);
					values_to_free[i] = true;
				} else if (bool_value == 1) { 
					set_args[set_value_idx] = RedisModule_CreateString(ctx, "true", 4);
					values_to_free[i] = true;
				} else {
					// Clean up allocated memory before error return
					for(int j = 0; j <= i; j++) {
						RedisModule_FreeString(ctx, set_args[1 + j * 2]); // Free prefixed keys
						if(j < i && values_to_free[j]) {
							RedisModule_FreeString(ctx, set_args[2 + j * 2]); // Free converted boolean values
						}
					}
					RedisModule_Free(set_args);
					RedisModule_Free(fields);
					RedisModule_Free(values_to_free);
					RedisModule_ReplyWithError(ctx, "Invalid boolean value");
					return REDISMODULE_ERR;
				}
				break;
			}
			default:
				ASSERT(false && "unsupported config type");
		}
	}

	// Execute the Redis CONFIG command with all key-value pairs
	RedisModuleCallReply *reply = RedisModule_Call(ctx, "CONFIG", "v", set_args, total_args);
	RedisModule_ReplyWithCallReply(ctx, reply);

	// Clean up allocated memory
	for(int i = 0; i < num_pairs; i++) {
		RedisModule_FreeString(ctx, set_args[1 + i * 2]);     // Free prefixed keys
		if(values_to_free[i]) {
			RedisModule_FreeString(ctx, set_args[2 + i * 2]); // Free converted boolean values
		}
	}
	RedisModule_Free(set_args);
	RedisModule_Free(fields);
	RedisModule_Free(values_to_free);
	
	return REDISMODULE_OK;
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
	if(strcasecmp(action, "get") == 0) {
		return Get_Config(ctx, argv, argc); 
	} else if(strcasecmp(action, "set") == 0) {
		return Set_Config(ctx, argv, argc); 
	} else {
		RedisModule_ReplyWithErrorFormat(ctx, "ERR unknown subcommand '%s'.",
				action);
		return REDISMODULE_OK;
	}
}
