/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include <stdlib.h>
#include "cmd_acl.h"
#include <stdbool.h>
#include "../globals.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "./util/run_redis_command_as.h"


typedef struct {
	const char        *name;
	int               length;
	char              **commands;
	RedisModuleString **redis_module_commands_plus;
	RedisModuleString **redis_module_commands_minus;
} CommandCategory;

static CommandCategory *GRAPH_READONLY_USER = NULL;
static CommandCategory *GRAPH_USER = NULL;
static CommandCategory *GRAPH_ADMIN = NULL;


// create a new CommandCategory structure
// with the given commands_str and name
// returns NULL on failure
static CommandCategory* _create_command_category
(
	RedisModuleCtx *ctx,
	const char* commands_str,
	const char *name 
);


// Initializes the command ACL by reading environment variables 
// GRAPH_READONLY_USER, GRAPH_USER GRAPH_ADMIN and GRAPH_USER
// and build the corrisponding CommandCategory structure for each.
// The environment variables should contain space-separated lists of commands.
// For example: INFO CLIENT DBSIZE PING HELLO AUTH
// If an environment variable is not set or its value is "false", 
// the corresponding ACL will be NULL 
// and the GRAPH.ACL command will not be activated.
// Returns REDISMODULE_OK if the ACL initialization is successful,
// indicating that the GRAPH.ACL command should be activated.
int init_cmd_acl
(
	RedisModuleCtx *ctx
) {
	ASSERT(GRAPH_READONLY_USER == NULL);
	ASSERT(GRAPH_USER == NULL);
	ASSERT(GRAPH_ADMIN == NULL);
	ASSERT(ctx != NULL);

	const char *graph_readonly_commands = getenv("GRAPH_READONLY_USER");
	if ((graph_readonly_commands == NULL) || 
		(strcasecmp(graph_readonly_commands, "false") == 0)
		|| (strcmp(graph_readonly_commands, "") == 0)) {
		goto cleanup;
    
    }
	// create the CommandCategory structures for readonly commands
	if ((GRAPH_READONLY_USER = 
		_create_command_category(ctx, graph_readonly_commands, 
			"@graph-readonly-user")) == NULL) {
		goto cleanup;
	}
	 
	const char *graph_commands = getenv("GRAPH_USER");
	if ((graph_commands == NULL) || 
		(strcasecmp(graph_commands, "false") == 0)
		|| (strcmp(graph_commands, "") == 0)) {
		goto cleanup;
    } 

	if ((GRAPH_USER = 
		_create_command_category(ctx, graph_commands, "@graph-user")) == NULL) {
		goto cleanup;
	}

	const char *graph_admin_commands = getenv("GRAPH_ADMIN");
	if ((graph_admin_commands == NULL) || 
		(strcasecmp(graph_admin_commands, "false") == 0)
		|| (strcmp(graph_admin_commands, "") == 0)) {
		goto cleanup;
    
    } 
	if ((GRAPH_ADMIN = 
		_create_command_category(ctx, graph_admin_commands, "@graph-admin")) 
			== NULL) {
		goto cleanup;
	}
	
	RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_DEBUG, 
		"loading GRAPH.ACL command:\nadmin: %s\nuser: %s\nread_only: %s", 
		graph_admin_commands,
		graph_commands,
		graph_readonly_commands);
	
	return REDISMODULE_OK;

	cleanup:
		free_cmd_acl();
		return REDISMODULE_ERR;
}

static void free_command_category(CommandCategory *category) {
	if (category != NULL) {
		if (category->commands != NULL) {
			for (int i = 0; i < category->length; i++) {
				if (category->commands[i] != NULL) {
					rm_free(category->commands[i]);
				}
				if (category->redis_module_commands_plus[i] != NULL) {
					RedisModule_FreeString(NULL, category->redis_module_commands_plus[i]);
				}
				if (category->redis_module_commands_minus[i] != NULL) {
					RedisModule_FreeString(NULL, category->redis_module_commands_minus[i]);
				}
			}
			rm_free(category->commands);
			rm_free(category->redis_module_commands_plus);
			rm_free(category->redis_module_commands_minus);
		}
		rm_free(category);
	}
}

// free the resource of the cmd acl command
// idempotent
void free_cmd_acl
(

) {
	free_command_category(GRAPH_READONLY_USER);
	free_command_category(GRAPH_USER);
	free_command_category(GRAPH_ADMIN);
}


static int _command_in_category
(
	const char *cmd,
	const CommandCategory *category,
	bool *result
) {
	ASSERT(cmd != NULL);
	ASSERT(category != NULL);	
	ASSERT(category->commands != NULL);
	for (int i = 0; i < category->length; i++) {
		// skip the '+' or '-' first char at category->commands[i]
		if (strcasecmp(cmd, category->commands[i] + 1) == 0) {
			*result = true;
			return REDISMODULE_OK;
		}
		// check if the permission contains '|'
		// if so, its prefix (excluding the '|') 
		// should be in category->commands[i]
		const char *pipe = strchr(cmd, '|');
		if (pipe != NULL) {
			size_t prefix_len = pipe - cmd;
			char *prefix = rm_malloc(prefix_len + 1);
			if (prefix == NULL) {
				return REDISMODULE_ERR;
			}
			strncpy(prefix, cmd, prefix_len);
			prefix[prefix_len] = '\0';
			// filter in permission with pipe (e.g FOO|ADD) 
			// where FOO is in category->commands[i] 
			bool allowed = false;
			if (strcasecmp(prefix, category->commands[i] + 1) == 0) {
				allowed = true;
			}
			rm_free(prefix);
			if (allowed) {
				*result = true;
				return REDISMODULE_OK;
			}
		}
	}
	
	*result = false;
	return REDISMODULE_OK;

}

// compute the extra space needed for the expnations of category names 
// to acl_args
static int _compute_expand_offset
(
	RedisModuleCtx *ctx,  
	RedisModuleString **acl_args,
	int acl_argc
) {
	ASSERT(ctx                 != NULL);
	ASSERT(acl_args            != NULL);
	ASSERT(GRAPH_USER          != NULL);
	ASSERT(GRAPH_ADMIN         != NULL);
	ASSERT(GRAPH_READONLY_USER != NULL);

	int res = 0;
	for (int i = 0; i < acl_argc; i++) {
		const char *arg_str = RedisModule_StringPtrLen(acl_args[i], NULL);
		if ((arg_str[0] == '+' || arg_str[0] == '-')) {
			if (strcasecmp(arg_str + 1, GRAPH_USER->name) == 0) {
				res = res + GRAPH_USER->length;
			} else if (strcasecmp(arg_str + 1, GRAPH_ADMIN->name) == 0) {
				res = res + GRAPH_ADMIN->length;
			} else if (strcasecmp(arg_str + 1, GRAPH_READONLY_USER->name) == 0) {
				res = res + GRAPH_READONLY_USER->length;	
			}
		}
	}
	return res;
}

// expands the pseudo category into its respective commands and adds them
// to acl_args.
// returns true if the category was expanded, false otherwise.
static bool _expand_acl_pseudo_category
(
    RedisModuleString **acl_args,
    int *acl_argc,
    RedisModuleString *arg,
    CommandCategory *category
) {
	ASSERT(arg      != NULL);
	ASSERT(acl_args != NULL);
	ASSERT(acl_argc != NULL);
	ASSERT(category != NULL);

    const char *arg_str = RedisModule_StringPtrLen(arg, NULL);
    if (strcasecmp(arg_str + 1, category->name) == 0) {
        for (int i = 0; i < category->length; i++) {
            acl_args[*acl_argc] = (arg_str[0] == '+') ?
				 category->redis_module_commands_plus[i] :
				 category->redis_module_commands_minus[i];
            (*acl_argc)++;
        }
        return true;
    }
    return false;
}

// prepare arguments for redis ACL SETUSER command, remove permissionts that
// are not in GRAPH_ADMIN, GRAPH_READONLY_USER or GRAPH_USER. 
// argv is of the form ["SETUSER", "user1", "on", "+@admin", ...]
// We should copy every argument that does not start with a "+" or "-"
// We should *NOT* copy any argument that, when stripped of the "+" or "-",
// is not in GRAPH_ADMIN, GRAPH_READONLY_USER or GRAPH_USER (case insensitive)
// We should *NOT* copy any argument that, after being stripped of the 
// "+" or "-", does not start with one of the values from 
// GRAPH_ADMIN, GRAPH_READONLY_USER or GRAPH_USER
// followed by '|' (case insensitive)
// This function allocates a new RedisModuleString ***argv_ptr and releases 
// the old one
// also expand the pseudo categories @graph-admin, @graph-user and
// @graph-readonly-user
// return REDISMODULE_OK on success, REDISMODULE_ERR on failure
static int _senitaze_acl_setuser
(
	RedisModuleCtx *ctx, 
	RedisModuleString ***argv_ptr,
	int *argc 
) {
	ASSERT(*argc               > 0);
	ASSERT(ctx                 != NULL);
	ASSERT(argc                != NULL);
	ASSERT(argv_ptr            != NULL);
	ASSERT(GRAPH_USER          != NULL);
	ASSERT(GRAPH_ADMIN         != NULL);
	ASSERT(GRAPH_READONLY_USER != NULL);
	
	int expand_offset = _compute_expand_offset(ctx, *argv_ptr, *argc);
	RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_DEBUG, 
		"expand_offset: %d", expand_offset);

	RedisModuleString **argv = *argv_ptr;
	RedisModuleString **acl_args = 
		rm_malloc(sizeof(RedisModuleString*) * (*argc + expand_offset));

	if (acl_args == NULL) {
		RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING, 
		"Failed to allocate memory for ACL arguments.");
		return REDISMODULE_ERR;
	}

	int acl_argc = 0;
    // iterate over the items in GRAPH_ADMIN and use _command_in_category
	// to check if the command is allowed
	for (int i = 0; i < *argc; i++) {
		const char *arg_str = RedisModule_StringPtrLen(argv[i], NULL);
		if (arg_str[0] != '+' && arg_str[0] != '-') {
			acl_args[i] = argv[i];
			acl_argc++;
			continue;
		}
		// if it is one of the pseudo categories 
		// (@graph-admin, @graph-user or @graph-readonly-user), expand it
		// and add the commands to acl_args
		if (_expand_acl_pseudo_category(acl_args, &acl_argc, argv[i], GRAPH_ADMIN)) {
			continue;
		}
		if (_expand_acl_pseudo_category(acl_args, &acl_argc, argv[i], GRAPH_USER)) {
			continue;
		}
		if (_expand_acl_pseudo_category(acl_args, &acl_argc, argv[i], GRAPH_READONLY_USER)) {
			continue;
		}

		bool allowed = false;
		if (_command_in_category(arg_str + 1, GRAPH_ADMIN, &allowed) 
			!= REDISMODULE_OK) {
				rm_free(argv);
				rm_free(acl_args);
				return REDISMODULE_ERR;
		}
		// same for GRAPH_USER
		if (!allowed && (_command_in_category(arg_str + 1, GRAPH_USER, &allowed) 
			!= REDISMODULE_OK)) {
				rm_free(argv);
				rm_free(acl_args);
				return REDISMODULE_ERR;
		}
		// same for GRAPH_READONLY_USER
		if (!allowed && (_command_in_category(arg_str + 1, GRAPH_READONLY_USER, 
			&allowed) != REDISMODULE_OK)) {
				rm_free(argv);
				rm_free(acl_args);
				return REDISMODULE_ERR;
		}
		if (allowed) {
			acl_args[i] = argv[i];
			acl_argc++;
		}
	}

	rm_free(argv);
	*argc = acl_argc;
	*argv_ptr = acl_args;
	return REDISMODULE_OK;

}

// This function should be called with the ADMIN_USER in context.
// The username is the user that calls the GRAPH.ACL SETUSER command.
// The command uses the Redis module high-level API to call the Redis 
// ACL SETUSER command.
// An example of argv is: ["GRAPH.ACL", "SETUSER", "user1", "+@admin"].
// argv can be of arbitrary length because the user can stack multiple ACL 
// commands as can be done in the Redis ACL SETUSER command.
// In case the command is SETUSER, it uses 
// remove_setuser_forbidden_permissions to clean the arguments from permissions 
// that are not allowed for graph users.
static int _execute_acl_cmd_fn
(
	RedisModuleCtx *ctx, 
	RedisModuleString **argv, 
	int argc, 
	const char *username, 
	void *privdata
) {
	ASSERT(ctx != NULL);
	ASSERT(argv != NULL);
	ASSERT(argc > 0);
	ASSERT(username != NULL);

	int acl_argc = argc - 1;
	// Construct the correct arguments for RedisModule_Call
	// remove the GRAPH.ACL part from the arguments
	RedisModuleString **acl_args = 
		rm_malloc(sizeof(RedisModuleString*) * acl_argc);
	if (acl_args == NULL) {
		RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING, 
		"Failed to allocate memory for ACL arguments.");
		RedisModule_ReplyWithError(ctx, "FAILED");
		return REDISMODULE_ERR;
	}

	// remove the command name (GRAPH.ACL) from the arguments
	for (int i = 1; i < argc; i++) {
		acl_args[i - 1] = argv[i];
	} 

	// If the subcommand is SETUSER, we need to filter acl_args to 
	// remove permissions that are not allowed.
	// expand @graph-user, @graph-admin and @graph-readonly-user
    if (strcasecmp(RedisModule_StringPtrLen(argv[1], NULL), "SETUSER") == 0) {

		if (_senitaze_acl_setuser(ctx, &acl_args, &acl_argc) 
			!= REDISMODULE_OK) {
			rm_free(acl_args);
			RedisModule_ReplyWithError(ctx, "FAILED");
			return REDISMODULE_ERR;
		}
	}
	
	// just for log level debug
	char log_msg[1024 * 8]; 
	log_msg[0] = '\0'; 

	// Append each argument to the log message
	for (int i = 0; i < acl_argc; i++) {
		const char *arg_str = RedisModule_StringPtrLen(acl_args[i], NULL);
		if (i > 0) {
			strncat(log_msg, " ", sizeof(log_msg) - strlen(log_msg) - 1);
		}
		strncat(log_msg, arg_str, sizeof(log_msg) - strlen(log_msg) - 1);
	}

	RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_DEBUG,
		"delegating to execute ACL command '%s'",log_msg);
		
	RedisModuleCallReply *reply = 
		RedisModule_Call(ctx, "ACL", "v", acl_args, acl_argc);
	
	if (reply == NULL) {
		RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING,
			"Failed to execute ACL command '%s'. Error: %d",log_msg, errno);
		RedisModule_ReplyWithError(ctx, "FAILED");
    	rm_free(acl_args); 
		return REDISMODULE_ERR;
	}

	RedisModule_ReplyWithCallReply(ctx, reply);
	RedisModule_FreeCallReply(reply);
    rm_free(acl_args); 
	return REDISMODULE_OK;
}

// call execute_acl_userset_func with the ADMIN_USER in context
static int _execute_acl_cmd_as_admin
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT(ctx != NULL);
	ASSERT(argv != NULL);
	return run_redis_command_as(ctx, argv, argc, _execute_acl_cmd_fn, ADMIN_USER, 
	NULL);
}


int graph_acl_cmd
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT(ctx != NULL);
	ASSERT(argv != NULL);

	if(argc < 2){
		return RedisModule_WrongArity(ctx);
	}

 	const char *command = RedisModule_StringPtrLen(argv[1], NULL);
	if (strcasecmp(command, "SETUSER") == 0) {
		if(argc < 3 ){
			return RedisModule_WrongArity(ctx);
		}
		RedisModuleString *subject = argv[2];
		// if the subject is ADMIN_USER return
		if (strcasecmp(RedisModule_StringPtrLen(subject, NULL), ADMIN_USER) == 0) {
			RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING, 
				"Cannot change the ACL of the default user.");
			RedisModule_ReplyWithError(ctx, "FAILED");
			return REDISMODULE_ERR;
		}

		// execute the command as admin using redis ACL
		return _execute_acl_cmd_as_admin(ctx, argv, argc);
	} else if (strcasecmp(command, "SAVE") == 0) {
		return _execute_acl_cmd_as_admin(ctx, argv, argc);
	} else if (strcasecmp(command, "GETUSER") == 0) {
		if(argc < 3 ){
			return RedisModule_WrongArity(ctx);
		}
	
		RedisModuleString *subject = argv[2];
		// if the subject is ADMIN_USER return
		if (strcmp(RedisModule_StringPtrLen(subject, NULL), ADMIN_USER) == 0) {
			RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING,
				"Cannot change the ACL of the default user.");
			RedisModule_ReplyWithError(ctx, "FAILED");
			return REDISMODULE_ERR;
		}
		// execute the command as admin using redis ACL
		return _execute_acl_cmd_as_admin(ctx, argv, argc);

	} else {
		RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING,
			"Unknown command: GRAPH.ACL %s", command);
		RedisModule_ReplyWithError(ctx, "Unknown command");
		return REDISMODULE_ERR;
	}

	return REDISMODULE_OK;
}


static CommandCategory* _create_command_category
(
	RedisModuleCtx *ctx,
	const char* commands_str,
	const char *name 
) {
	ASSERT(ctx != NULL);
	ASSERT(commands_str != NULL);
	ASSERT(name != NULL);

	CommandCategory *category = (CommandCategory*)rm_malloc(sizeof(CommandCategory));
	if (category == NULL) {
		return NULL;
	}
	category->name = name;

	// Count the number of substrings
	int substrings = 0;
	const char* ptr = commands_str;
	while (*ptr != '\0') {
		if (*ptr == ' ') {
			substrings++;
		}
		ptr++;
	}

	substrings++; // Add one for the last substring

	// Allocate memory for the array of pointers
	category->commands = (char**)rm_malloc(substrings * sizeof(const char*));
	if (category->commands == NULL) {
		free_command_category(category);
		return NULL;
	} 
	category->redis_module_commands_plus = (RedisModuleString**)rm_malloc(substrings * sizeof(RedisModuleString*));
	if (category->redis_module_commands_plus == NULL) {
		free_command_category(category);
		return NULL;
	}

	category->redis_module_commands_minus = (RedisModuleString**)rm_malloc(substrings * sizeof(RedisModuleString*));
	if (category->redis_module_commands_minus == NULL) {
		free_command_category(category);
		return NULL;
	}

	// Split the string into substrings
	const char* start = commands_str;
	for (int i = 0; i < substrings; i++) {
		const char* end = strchr(start, ' ');
		if (end == NULL) {
			// If no more spaces, end at the end of the string
			end = strchr(start, '\0'); 
		}
		size_t length = end - start;
		// Allocate memory for the substring, substr [0] is '+' or '-'
		char* substr = (char*)rm_malloc(length + 2);
		if (substr == NULL) {
			free_command_category(category);
			return NULL;
		}
		substr[0] = '+';
		strncpy(substr + 1, start , length);
		substr[length + 1] = '\0'; // Null-terminate the substring
		category->commands[i] = substr;

		RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_DEBUG,
			"adding substr %s to category", substr + 1);

		// create 2 strings for each command, one with '+' and one with '-'
		
		category->redis_module_commands_plus[i] = 
			RedisModule_CreateString(ctx, substr, length + 1);

		substr[0] = '-';
		category->redis_module_commands_minus[i] = 
			RedisModule_CreateString(ctx, substr, length + 1);	

		if (category->redis_module_commands_plus[i] == NULL || 
			category->redis_module_commands_minus[i] == NULL) {
			RedisModule_Log(ctx, "error",
				"creation of redis module string %s failed", substr);
			free_command_category(category);
			return NULL;	
		}
		start = end + 1; // Move to the next substring
	}

	category->length = substrings;
	return category;
	
}
