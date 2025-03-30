/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "cmd_acl.h"
#include "../globals.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "./util/run_redis_command_as.h"

#include <stdlib.h>
#include <stdbool.h>

// data structure describing an ACL category
// a category contains a name and a set of commands which can be either
// enabled or disabled
typedef struct {
	const char        *name;                          // categoty name
	int               length;                         // #commands in categoty
	char              **commands;                     // commands in categoty
	bool [length] true, true, false, true
	RedisModuleString **redis_module_commands_plus;   // 
	RedisModuleString **redis_module_commands_minus;  // 
} CommandCategory;

//------------------------------------------------------------------------------
// GRAPH ACL categories
//------------------------------------------------------------------------------

// 1. readonly user categoty
// 2. user category
// 3. admin category
static CommandCategory *ACL_GRAPH_USER          = NULL;
static CommandCategory *ACL_GRAPH_ADMIN         = NULL;
static CommandCategory *ACL_GRAPH_READONLY_USER = NULL;

// create a new CommandCategory structure
// with the given commands_str and name
// returns NULL on failure
static CommandCategory* _create_command_category
(
	RedisModuleCtx *ctx,       // redis module context
	const char *commands_str,  // space seperated list of redis commands
	const char *name           // category name
) {
	ASSERT(ctx          != NULL);
	ASSERT(name         != NULL);
	ASSERT(commands_str != NULL);

	CommandCategory *category =
		(CommandCategory*)rm_calloc(1, sizeof(CommandCategory));

	if(category == NULL) {
		return NULL;
	}

	// set categoty name
	category->name = name;

	// count the number of commands
	uint n_cmds = 0;

	// TODO: swtich to strtok
	// consider using arr.h array_append
	assert(false && "see TODO");
	const char* ptr = commands_str;
	while(*ptr != '\0') {
		if (*ptr == ' ') {
			n_cmds++;
		}
		ptr++;
	}

	n_cmds++; // add one for the last substring

	// allocate commands array
	category->commands = (char**)rm_malloc(n_cmds * sizeof(const char*));
	if(category->commands == NULL) {
		goto cleanup;
	} 

	category->redis_module_commands_plus =
		(RedisModuleString**)rm_malloc(n_cmds * sizeof(RedisModuleString*));
	if(category->redis_module_commands_plus == NULL) {
		goto cleanup;
	}

	category->redis_module_commands_minus =
		(RedisModuleString**)rm_malloc(n_cmds * sizeof(RedisModuleString*));
	if(category->redis_module_commands_minus == NULL) {
		goto cleanup;
	}

	// split the string into n_cmds
	const char *start = commands_str;
	for(int i = 0; i < n_cmds; i++) {
		const char* end = strchr(start, ' ');
		if (end == NULL) {
			// if no more spaces, end at the end of the string
			end = strchr(start, '\0'); 
		}
		size_t length = end - start;
		// allocate memory for the substring, substr [0] is '+' or '-'
		char* substr = (char*)rm_malloc(length + 2);
		if (substr == NULL) {
			goto cleanup;
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
			goto cleanup;	
		}
		start = end + 1; // Move to the next substring
	}

	category->length = n_cmds;
	return category;
	
	cleanup:
		free_command_category(&category);
		return NULL;
}

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
) {
	// validations
	ASSERT(ctx                     != NULL);
	ASSERT(ACL_GRAPH_USER          == NULL);
	ASSERT(ACL_GRAPH_ADMIN         == NULL);
	ASSERT(ACL_GRAPH_READONLY_USER == NULL);

	//--------------------------------------------------------------------------
	// initialize ACL graph readonly user
	//--------------------------------------------------------------------------

	const char *category_name = NULL;

	category_name = getenv("ACL_GRAPH_READONLY_USER");
	if((category_name == NULL) || 
		(strcasecmp(category_name, "false") == 0)
		|| (strcmp(category_name, "") == 0)) {
		goto cleanup;
    }

	// create the CommandCategory structures for readonly commands
	if ((ACL_GRAPH_READONLY_USER = 
		_create_command_category(ctx, category_name, 
			"@graph-readonly-user")) == NULL) {
		goto cleanup;
	}
	 
	//--------------------------------------------------------------------------
	// initialize ACL graph user
	//--------------------------------------------------------------------------

	category_name = getenv("ACL_GRAPH_USER");
	if ((category_name == NULL) || 
		(strcasecmp(category_name, "false") == 0)
		|| (strcmp(category_name, "") == 0)) {
		goto cleanup;
    } 

	if ((GRAPH_USER = 
		_create_command_category(ctx, category_name, "@graph-user")) == NULL) {
		goto cleanup;
	}

	//--------------------------------------------------------------------------
	// initialize ACL admin
	//--------------------------------------------------------------------------

	category_name = getenv("ACL_GRAPH_ADMIN");
	if ((category_name == NULL) || 
		(strcasecmp(category_name, "false") == 0)
		|| (strcmp(category_name, "") == 0)) {
		goto cleanup;
    
    } 
	if ((GRAPH_ADMIN = 
		_create_command_category(ctx, category_name, "@graph-admin")) 
			== NULL) {
		goto cleanup;
	}
	
	// return success
	return REDISMODULE_OK;

	// something went wrong, return failure
	cleanup:
		free_cmd_acl();
		return REDISMODULE_ERR;
}


// checks if given command is part of the graph ACL categoty
// returns true if the cmd is part of the categoty, false otherwise
static bool _command_in_category
(
	const char *cmd,                 // command in question
	const CommandCategory *category  // category to search in
) {
	ASSERT(cmd                != NULL);
	ASSERT(category           != NULL);
	ASSERT(category->commands != NULL);

	for(int i = 0; i < category->length; i++) {
		// TODO: double check if indeed '+' or '-' can show up in commands[i]
		// skip the '+' or '-' first char at category->commands[i]
		if(strcasecmp(cmd, category->commands[i] + 1) == 0) {
			return true;
		}

		// check if the permission contains '|'
		// if so, its prefix (excluding the '|') 
		// should be in category->commands[i]
		const char *pipe = strchr(cmd, '|');
		if(pipe != NULL) {
			size_t prefix_len = pipe - cmd;

			// filter in permission with pipe (e.g FOO|ADD) 
			// where FOO is in category->commands[i] 
			if(strncasecmp(cmd, prefix_len, category->commands[i] + 1) == 0) {
				return true;
			}
		}
	}
	
	// no match found, report false
	return false;
}

// expands the pseudo category into its respective commands and adds them
// to acl_args
// returns true if the category was expanded, false otherwise
//
// example:
// GRAPH.ACL SETUSER foo +@graph.acl.admin
static bool _expand_acl_pseudo_category
(
    RedisModuleString **acl_args,  // [input/output]
    int *acl_argc,                 // [input/output]
    RedisModuleString *arg,        // 
    CommandCategory *category      // 
) {
	ASSERT(arg      != NULL);
	ASSERT(acl_args != NULL);
	ASSERT(acl_argc != NULL);
	ASSERT(category != NULL);

    const char *arg_str = RedisModule_StringPtrLen(arg, NULL);

	// assuming '-' assign relevant commands array
	RedisModuleString **commands = category->redis_module_commands_minus;

	// make sure string starts with either '+' or '-'
	if((arg_str[0] == '+' || arg_str[0] == '-')) {
		if (arg_str[0] == '+') {
			// update commands array
			commands = category->redis_module_commands_plus;
		}
	} else {
		// arg_str doesn't start with either '+' nor '-', return false
		return false;
	}

	// check if arg represents a categoty? skip '+' or '-'
    if(strcasecmp(arg_str + 1, category->name) != 0) {
		return false;
	}

	// arg refers to categoty
	int n = *acl_argc;
	for(int i = 0; i < category->length; i++) {
		acl_args[n++] = commands[i];
	}

	// update acl_argc
	(*acl_argc) = n;

	return true;
}

// compute the extra space needed for the expnations of category names 
// to acl_args
static int _compute_expand_offset
(
	RedisModuleCtx *ctx,            // redis module context
	RedisModuleString **acl_args,   // [input/output]
	int acl_argc                    // 
) {
	ASSERT(ctx                 != NULL);
	ASSERT(acl_args            != NULL);
	ASSERT(GRAPH_USER          != NULL);
	ASSERT(GRAPH_ADMIN         != NULL);
	ASSERT(GRAPH_READONLY_USER != NULL);

	int res = 0;
	for(int i = 0; i < acl_argc; i++) {
		const char *arg_str = RedisModule_StringPtrLen(acl_args[i], NULL);

		// if arg doesn't begins with either '+' or '-' then this is not
		// a command and we should ignore
		if((arg_str[0] == '+' || arg_str[0] == '-')) {
			continue;
		}

		// is this a graph user categoty?
		if(strcasecmp(arg_str + 1, GRAPH_USER->name) == 0) {
			res = res + GRAPH_USER->length;
		}

		// is this a graph admin categoty?
		else if(strcasecmp(arg_str + 1, GRAPH_ADMIN->name) == 0) {
			res = res + GRAPH_ADMIN->length;
		}

		// is this a graph readonly user categoty?
		else if(strcasecmp(arg_str + 1, GRAPH_READONLY_USER->name) == 0) {
			res = res + GRAPH_READONLY_USER->length;	
		}
	}

	return res;
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
    // iterate over the items in GRAPH_ADMIN, 
	// GRAPH_USER and GRAPH_READONLY_USER and use _command_in_category
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
				goto cleanup;
		}
		// same for GRAPH_USER
		if (!allowed && (_command_in_category(arg_str + 1, GRAPH_USER, &allowed) 
			!= REDISMODULE_OK)) {
				goto cleanup;
		}
		// same for GRAPH_READONLY_USER
		if (!allowed && (_command_in_category(arg_str + 1, GRAPH_READONLY_USER, 
			&allowed) != REDISMODULE_OK)) {
				goto cleanup;
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

	cleanup:
		rm_free(argv);
		rm_free(acl_args);
		*argv_ptr = NULL;
		*argc = 0;
		return REDISMODULE_ERR;

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
		
	// printf("delegating to execute ACL command '%s'\n",log_msg);	
		
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

	if (argc < 2) {
		return RedisModule_WrongArity(ctx);
	}

 	const char *command = RedisModule_StringPtrLen(argv[1], NULL);
	if (strcasecmp(command, "SETUSER") == 0) {
		if (argc < 3) {
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
		if (argc != 2) {
			return RedisModule_WrongArity(ctx);
		}
		return _execute_acl_cmd_as_admin(ctx, argv, argc);
	} else if (strcasecmp(command, "GETUSER") == 0) {
		if (argc < 3) {
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

// free individual graph ACL category
static void free_command_category
(
	CommandCategory **category  // category to free
) {
	if(category == NULL) {
		return;
	}

	CommandCategory *_category = *category;
	ASSERT(_category != NULL);

	// free commands
	if(_category->commands != NULL) {
		// free each command
		for(int i = 0; i < _category->length; i++) {
			// command name
			if(_category->commands[i] != NULL) {
				rm_free(_category->commands[i]);
			}

			// plus variation
			if(_category->redis_module_commands_plus[i] != NULL) {
				RedisModule_FreeString(NULL,
						_category->redis_module_commands_plus[i]);
			}

			// minus variation
			if(_category->redis_module_commands_minus[i] != NULL) {
				RedisModule_FreeString(NULL,
						_category->redis_module_commands_minus[i]);
			}
		}

		// free arrays
		rm_free(_category->commands);
		rm_free(_category->redis_module_commands_plus);
		rm_free(_category->redis_module_commands_minus);
	}

	// deallocate data structure
	rm_free(_category);

	// nullify input
	*category = NULL;
}

// free the resources consumed by the ACL command
void free_cmd_acl(void) {
	free_command_category(&ACL_GRAPH_USER);
	free_command_category(&ACL_GRAPH_ADMIN);
	free_command_category(&ACL_GRAPH_READONLY_USER);
}

