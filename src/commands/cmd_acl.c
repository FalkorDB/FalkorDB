/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "cmd_acl.h"
#include "../globals.h"
#include "../redismodule.h"
#include "../util/sds/sds.h"
#include "../graph/graphcontext.h"
#include "./util/run_redis_command_as.h"

#include <stdlib.h>
#include <stdbool.h>

// data structure describing an ACL category
// a category contains a name and a set of commands which can be either
// enabled or disabled
typedef struct {
	const char        *name;                          // categoty name
	char              **commands;                     // commands in categoty
	RedisModuleString **redis_module_commands_plus;   // commands in category
	                                                  // prefix with +
	RedisModuleString **redis_module_commands_minus;  // commands in category 
	                                                  // prefix with -
} CommandCategory;

// free individual graph ACL category
static void free_command_category
(
	CommandCategory **category  // category to free
);

// create a new RedisModuleString with the given prefix
// and the given command
// the prefix is a single character that will be added to the command
// the command is a string that will be copied to the new RedisModuleString
// returns NULL on failure
// the caller is responsible for freeing the returned RedisModuleString
RedisModuleString *_create_command_with_prefix
(
	RedisModuleCtx *ctx, // redis module context
	char  ch,            // the prefix that will be attached to the command
	const char *cmd      // the string representing the command
);

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
static CommandCategory *_create_command_category
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

	// parse the (space separated) commands string to array of commands
	category->commands = array_new(char*, 0);
	category->redis_module_commands_minus = array_new(RedisModuleString*, 0);
	category->redis_module_commands_plus  = array_new(RedisModuleString*, 0);

	char *token, *saveptr;
	char *commands_copy = rm_strdup(commands_str);

	if(commands_copy == NULL) goto cleanup;

	// split commands using strtok
	// and fill the commands array, the minus and plus arrays
	for(token = strtok_r(commands_copy, " ", &saveptr);
		token != NULL;
		token = strtok_r(NULL, " ", &saveptr)) {
        
		// skip empty tokens from multiple spaces
		if(strlen(token) == 0) continue;
        
		char *cmd = rm_strdup(token);
		if(cmd == NULL) {
			rm_free(commands_copy);
			goto cleanup;
		}

		array_append(category->commands, cmd);
	
		// add the command to the minus and plus arrays
		array_append(category->redis_module_commands_minus,
			_create_command_with_prefix(ctx, '-', cmd));
		array_append(category->redis_module_commands_plus,
			_create_command_with_prefix(ctx, '+', cmd));
	}

    rm_free(commands_copy);
	return category;
	
	cleanup:
		free_command_category(&category);
		return NULL;
}

// create a new RedisModuleString with the given prefix
// and the given command
// the prefix is a single character that will be added to the command
// the command is a string that will be copied to the new RedisModuleString
// returns NULL on failure
// the caller is responsible for freeing the returned RedisModuleString
RedisModuleString *_create_command_with_prefix
(
	RedisModuleCtx *ctx, // redis module context
	char ch,             // the prefix that will be attached to the command
	const char *cmd      // the string representing the command
) {

	ASSERT(cmd != NULL);
	ASSERT(ch == '-' || ch == '+');

	RedisModuleString *str = 
		RedisModule_CreateStringPrintf(ctx, "%c%s", ch, cmd);
	
	return str;
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

	if ((ACL_GRAPH_USER = 
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
	if ((ACL_GRAPH_ADMIN = 
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

	for(int i = 0; i < array_len(category->commands); i++) {
		if(strcasecmp(cmd, category->commands[i]) == 0) {
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
			if(strncasecmp(cmd, category->commands[i], prefix_len) == 0) {
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
// example, for acl_args:
// ["GRAPH.ACL", "SETUSER", "falkordb-admin", "on", ">pass", "+@graph-admin"]
// +@graph-admin should expand to all the commands in the
// environment variable ACL_GRAPH_ADMIN prefix with +
// assuming ACL_GRAPH_ADMIN = "INFO CLIENT DBSIZE PING HELLO AUTH"
// the expanded acl_args will be:
// ["GRAPH.ACL", "SETUSER", "falkordb-admin", "on", ">pass", "+INFO", "+CLIENT",
// "+DBSIZE", "+PING", "+HELLO", "+AUTH"]
static bool _expand_acl_pseudo_category
(
    RedisModuleString **acl_args,  // input/output the command args
    int *acl_argc,                 // [input/output] the commands args count
    RedisModuleString *arg,        // the argument that consider expending
    CommandCategory *category      // the category that is used in this expand
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

	// arg refers to category
	int n = *acl_argc;
	for(int i = 0; i < array_len(category->commands); i++) {
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
	RedisModuleString **acl_args,   // [input/output] ACL command args
	int acl_argc                    // number of elements in acl_args
) {
	ASSERT(ctx                     != NULL);
	ASSERT(acl_args                != NULL);
	ASSERT(ACL_GRAPH_USER          != NULL);
	ASSERT(ACL_GRAPH_ADMIN         != NULL);
	ASSERT(ACL_GRAPH_READONLY_USER != NULL);

	int res = 0;
	for(int i = 0; i < acl_argc; i++) {
		const char *arg_str = RedisModule_StringPtrLen(acl_args[i], NULL);

		// if arg doesn't begins with either '+' or '-' then this is not
		// a command and we should ignore
		if((arg_str[0] != '+' && arg_str[0] != '-')) {
			continue;
		}

		// is this a graph user categoty?
		if(strcasecmp(arg_str + 1, ACL_GRAPH_USER->name) == 0) {
			res = res + array_len(ACL_GRAPH_USER->commands);
			continue;
		}

		// is this a graph admin categoty?
		else if(strcasecmp(arg_str + 1, ACL_GRAPH_ADMIN->name) == 0) {
			res = res + array_len(ACL_GRAPH_ADMIN->commands);
			continue;
		}

		// is this a graph readonly user categoty?
		else if(strcasecmp(arg_str + 1, ACL_GRAPH_READONLY_USER->name) == 0) {
			res = res + array_len(ACL_GRAPH_READONLY_USER->commands);
			continue;	
		}
	}

	return res;
}

// prepare arguments for redis ACL SETUSER command, remove permissions that
// are not in ACL_GRAPH_ADMIN, ACL_GRAPH_READONLY_USER or ACL_GRAPH_USER 
// argv is of the form ["SETUSER", "user1", "on", "+@admin", ...]
// we should copy every argument that does not start with a "+" or "-"
// we should *NOT* copy any argument that, when stripped of the "+" or "-",
// is not in ACL_GRAPH_ADMIN, ACL_GRAPH_READONLY_USER or ACL_GRAPH_USER 
// (case insensitive) we should *NOT* copy any argument that
// after being stripped of the "+" or "-" does not start with one of the
// values from ACL_GRAPH_ADMIN, ACL_GRAPH_READONLY_USER or ACL_GRAPH_USER
// followed by '|' (case insensitive)
//
// this function allocates a new RedisModuleString ***argv_ptr and releases 
// the old one
// also expand the pseudo categories @graph-admin, @graph-user and
// @graph-readonly-user
// return REDISMODULE_OK on success, REDISMODULE_ERR on failure
static int _senitaze_acl_setuser
(
	RedisModuleCtx *ctx,            // the redis module context
	RedisModuleString ***argv_ptr,  // [input/output] a pointer to the arguments
	                                // to be sanitized
	int *argc                       // number of arguments
) {
	ASSERT(*argc                   > 0);
	ASSERT(ctx                     != NULL);
	ASSERT(argc                    != NULL);
	ASSERT(argv_ptr                != NULL);
	ASSERT(ACL_GRAPH_USER          != NULL);
	ASSERT(ACL_GRAPH_ADMIN         != NULL);
	ASSERT(ACL_GRAPH_READONLY_USER != NULL);
	
	int expand_offset = _compute_expand_offset(ctx, *argv_ptr, *argc);

	RedisModuleString **argv = *argv_ptr;
	RedisModuleString **acl_args = 
		rm_malloc(sizeof(RedisModuleString*) * (*argc + expand_offset));

	if(acl_args == NULL) {
		RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING, 
		"Failed to allocate memory for ACL arguments.");
		return REDISMODULE_ERR;
	}

	int acl_argc = 0;
	// iterate over the items in GRAPH_ADMIN
	// ACL_GRAPH_USER and ACL_GRAPH_READONLY_USER and use _command_in_category
	// to check if the command is allowed
	for(int i = 0; i < *argc; i++) {
		const char *arg_str = RedisModule_StringPtrLen(argv[i], NULL);
		if(arg_str[0] != '+' && arg_str[0] != '-') {
			acl_args[acl_argc++] = argv[i];
			continue;
		}

		// skip '+' | '-' sign
		arg_str++;

		// if it is one of the pseudo categories 
		// (@graph-admin, @graph-user or @graph-readonly-user), expand it
		// and add the commands to acl_args
		if(_expand_acl_pseudo_category(acl_args, &acl_argc, argv[i],
			ACL_GRAPH_ADMIN)) {
			continue;
		}

		if(_expand_acl_pseudo_category(acl_args, &acl_argc, argv[i],
			ACL_GRAPH_USER)) {
			continue;
		}

		if(_expand_acl_pseudo_category(acl_args, &acl_argc, argv[i],
			ACL_GRAPH_READONLY_USER)) {
			continue;
		}

		// check if the command is in the graph admin category
		bool allowed = _command_in_category(arg_str, ACL_GRAPH_ADMIN); 
		
		// same for ACL_GRAPH_USER
		allowed = allowed || _command_in_category(arg_str, ACL_GRAPH_USER);

		// same for ACL_GRAPH_READONLY_USER
		allowed = allowed || _command_in_category(arg_str,
				ACL_GRAPH_READONLY_USER);

		// if the command is in any of the categories
		// add it to acl_args otherwise skip it
		if(allowed) {
			acl_args[acl_argc++] = argv[i];
		}
	}

	// update argv_ptr, do not free argv, it will free by redis
	*argc     = acl_argc;
	*argv_ptr = acl_args;

	return REDISMODULE_OK;

	cleanup:
		rm_free(acl_args);
		*argc     = 0;
		*argv_ptr = NULL;
		return REDISMODULE_ERR;
}

// this function should be called with the ADMIN_USER in context
// the username is the user that calls the GRAPH.ACL SETUSER command
// the command uses the Redis module high-level API to call the Redis 
// ACL SETUSER command
// an example of argv is: ["GRAPH.ACL", "SETUSER", "user1", "+@admin"]
// argv can be of arbitrary length because the user can stack multiple ACL 
// commands as can be done in the Redis ACL SETUSER command
// in case the command is SETUSER, it uses 
// remove_setuser_forbidden_permissions to clean the arguments from permissions 
// that are not allowed for graph users
static int _execute_acl_cmd_fn
(
	RedisModuleCtx *ctx,       // the redis module context
	RedisModuleString **argv,  // the arguments to the command
	int argc,                  // the number of arguments
	void *privdata             // optional private data
) {
	ASSERT(argc     > 0);
	ASSERT(ctx      != NULL);
	ASSERT(argv     != NULL);

	// construct the correct arguments for RedisModule_Call
	// remove the GRAPH.ACL part from the arguments

	argv++;
	argc--;
	
	bool should_free_argv = false;

	// if the subcommand is SETUSER, we need to filter argv to 
	// remove permissions that are not allowed
	// expand @graph-user, @graph-admin and @graph-readonly-user
	if(strcasecmp(RedisModule_StringPtrLen(argv[0], NULL), "SETUSER") == 0) {
		if(_senitaze_acl_setuser(ctx, &argv, &argc) 
			!= REDISMODULE_OK) {
			RedisModule_ReplyWithError(ctx, "FAILED");
			return REDISMODULE_ERR;
		}
		should_free_argv = true;
	}
	
	// just for log level debug
#ifdef RG_DEBUG
	sds msg = sdsempty();
	// append each argument to the log message
	for(int i = 0; i < argc; i++) {
		const char *arg_str = RedisModule_StringPtrLen(argv[i], NULL);
		if(i > 0) {
			msg = sdscat(msg, " ");
		}
		msg = sdscat(msg, arg_str);
	}

	RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_DEBUG,
		"delegating to execute ACL command '%s'", msg);
	sdsfree(msg);
#endif

    RedisModule_ReplicateVerbatim(ctx);	

	RedisModuleCallReply *reply = 
		RedisModule_Call(ctx, "ACL", "v", argv, argc);
 	if(reply == NULL) {
		RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING,
			"Failed to execute ACL command, Error: %d", errno);
		RedisModule_ReplyWithError(ctx, "FAILED");
		if(should_free_argv) {
			rm_free(argv);
		}
		return REDISMODULE_ERR;
	}

	RedisModule_ReplyWithCallReply(ctx, reply);
	RedisModule_FreeCallReply(reply);

	if(should_free_argv) {
		rm_free(argv);
	}

	return REDISMODULE_OK;
}

// call execute_acl_userset_func with the ADMIN_USER in context
static int _execute_acl_cmd_as_admin
(
	RedisModuleCtx *ctx,       // the redis module context
	RedisModuleString **argv,  // the arguments to call
	int argc                   // the number of arguments
) {
	ASSERT(ctx  != NULL);
	ASSERT(argv != NULL);

	return run_acl_function_as(ctx, argv, argc, _execute_acl_cmd_fn,
		ACL_ADMIN_USER, NULL);
}

// this function is the main entry point for the GRAPH.ACL command
// it manipulates the arguments, impersonate and calls redis ACL command
int graph_acl_cmd
(
	RedisModuleCtx *ctx,       // the redis module context
	RedisModuleString **argv,  // the arguments to the command
	int argc                   // the number of arguments
) {
	ASSERT(ctx  != NULL);
	ASSERT(argv != NULL);

	if(argc < 2) {
		return RedisModule_WrongArity(ctx);
	}

 	const char *command = RedisModule_StringPtrLen(argv[1], NULL);

	if(strcasecmp(command, "SETUSER") == 0) {
		if (argc < 3) {
			return RedisModule_WrongArity(ctx);
		}

		RedisModuleString *sub = argv[2];

		// if the sub is ADMIN_USER return
		if(strcasecmp(RedisModule_StringPtrLen(sub, NULL), ACL_ADMIN_USER) == 0) 
		{
			RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING, 
				"Cannot change the ACL of the admin user.");
			RedisModule_ReplyWithError(ctx, "FAILED");
			return REDISMODULE_ERR;
		}

		// execute the command as admin using redis ACL
		return _execute_acl_cmd_as_admin(ctx, argv, argc);

	} else if(strcasecmp(command, "SAVE") == 0) {
		if(argc != 2) {
			return RedisModule_WrongArity(ctx);
		}

		return _execute_acl_cmd_as_admin(ctx, argv, argc);
	} else if(strcasecmp(command, "GETUSER") == 0) {
		if(argc < 3) {
			return RedisModule_WrongArity(ctx);
		}
	
		RedisModuleString *sub = argv[2];
		// if the sub is ADMIN_USER return
		if(strcmp(RedisModule_StringPtrLen(sub, NULL), ACL_ADMIN_USER) == 0) {
			RedisModule_Log(ctx, REDISMODULE_LOGLEVEL_WARNING,
				"Cannot read the ACL of the admin user.");
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
	if(category == NULL || *category == NULL) {
		return;
	}

	CommandCategory *_category = *category;
	ASSERT(_category != NULL);

	// free commands
	if(_category->commands != NULL) {
		// free each command
		for(int i = 0; i < array_len(_category->commands); i++) {
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
		array_free(_category->commands);
		array_free(_category->redis_module_commands_plus);
		array_free(_category->redis_module_commands_minus);
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

