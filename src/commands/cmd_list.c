/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../globals.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include "../deps/oniguruma/src/oniguruma.h"

#include <ctype.h>
#include <stdbool.h>
#include <string.h>

// matches a string against a regex pattern
// returns true if the string matches the pattern, false otherwise
static bool _stringmatch_regex
(
	regex_t *regex,  // regex pattern
	const char *s,   // string to match
	int l            // string length
) {
    return onig_match(regex, (const UChar *)s, (const UChar *)(s + l),
			(const UChar *)s, NULL, ONIG_OPTION_DEFAULT) >= 0;
}

// list graphs in DB
int Graph_List
(
	RedisModuleCtx *ctx,       // redis module context
	RedisModuleString **argv,  // command arguments
	int argc                   // number of arguments
) {
	ASSERT(ctx != NULL);

	// expecting up to 1 argument: command name + optional argument
	if(argc > 2) {
		return RedisModule_WrongArity(ctx);
	}

	regex_t *regex = NULL;

	// pattern specified
	if(argc == 2) {
		OnigErrorInfo einfo;
		const char *pattern = RedisModule_StringPtrLen(argv[1], NULL);

		// try to compile the regex expression
		int rv = onig_new(&regex, (const UChar *)pattern,
				(const UChar *)(pattern + strlen(pattern)), ONIG_OPTION_DEFAULT,
				ONIG_ENCODING_UTF8, ONIG_SYNTAX_JAVA, &einfo);

		// failed to compile regex
		// reply with error
		if(rv != ONIG_NORMAL) {
			char err_msg[ONIG_MAX_ERROR_MESSAGE_LEN];
			onig_error_code_to_str((UChar* )err_msg, rv, &einfo);
			RedisModule_ReplyWithError(ctx, err_msg);

			onig_free(regex);
			return REDISMODULE_OK;
		}

		ASSERT(regex != NULL);
	}

	//--------------------------------------------------------------------------
	// compose response
	//--------------------------------------------------------------------------

	// iterate over each graph in the keyspace
	KeySpaceGraphIterator it;
	Globals_ScanGraphs(&it);
	RedisModule_ReplyWithArray(ctx, REDISMODULE_POSTPONED_LEN);

	// reply with each graph name
	uint64_t      n  = 0;     // result-set size
	GraphContext *gc = NULL;  // current graph

	while((gc = GraphIterator_Next(&it)) != NULL) {
		const char *name = GraphContext_GetName(gc);
        size_t len = strlen(name);

		// match graph name against the regex pattern if specified
		if(regex == NULL || _stringmatch_regex(regex, name, len)) {
			RedisModule_ReplyWithStringBuffer(ctx, name, len);
			n++;
		}

		// dec graph's ref count
		GraphContext_DecreaseRefCount(gc);
	}

	// update result-set size
	RedisModule_ReplySetArrayLength(ctx, n);

	// clean up
	if(regex != NULL) {
		onig_free(regex);
	}

	return REDISMODULE_OK;
}

