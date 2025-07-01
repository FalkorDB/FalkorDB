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
#include <stdbool.h>
#include <string.h>
#include <ctype.h>

// matches a string against a regex pattern
// returns true if the string matches the pattern, false otherwise
static bool _stringmatch_regex(const regex_t *regex, const char *string, int stringLen) {
    if(regex == NULL || string == NULL) return true; // No pattern: always match

    int result = onig_match(
        regex, 
        (const UChar *)string, 
        (const UChar *)(string + stringLen), 
        (const UChar *)string, 
        NULL, 
        ONIG_OPTION_DEFAULT
    );

    return (result >= 0);
}

int Graph_List
(
	RedisModuleCtx *ctx,
	RedisModuleString **argv,
	int argc
) {
	ASSERT(ctx != NULL);

	if(argc != 1 && argc != 2) {
		return RedisModule_WrongArity(ctx);
	}

	const char *pattern = NULL;
	regex_t *regex = NULL;
	if(argc == 2) {
		pattern = RedisModule_StringPtrLen(argv[1], NULL);
		if(pattern) {
			OnigErrorInfo einfo;
			int rv = onig_new(&regex, (const UChar *)pattern, (const UChar *)(pattern + strlen(pattern)), ONIG_OPTION_DEFAULT, ONIG_ENCODING_UTF8, ONIG_SYNTAX_JAVA, &einfo);
			if(rv != ONIG_NORMAL) {
				regex = NULL; // fallback: no match
			}
		}
	}

	KeySpaceGraphIterator it;
	Globals_ScanGraphs(&it);
	RedisModule_ReplyWithArray(ctx, REDISMODULE_POSTPONED_LEN);

	// reply with each graph name
	uint64_t     n   = 0;
	GraphContext *gc = NULL;

	while((gc = GraphIterator_Next(&it)) != NULL) {
		const char *name = GraphContext_GetName(gc);
        size_t len = strlen(name);
		if(_stringmatch_regex(regex, name, len)) {
			RedisModule_ReplyWithStringBuffer(ctx, name, len);
			n++;
		}
		GraphContext_DecreaseRefCount(gc);
	}

	if(regex) onig_free(regex);
	RedisModule_ReplySetArrayLength(ctx, n);
	return REDISMODULE_OK;
}

