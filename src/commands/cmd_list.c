/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../globals.h"
#include "../redismodule.h"
#include "../graph/graphcontext.h"
#include <stdbool.h>
#include <string.h>
#include <ctype.h>

// Redis-style glob pattern matcher (supports *, ?, [], [^], and ranges)
//
// e.g.
// h?llo matches hello, hallo and hxllo
// h*llo matches hllo and heeeello
// h[ae]llo matches hello and hallo, but not hillo
// h[^e]llo matches hallo, hbllo, ... but not hello
// h[a-b]llo matches hallo and hbllo
static bool _stringmatchlen(const char *pattern, int patternLen, const char *string, int stringLen, int nocase) {
    while (patternLen) {
        switch (*pattern) {
            case '*':
                pattern++;
                patternLen--;
                if (!patternLen) return 1;
                while (stringLen) {
                    if (_stringmatchlen(pattern, patternLen, string, stringLen, nocase))
                        return 1;
                    string++;
                    stringLen--;
                }
                return 0;
            case '?':
                if (!stringLen) return 0;
                string++;
                stringLen--;
                pattern++;
                patternLen--;
                break;
            case '[': {
                pattern++;
                patternLen--;
                int notMatch = (*pattern == '^');
                if (notMatch) {
                    pattern++;
                    patternLen--;
                }
                int match = 0;
                while (patternLen && *pattern != ']') {
                    if (patternLen > 2 && pattern[1] == '-' && pattern[2] != ']') {
                        char start = pattern[0];
                        char end = pattern[2];
                        if (nocase) {
                            start = tolower(start);
                            end = tolower(end);
                        }
                        char c = *string;
                        if (nocase) c = tolower(c);
                        if (c >= start && c <= end) match = 1;
                        pattern += 3;
                        patternLen -= 3;
                    } else {
                        char c = *string;
                        char p = *pattern;
                        if (nocase) {
                            c = tolower(c);
                            p = tolower(p);
                        }
                        if (c == p) match = 1;
                        pattern++;
                        patternLen--;
                    }
                }
                if (notMatch) match = !match;
                if (!match) return 0;
                while (patternLen && *pattern != ']') {
                    pattern++;
                    patternLen--;
                }
                if (patternLen) {
                    pattern++;
                    patternLen--;
                }
                string++;
                stringLen--;
                break;
            }
            case '\\':
                if (patternLen >= 2) {
                    pattern++;
                    patternLen--;
                }
                // fall through
            default: {
                if (!stringLen) return 0;
                char c1 = *pattern;
                char c2 = *string;
                if (nocase) {
                    c1 = tolower(c1);
                    c2 = tolower(c2);
                }
                if (c1 != c2) return 0;
                pattern++;
                patternLen--;
                string++;
                stringLen--;
                break;
            }
        }
    }
    return stringLen == 0;
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
	if(argc == 2) {
		pattern = RedisModule_StringPtrLen(argv[1], NULL);
	}

	KeySpaceGraphIterator it;
	Globals_ScanGraphs(&it);
	RedisModule_ReplyWithArray(ctx, REDISMODULE_POSTPONED_LEN);

	// reply with each graph name
	uint64_t     n   = 0;
	GraphContext *gc = NULL;

	while((gc = GraphIterator_Next(&it)) != NULL) {
		const char *name = GraphContext_GetName(gc);
		if(pattern == NULL || _stringmatchlen(pattern, strlen(pattern), name, strlen(name), 0)) {
			RedisModule_ReplyWithStringBuffer(ctx, name, strlen(name));
			n++;
		}
		GraphContext_DecreaseRefCount(gc);
	}

	RedisModule_ReplySetArrayLength(ctx, n);
	return REDISMODULE_OK;
}

