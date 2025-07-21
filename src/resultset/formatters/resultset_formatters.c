/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "resultset_formatters.h"

ResultSetFormatter *ResultSetFormatter_GetFormatter(ResultSetFormatterType t) {
	ResultSetFormatter *formatter = NULL;
	switch(t) {
	case FORMATTER_NOP:
		formatter = &ResultSetNOP;
		break;
	case FORMATTER_VERBOSE:
		formatter = &ResultSetFormatterVerbose;
		break;
	case FORMATTER_COMPACT:
		formatter = &ResultSetFormatterCompact;
		break;
	default:
		RedisModule_Assert(false && "Unknown formatter");
	}

	return formatter;
}

