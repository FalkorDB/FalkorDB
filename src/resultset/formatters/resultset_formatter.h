/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../../redismodule.h"
#include "../../graph/graphcontext.h"
#include "../../graph/query_graph.h"

typedef enum {
	COLUMN_UNKNOWN = 0,
	COLUMN_SCALAR = 1,
	COLUMN_NODE = 2,      // Unused, retained for client compatibility.
	COLUMN_RELATION = 3,  // Unused, retained for client compatibility.
} ColumnType;

typedef enum {
	VALUE_UNKNOWN = 0,
	VALUE_NULL = 1,
	VALUE_STRING = 2,
	VALUE_INTEGER = 3,
	VALUE_BOOLEAN = 4,
	VALUE_DOUBLE = 5,
	VALUE_ARRAY = 6,
	VALUE_EDGE = 7,
	VALUE_NODE = 8,
	VALUE_PATH = 9,
	VALUE_MAP = 10,
	VALUE_POINT = 11
} ValueType;

// Typedef for header formatters.
typedef void (*EmitHeaderFunc)(RedisModuleCtx *ctx, const char **columns,
							   uint *col_rec_map);

// Typedef for row formatters.
typedef void (*EmitRowFunc)(RedisModuleCtx *ctx, GraphContext *gc,
		SIValue **row, uint numcols);
							   
typedef struct {
	EmitRowFunc    EmitRow;
	EmitHeaderFunc EmitHeader;
} ResultSetFormatter;

/* Redis prints doubles with up to 17 digits of precision, which captures
 * the inaccuracy of many floating-point numbers (such as 0.1).
 * By using the %g format and a precision of 15 significant digits, we avoid many
 * awkward representations like RETURN 0.1 emitting "0.10000000000000001",
 * though we're still subject to many of the typical issues with floating-point error. */
static inline void _ResultSet_ReplyWithRoundedDouble(RedisModuleCtx *ctx, double d) {
	// Get length required to print number
	int len = snprintf(NULL, 0, "%.15g", d);
	char str[len + 1]; // TODO a reusable buffer would be far preferable
	sprintf(str, "%.15g", d);
	// Output string-formatted number
	RedisModule_ReplyWithStringBuffer(ctx, str, len);
}
