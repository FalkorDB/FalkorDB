/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "resultset_formatter.h"
#include "resultset_replynop.h"
#include "resultset_replybolt.h"
#include "resultset_replycompact.h"
#include "resultset_replyverbose.h"

typedef enum {
	FORMATTER_NOP = 0,
	FORMATTER_VERBOSE = 1,
	FORMATTER_COMPACT = 2,
	FORMATTER_BOLT    = 3,
} ResultSetFormatterType;

/* Retrieves result-set formatter.
 * Returns NULL for an unknown formatter type. */
ResultSetFormatter* ResultSetFormatter_GetFormatter(ResultSetFormatterType t);

/* Reply formater which does absolutely nothing.
 * used when profiling a query */
static ResultSetFormatter ResultSetNOP __attribute__((used)) = {
	.EmitRow = ResultSet_EmitNOPRow,
	.EmitStats = ResultSet_EmitNOPStats,
	.EmitHeader = ResultSet_EmitNOPHeader
};

/* Compact reply formatter, this is the default formatter. */
static ResultSetFormatter ResultSetFormatterCompact __attribute__((used)) = {
	.EmitRow = ResultSet_EmitCompactRow,
	.EmitStats = ResultSet_EmitCompactStats,
	.EmitHeader = ResultSet_ReplyWithCompactHeader
};

/* Verbose reply formatter, used when querying via CLI. */
static ResultSetFormatter ResultSetFormatterVerbose __attribute__((used)) = {
	.EmitRow = ResultSet_EmitVerboseRow,
	.EmitStats = ResultSet_EmitVerboseStats,
	.EmitHeader = ResultSet_ReplyWithVerboseHeader
};

/* Bolt reply formatter, used when querying via bolt driver. */
static ResultSetFormatter ResultSetFormatterBolt __attribute__((used)) = {
	.EmitRow = ResultSet_EmitBoltRow,
	.EmitStats = ResultSet_EmitBoltStats,
	.EmitHeader = ResultSet_ReplyWithBoltHeader
};

