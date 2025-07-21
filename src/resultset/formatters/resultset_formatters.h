/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "resultset_formatter.h"
#include "resultset_replynop.h"
#include "resultset_replycompact.h"
#include "resultset_replyverbose.h"

typedef enum {
	FORMATTER_NOP = 0,
	FORMATTER_VERBOSE = 1,
	FORMATTER_COMPACT = 2,
} ResultSetFormatterType;

/* Retrieves result-set formatter.
 * Returns NULL for an unknown formatter type. */
ResultSetFormatter* ResultSetFormatter_GetFormatter(ResultSetFormatterType t);

/* Reply formater which does absolutely nothing.
 * used when profiling a query */
static ResultSetFormatter ResultSetNOP __attribute__((used)) = {
	.EmitRow = ResultSet_EmitNOPRow,
	.EmitHeader = ResultSet_EmitNOPHeader
};

/* Compact reply formatter, this is the default formatter. */
static ResultSetFormatter ResultSetFormatterCompact __attribute__((used)) = {
	.EmitRow = ResultSet_EmitCompactRow,
	.EmitHeader = ResultSet_ReplyWithCompactHeader
};

/* Verbose reply formatter, used when querying via CLI. */
static ResultSetFormatter ResultSetFormatterVerbose __attribute__((used)) = {
	.EmitRow = ResultSet_EmitVerboseRow,
	.EmitHeader = ResultSet_ReplyWithVerboseHeader
};

