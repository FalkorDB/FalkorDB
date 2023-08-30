/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// Formatter for bolt replies
void ResultSet_ReplyWithBoltHeader
(
	RedisModuleCtx *ctx,
	bolt_client_t *bolt_client,
	const char **columns,
	uint *col_rec_map
);

void ResultSet_EmitBoltRow
(
	RedisModuleCtx *ctx,
	bolt_client_t *bolt_client,
	GraphContext *gc,
	SIValue **row,
	uint numcols
);

