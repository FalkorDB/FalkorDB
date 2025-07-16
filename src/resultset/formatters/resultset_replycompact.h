/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

// Formatter for compact (client-parsed) replies
void ResultSet_ReplyWithCompactHeader(RedisModuleCtx *ctx, const char **columns, uint *col_rec_map);

void ResultSet_EmitCompactRow(RedisModuleCtx *ctx, GraphContext *gc,
		SIValue **row, uint numcols);

