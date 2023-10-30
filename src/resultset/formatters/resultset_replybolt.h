/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// Formatter for bolt replies
void ResultSet_ReplyWithBoltHeader
(
	ResultSet *set
);

void ResultSet_EmitBoltRow
(
	ResultSet *set,
	SIValue **row
);

void ResultSet_EmitBoltStats
(
	ResultSet *set
);
