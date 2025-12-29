/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// Formatter for bolt replies

// emit a header
void ResultSet_ReplyWithBoltHeader
(
	ResultSet *set
);

// emit a row
void ResultSet_EmitBoltRow
(
	ResultSet *set,
	SIValue *row
);

// emit statistics
void ResultSet_EmitBoltStats
(
	ResultSet *set
);

