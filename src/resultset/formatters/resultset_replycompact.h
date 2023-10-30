/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// Formatter for compact (client-parsed) replies
void ResultSet_ReplyWithCompactHeader
(
	ResultSet *set
);

void ResultSet_EmitCompactRow
(
	ResultSet *set,
	SIValue **row
);

void ResultSet_EmitCompactStats
(
	ResultSet *set
);
