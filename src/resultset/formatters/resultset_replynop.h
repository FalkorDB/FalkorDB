/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

// Formatter for no operation (replication) replies

// emit a header
void ResultSet_EmitNOPHeader
(
    ResultSet *set
);

// emit a row
void ResultSet_EmitNOPRow
(
    ResultSet *set,
    SIValue **row
);

// emit statistics
void ResultSet_EmitNOPStats
(
    ResultSet *set
);
