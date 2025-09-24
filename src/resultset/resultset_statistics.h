/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include <stdbool.h>
#include "../redismodule.h"

typedef struct {
	bool cached;                // indication for a cached query execution
	bool index_creation;        // index creation operation executed
	bool index_deletion;        // index deletion operation executed
	bool constraint_creation;   // constraint creation operation executed
	bool constraint_deletion;   // constraint deletion operation executed
	int labels_added;           // number of labels added as part of a create/update query
	int nodes_created;          // number of nodes created as part of a create query
	int nodes_deleted;          // number of nodes removed as part of a delete query
	int labels_removed;         // number of labels removed as part of an update query
	int properties_set;         // number of properties created as part of a create query
	int indices_created;        // number of indices created
	int indices_deleted;        // number of indices deleted
	int constraints_created;    // number of constraints created
	int constraints_deleted;    // number of constraints deleted
	int properties_removed;     // number of properties removed as part of a remove query
	int relationships_created;  // number of edges created as part of a create query
	int relationships_deleted;  // number of edges removed as part of a delete query
} ResultSetStatistics;

// Checks to see if resultset-statistics indicate that a modification was made
bool ResultSetStat_IndicateModification
(
	const ResultSetStatistics *stats // resultset statistics to inquery
);

// initialize resultset statistics
void ResultSetStat_init
(
	ResultSetStatistics *stats  // resultset statistics to initialize
);

// Clear result-set statistics
void ResultSetStat_Clear
(
	ResultSetStatistics *stats
);
