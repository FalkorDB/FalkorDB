/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "resultset_statistics.h"
#include <string.h>

bool ResultSetStat_IndicateModification
(
	const ResultSetStatistics *stats
) {
	ASSERT(stats != NULL);

	return (
			stats->labels_added          |
			stats->nodes_created         |
			stats->nodes_deleted         |
			stats->properties_set        |
			stats->labels_removed        |
			stats->indices_deleted       |
			stats->indices_created       |
			stats->properties_removed    |
			stats->relationships_created |
			stats->relationships_deleted
		);
}

// initialize resultset statistics
void ResultSetStat_init
(
	ResultSetStatistics *stats  // resultset statistics to initialize
) {
	ASSERT(stats != NULL);

	memset(stats, 0, sizeof(ResultSetStatistics));
}

void ResultSetStat_Clear
(
	ResultSetStatistics *stats
) {
	ASSERT(stats != NULL);

	stats->index_creation = false;
	stats->index_deletion = false;
	stats->constraint_creation = false;
	stats->constraint_deletion = false;
	stats->labels_added          = 0;
	stats->nodes_deleted         = 0;
	stats->nodes_created         = 0;
	stats->properties_set        = 0;
	stats->labels_removed        = 0;
	stats->indices_created       = 0;
	stats->indices_deleted       = 0;
	stats->constraints_created   = 0;
	stats->constraints_deleted   = 0;
	stats->properties_removed    = 0;
	stats->relationships_created = 0;
	stats->relationships_deleted = 0;
}

