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
	ASSERT (stats != NULL) ;

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
	ASSERT (stats != NULL) ;
	memset (stats, 0, sizeof (ResultSetStatistics)) ;
}

void ResultSetStat_Clear
(
	ResultSetStatistics *stats
) {
	ASSERT (stats != NULL) ;
	memset (stats, 0, sizeof (ResultSetStatistics)) ;
}

