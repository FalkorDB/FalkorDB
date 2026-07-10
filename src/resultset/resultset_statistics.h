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

	int labels_added;           // incremented each time a label is added to a
								// node this happens during CREATE
								// (each label on the new node counts)
								// or explicitly via SET n:LabelName.

	int labels_removed;         // incremented each time a label is removed from
								// a node triggered by REMOVE n:LabelName
								// trying to remove a label that doesn't exist
								// on the node does not increment the counter

	int nodes_created;          // incremented each time a new node is created
								// in the graph
								// triggered by CREATE (n) or a MERGE that
								// doesn't find a match and creates the node
								// each node counts as +1, regardless of how
								// many properties or labels it has

	int nodes_deleted;          // incremented each time a node is removed from
								// the graph. Triggered by DELETE n


	int properties_set;         // incremented each time a property is written
								// on a node or relationship
								// whether it's a new property being added, or
								// an existing one being updated
								// setting a property to null (which removes it)
								// doesn't counts

	int properties_removed;     // number of properties removed as part of a remove query


	int relationships_created;  // incremented each time a new relationship is
								// created triggered by CREATE or an unmatched
								// MERGE on a relationship pattern
								// each counts as +1

	int relationships_deleted;  // incremented each time a relationship is
								// removed triggered explicitly by DELETE r on
								// a relationship variable, or implicitly when
								// using DELETE on a node

	int indices_created;        // number of indices created
	int indices_deleted;        // number of indices deleted
	int constraints_created;    // number of constraints created
	int constraints_deleted;    // number of constraints deleted
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
