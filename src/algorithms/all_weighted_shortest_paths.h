/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/graph.h"
#include "../datatypes/datatypes.h"

// enumerate *all* minimum-weight loopless paths from src to dst.
//
// this is the weighted twin of all_shortest_paths.c (which enumerates all
// paths of minimum hop-count): it returns every path whose total weightProp
// sum equals the minimum achievable, not just one.
//
// runs a bidirectional Dijkstra -- forward from src, backward from dst -- to
// obtain d_src(v) for all v and the minimum weight D = d_src(dst), then
// enumerates the shortest-path DAG: it follows only edges (u->v) that are
// tight (d_src(u)+w(u,v) == d_src(v)) and lead to a node that can still reach
// dst. every such path has weight exactly D and every prefix is a genuine
// shortest sub-path, so there is no dead-end exploration.
//
// ASSUMES weightProp is non-negative for every edge (Dijkstra precondition,
// see Dijkstra.h). with strictly positive weights the DAG is acyclic; a
// Path_ContainsNode guard covers the zero-weight-edge corner case.
//
// NOTE: tightness is tested with exact floating-point equality, matching the
// existing exhaustive all-minimal DFS. for integer/exact weights this is
// precise; for arbitrary doubles it inherits the same tie-sensitivity.
//
// returns the number of paths found (0 if dst is unreachable). '*paths' is set
// to a newly allocated array_t of Path* (all of weight '*min_weight'); the
// caller owns the array and each Path in it. 'min_weight' is left untouched
// when 0 is returned.
uint AllWeightedShortestPaths
(
	Graph *g,                  // graph to traverse
	NodeID src,                // source node
	NodeID dst,                // destination node
	GRAPH_EDGE_DIR dir,        // traverse direction
	RelationID *relationIDs,   // edge type(s) to traverse
	Tensor *relationMatrices,  // relation matrix per relationIDs entry
	int relationCount,         // length of relationIDs
	AttributeID weight_prop,   // weight attribute id
	Path ***paths,             // [output] array_t of Path*, caller owns
	double *min_weight         // [output] the minimum weight D
);
