/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/graph.h"
#include "../datatypes/datatypes.h"

// find up to 'k' shortest loopless paths from src to dst by ascending weight,
// via Yen's algorithm using Dijkstra (see Dijkstra.h) as the single-pair
// subroutine.
//
// this is the efficient replacement for exhaustively enumerating simple paths
// and keeping the k lightest: it computes the shortest path, then repeatedly
// derives the next-shortest by "spurring" off each node of the previous one --
// searching a subgraph with the relevant root nodes/edges virtually removed
// (via Dijkstra's blocked-set support) so it can't rediscover an earlier path.
//
// ASSUMES weightProp is non-negative for every edge (Dijkstra precondition,
// see Dijkstra.h).
//
// candidate paths are deduplicated by a 64-bit hash of their edge-id sequence;
// with a good hash over the modest number of paths any k-shortest query
// produces, a false collision (which would drop a distinct path) is
// astronomically unlikely.
//
// returns the number of paths found (<= k; 0 if dst is unreachable). '*paths'
// and '*weights' are set to newly allocated parallel array_t buffers (Path*
// and its total weight, ascending); the caller owns both arrays and each Path.
uint Yen_KShortestPaths
(
	Graph *g,                  // graph to traverse
	NodeID src,                // source node
	NodeID dst,                // destination node
	uint64_t k,                // number of paths to find
	GRAPH_EDGE_DIR dir,        // traverse direction
	RelationID *relationIDs,   // edge type(s) to traverse
	Tensor *relationMatrices,  // relation matrix per relationIDs entry
	int relationCount,         // length of relationIDs
	AttributeID weight_prop,   // weight attribute id
	Path ***paths,             // [output] array_t of Path*, ascending weight
	double **weights           // [output] array_t of matching total weights
);
