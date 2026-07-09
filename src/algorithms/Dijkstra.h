/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/graph.h"
#include "../datatypes/datatypes.h"

// exact single-shortest-path search via Dijkstra (label-setting, one
// best-known weight per node, each node finalized exactly once).
//
// only valid when there is no maxCost constraint: with cost unconstrained,
// weight alone determines optimality, so classic per-node dedup applies
// and this always terminates in O((V+E) log V) -- unlike exhaustive DFS
// enumeration, which can blow up combinatorially on graphs with many
// similar-weight alternative routes (e.g. dense/mesh-like road networks).
//
// ASSUMES weightProp is non-negative for every edge. Dijkstra's
// "finalize once, never revisit" invariant is unsound with negative
// weights (a node reached later via a heavier edge can hold a negative
// edge that retroactively beats an already-finalized node), and this is
// NOT detected or guarded against here: making that safe would require
// giving up the early-termination-at-dst optimization (running to full
// completion over the whole reachable component instead), which was
// judged not worth it given weightProp values are expected to represent
// real, non-negative quantities (distance, time, cost) in practice. if
// negative weights are ever a real requirement, this function must not
// be used as-is.
//
// returns true and populates 'path' and 'weight' if 'dst' is reachable
// from 'src'; returns false (leaving them untouched) otherwise.
//
// only 'weight' is computed -- it's the metric Dijkstra actually
// optimizes for. any other per-path attribute (e.g. a "cost" property)
// isn't part of the search and should be summed by the caller over the
// returned path's edges instead.
bool Dijkstra_ShortestPath
(
	Path **path,               // [output] src -> dst path
	double *weight,            // [output] total path weight
	Graph *g,                  // graph to traverse
	NodeID src_id,             // source node
	NodeID dst_id,             // destination node
	GRAPH_EDGE_DIR dir,        // traverse direction
	RelationID *relationIDs,   // edge type(s) to traverse
	Tensor *relationMatrices,  // relation matrix per relationIDs entry
	int relationCount,         // length of relationIDs
	AttributeID weight_prop    // weight attribute id
);

