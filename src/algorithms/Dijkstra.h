/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/graph.h"
#include "../util/dict.h"
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
	Path **path,                     // [output] src -> dst path
	double *weight,                  // [output] total path weight
	const Graph *g,                  // graph to traverse
	NodeID src_id,                   // source node
	NodeID dst_id,                   // destination node
	GRAPH_EDGE_DIR dir,              // traverse direction
	const RelationID *relationIDs,   // edge type(s) to traverse
	const Tensor *relationMatrices,  // relation matrix per relationIDs entry
	int relationCount,               // length of relationIDs
	AttributeID weight_prop          // weight attribute id
);

//------------------------------------------------------------------------------
// DijkstraCtx: reusable Dijkstra engine
//------------------------------------------------------------------------------

// A reusable Dijkstra search engine. The expensive per-search setup --
// attaching one TensorIterator per (direction, relation) pair -- is done once
// at construction and amortized across many DijkstraCtx_Run calls, which
// re-seek (rather than re-attach) those iterators. This is what makes it
// affordable to run the search O(k*V) times, as Yen's k-shortest-paths does.
//
// The engine supports two modes (selected per-run by DijkstraCtx_Run):
//   - single-pair: stop as soon as 'dst' is finalized (classic early exit).
//   - single-source: run to completion over the whole reachable component,
//     exposing the final distance to every node via DijkstraCtx_Distance
//     (used to build the shortest-path DAG for all-minimal enumeration).
//
// Per-run it also accepts optional blocked node/edge sets (see DijkstraCtx_Run)
// so a caller can search a subgraph with certain nodes/edges removed without
// mutating the graph -- exactly what Yen's spur searches require.
//
// The same non-negative-weight precondition as Dijkstra_ShortestPath applies.
typedef struct DijkstraCtx DijkstraCtx;

// create a reusable Dijkstra engine over the given graph/direction/relations.
// relationIDs and relationMatrices are borrowed (not copied); they must
// outlive the returned context. iterators are attached here, once.
DijkstraCtx *DijkstraCtx_New
(
	const Graph *g,                  // graph to traverse
	GRAPH_EDGE_DIR dir,              // traverse direction
	const RelationID *relationIDs,   // edge type(s) to traverse
	const Tensor *relationMatrices,  // relation matrix per relationIDs entry
	int relationCount,               // length of relationIDs
	AttributeID weight_prop          // weight attribute id
);

// run a search from 'src'. internal scratch (labels/heap/map) is reset at
// entry, so a single context can be run repeatedly.
//
// if 'dst' != INVALID_ENTITY_ID the search stops as soon as 'dst' is
// finalized (single-pair mode) and the return value reports whether 'dst'
// was reached. if 'dst' == INVALID_ENTITY_ID the search runs to completion
// (single-source mode) and always returns true; read results via
// DijkstraCtx_Distance.
//
// blocked_nodes / blocked_edges, when non-NULL, are membership sets (keyed by
// (uintptr_t)id, any non-NULL value means "present", e.g. built with def_dt)
// of nodes/edges to skip during relaxation -- as if they were removed from the
// graph for the duration of this run. pass NULL for either to disable it.
bool DijkstraCtx_Run
(
	DijkstraCtx *dc,             // engine
	NodeID src_id,              // source node
	NodeID dst_id,             // destination, or INVALID_ENTITY_ID for all
	const dict *blocked_nodes,  // nodes to skip, or NULL
	const dict *blocked_edges   // edges to skip, or NULL
);

// bounded single-source run: finalizes every node whose shortest-path distance
// from 'src_id' is <= 'dist_bound' and stops as soon as the next-closest node
// exceeds it (nodes are finalized in nondecreasing distance order). pass
// DBL_MAX for no bound. if 'dst_id' != INVALID_ENTITY_ID, reaching it tightens
// the bound to dst's own distance -- so the run finalizes exactly the ball of
// nodes no farther than the src->dst shortest distance, and the return value
// reports whether dst was reached. this lets the all-shortest-paths DAG explore
// only the shortest-path region instead of the whole graph. results are read
// via DijkstraCtx_Distance. never early-exits at dst.
bool DijkstraCtx_RunBounded
(
	DijkstraCtx *dc,            // engine
	NodeID src_id,             // source node
	NodeID dst_id,             // target whose distance bounds the ball, or
	                           //   INVALID_ENTITY_ID to use dist_bound as-is
	double dist_bound,         // finalize nodes within this distance (DBL_MAX = none)
	const dict *blocked_nodes,  // nodes to skip, or NULL
	const dict *blocked_edges   // edges to skip, or NULL
);

// after a run: report the finalized shortest-path weight to 'v'. returns
// false if 'v' was never discovered by the last run (leaving *weight
// untouched). after a single-source run every reachable node is finalized,
// so this yields exact distances graph-wide.
bool DijkstraCtx_Distance
(
	const DijkstraCtx *dc,  // engine
	NodeID v,               // node to query
	double *weight          // [output] shortest-path weight to 'v'
);

// after a single-pair run that reached 'dst': reconstruct the src -> dst path
// by walking parent pointers. caller owns the returned Path.
Path *DijkstraCtx_Path
(
	const DijkstraCtx *dc,  // engine
	NodeID src_id,          // source node (walk stops here)
	NodeID dst_id           // destination node (walk starts here)
);

// free a Dijkstra engine and all its scratch/iterators.
void DijkstraCtx_Free
(
	DijkstraCtx *dc  // engine
);
