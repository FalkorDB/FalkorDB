/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/graph.h"
#include "../datatypes/datatypes.h"

// exact single-shortest-path search via A*: a generalization of Dijkstra
// that steers the search toward dst using an admissible heuristic (the
// haversine great-circle distance between a node's latitude/longitude
// properties and dst's), typically finalizing far fewer nodes than plain
// Dijkstra when the underlying graph is roughly embedded in real-world
// geographic space (e.g. road networks). Worst case (no usable
// coordinates) degrades to Dijkstra with h == 0 everywhere.
//
// ASSUMES weightProp is non-negative for every edge (same as Dijkstra,
// same rationale, not enforced -- see Dijkstra.h).
//
// The haversine heuristic is computed in meters; 'heur_scale' converts it
// into weightProp's units. For A* to remain admissible (never overestimate
// the true remaining cost to dst, and therefore guarantee an optimal result)
// heur_scale MUST be a lower bound on the weight accrued per meter of
// straight-line progress across every edge:
//   weightProp == distance in meters      -> heur_scale == 1
//   weightProp == travel time (e.g hours) -> heur_scale == 1 / max_speed
//                                            (max_speed in meters per hour)
//   weightProp == hop count / abstract    -> heur_scale == 0 (h == 0, i.e.
//                                            plain Dijkstra; any other value
//                                            has no meaningful meter relation)
// A heur_scale larger than that lower bound makes the heuristic inadmissible,
// so A* can return a suboptimal (but still structurally valid) path; a
// heur_scale <= 0 disables the heuristic (plain Dijkstra, always optimal).
// Picking it is a caller responsibility, not enforced at runtime -- mirrors
// how Dijkstra.c documents but does not enforce its non-negative-weight
// precondition. When unsure, underestimate: a smaller heur_scale only costs
// extra exploration, never correctness.
//
// if dst, or any other node discovered during the search, is missing a
// numeric latitudeProperty/longitudeProperty, its heuristic degrades to
// 0 for that node -- still admissible, just locally equivalent to
// Dijkstra rather than an error.
//
// returns true and populates 'path' and 'weight' if 'dst' is reachable
// from 'src'; returns false (leaving them untouched) otherwise.
//
// only 'weight' is computed -- it's the metric being optimized. any
// other per-path attribute (e.g. a "cost" property) isn't part of the
// search and should be summed by the caller over the returned path's
// edges instead.
bool AStar_ShortestPath
(
	Path **path,               // [output] src -> dst path
	double *weight,            // [output] total path weight
	const Graph *g,            // graph to traverse
	NodeID src_id,             // source node
	NodeID dst_id,             // destination node
	GRAPH_EDGE_DIR dir,        // traverse direction
	RelationID *relationIDs,   // edge type(s) to traverse
	Tensor *relationMatrices,  // relation matrix per relationIDs entry
	int relationCount,         // length of relationIDs
	AttributeID weight_prop,   // weight attribute id
	AttributeID lat_prop,      // latitude attribute id, used for the heuristic
	AttributeID lon_prop,      // longitude attribute id, used for the heuristic
	double heur_scale          // meters -> weightProp units heuristic scale
);

// find up to 'k' shortest loopless paths from src to dst by ascending weight,
// via Yen's algorithm using A* (see AStar_ShortestPath) as the single-pair
// spur subroutine. because spur searches are medium-distance point-to-point
// queries, the A* heuristic gives a large speedup over plain-Dijkstra Yen on
// geographically-embedded graphs (road networks).
//
// same weightProp / lat/lon preconditions as AStar_ShortestPath. candidate
// paths are deduplicated by a 64-bit hash of their edge-id sequence.
//
// returns the number of paths found (<= k; 0 if dst is unreachable). '*paths'
// and '*weights' are set to newly allocated parallel array_t buffers (Path*
// and its total weight, ascending); the caller owns both arrays and each Path.
uint AStar_KShortestPaths
(
	Path ***paths,             // [output] array_t of Path*, ascending weight
	double **weights,          // [output] array_t of matching total weights
	const Graph *g,            // graph to traverse
	NodeID src,                // source node
	NodeID dst,                // destination node
	uint64_t k,                // number of paths to find
	GRAPH_EDGE_DIR dir,        // traverse direction
	RelationID *relationIDs,   // edge type(s) to traverse
	Tensor *relationMatrices,  // relation matrix per relationIDs entry
	int relationCount,         // length of relationIDs
	AttributeID weight_prop,   // weight attribute id
	AttributeID lat_prop,      // latitude attribute id, used for the heuristic
	AttributeID lon_prop,      // longitude attribute id, used for the heuristic
	double heur_scale          // meters -> weightProp units heuristic scale
);

