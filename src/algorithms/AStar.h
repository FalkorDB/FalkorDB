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
// ASSUMES weightProp is expressed in the same units as the haversine
// heuristic (meters) for the heuristic to remain admissible (never
// overestimate the true remaining cost to dst) and therefore for A* to
// guarantee an optimal result. If weightProp represents something else
// (time, hop count, an abstract cost), the heuristic may overestimate
// and A* can return a suboptimal (but still structurally valid) path.
// This is a caller responsibility, not enforced at runtime -- mirrors
// how Dijkstra.c documents but does not enforce its non-negative-weight
// precondition.
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
	Graph *g,                  // graph to traverse
	NodeID src_id,             // source node
	NodeID dst_id,             // destination node
	GRAPH_EDGE_DIR dir,        // traverse direction
	RelationID *relationIDs,   // edge type(s) to traverse
	Tensor *relationMatrices,  // relation matrix per relationIDs entry
	int relationCount,         // length of relationIDs
	AttributeID weight_prop,   // weight attribute id
	AttributeID lat_prop,      // latitude attribute id, used for the heuristic
	AttributeID lon_prop       // longitude attribute id, used for the heuristic
);

