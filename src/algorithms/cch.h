/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "GraphBLAS.h"
#include <stdint.h>

//------------------------------------------------------------------------------
// Customizable Contraction Hierarchies (CCH)
//------------------------------------------------------------------------------
//
// unlike classic contraction hierarchies (see contraction_hierarchies.c),
// which mixes topology and edge weights into one expensive per-node
// witness-search loop, CCH decouples the two into three phases:
//
//   [Phase 1: Preprocessing] -> [Phase 2: Customization] -> [Phase 3: Query]
//     metric-independent          metric-dependent            bidirectional
//     topology only, done         apply edge weights,          upward search
//     once, runs in seconds       runs in milliseconds         microseconds
//
// this file implements Phase 1. Phase 1 has no notion of edge weight or
// direction at all -- it only ever looks at *whether* two nodes are
// connected, never at what an edge costs -- which is exactly what makes it
// reusable across arbitrarily many future Phase 2 customizations (e.g. a
// different weightProp, or live traffic-style weight updates) without
// rerunning it.
//
// Phase 1 has two steps:
//
//   1. elimination order (CCH_EliminationOrder): a nested-dissection
//      ordering computed via METIS_NodeND. Nested dissection recursively
//      splits the graph via small *node separators*: find a small set S
//      that disconnects the graph into two roughly-equal pieces, recurse
//      into each piece, and repeat. Ranks are then assigned so that the
//      deepest, smallest recursive pieces get the *lowest* ranks and each
//      level's separator gets progressively *higher* ranks, with the
//      top-level separator ending up with the highest ranks of all. Since
//      every path between the two pieces a separator splits has to pass
//      through it, this placement is what keeps shortcuts localized within
//      pockets instead of spreading everywhere.
//
//   2. elimination tree + chordal supergraph (CCH_ChordalTriangulation):
//      simulates the elimination game implied by that order. Eliminating
//      a node in rank order and connecting each pair of its not-yet-
//      eliminated neighbors (if they aren't already connected) is exactly
//      one step of symbolic Gaussian elimination; repeating it for every
//      node turns the original graph into its chordal supergraph, the
//      structure that contains every edge -- original or shortcut -- that
//      Phase 2 could ever need to write a weight into, for *any* metric.
//      Every node's parent in the resulting elimination tree T_G is its
//      lowest-ranked neighbor with a higher rank than itself, and the
//      chordal supergraph is stored as what Phase 3's query later walks
//      directly: the "upward graph" (see 'up' below), where each node
//      only lists the neighbors above it in rank.

// a CCH context: accumulates Phase 1's outputs as each step runs. every
// array below is rank-space (indexed 0..n-1 by elimination rank) except
// 'perm'/'iperm', which are the two translations between rank space and
// node-id space.
typedef struct {
	int64_t n ;      // number of nodes

	int64_t *perm  ; // perm[rank]  = node id at elimination rank 'rank'
	int64_t *iperm ; // iperm[node] = elimination rank of 'node' -- this
	                 // is "rank(node)" in the discussion above

	// CSR of the symmetrized, self-loop-free skeleton of the input graph
	// (node-id space, not rank space) -- the plain topology nested
	// dissection partitions in step 1. cached here from
	// CCH_EliminationOrder for reuse by CCH_ChordalTriangulation, which
	// needs the same topology to simulate the elimination game.
	int64_t *xadj   ;    // size n+1
	int64_t *adjncy ;

	// the elimination tree T_G: parent[rank] = the elimination rank of
	// rank's parent (its lowest-ranked neighbor with a higher rank), or -1
	// if 'rank' is a root. a graph need not be connected, so more than one
	// root is possible -- each connected component gets its own.
	int64_t *parent ;

	// the chordal supergraph, stored as the upward graph: up[rank] is an
	// arr_t (util/arr.h) of every rank' > rank that 'rank' is adjacent to
	// once every fill-in edge from the elimination game has been added --
	// i.e. the original edge (if any) plus every shortcut Phase 2 might
	// ever need to weigh in. only the upward direction is kept because
	// that's the only direction Phase 2's sweep and Phase 3's query ever
	// walk. owned dynamic arrays, NULL until populated.
	int64_t **up ;

	//--------------------------------------------------------------------------
	// Phase 2 (customization) outputs -- metric-dependent
	//--------------------------------------------------------------------------

	// per-arc metric weights, laid out parallel to 'up': up_w[rank][i] and
	// dn_w[rank][i] are the two directed weights of the chordal arc between
	// 'rank' and its i-th upper neighbor up[rank][i]. up_w is the upward
	// direction (rank -> up[rank][i]); dn_w is the downward direction
	// (up[rank][i] -> rank). original edges are seeded from the metric matrix
	// W passed to CCH_Customize; shortcut arcs start at +INFINITY and are
	// filled in by triangle relaxation. each is a plain double[] of length
	// arr_len(up[rank]). NULL until CCH_Customize runs; recomputable any number
	// of times for different metrics without rerunning Phase 1.
	double **up_w ;
	double **dn_w ;

	//--------------------------------------------------------------------------
	// Phase 3 (query) scratch -- reused across CCH_Query calls
	//--------------------------------------------------------------------------

	// two size-n distance arrays (rank space), held at +INFINITY between
	// queries. CCH_Query touches only the O(tree-height) ancestors of src/dst
	// and resets exactly those entries afterwards, so the arrays never need an
	// O(n) clear per query. NOT reentrant -- a single CCH must not service two
	// queries concurrently. NULL until the first CCH_Query allocates them.
	double *q_df ;   // forward  distances (from src, upward via up_w)
	double *q_db ;   // backward distances (from dst, upward via dn_w)
} CCH ;

// allocates a CCH context for a graph of 'n' nodes. every field besides
// 'n' is zeroed/NULL until the corresponding Phase 1 step populates it.
CCH *CCH_New
(
	int64_t n
) ;

void CCH_Free
(
	CCH *cch
) ;

// step 1: computes a nested-dissection elimination order for 'A' via
// METIS's METIS_NodeND (see the file-level comment above), populating
// cch->perm/iperm -- and, as a side effect, cch->xadj/adjncy, the
// symmetrized skeleton graph built to feed METIS, cached here for
// CCH_ChordalTriangulation to reuse. only A's topology is used (the
// symmetrized union of A's pattern and its transpose's pattern, diagonal
// dropped); edge weights and direction play no role, hence
// "metric-independent".
void CCH_EliminationOrder
(
	CCH        *cch,
	GrB_Matrix  A   // input graph (only its pattern is used)
) ;

// step 2: simulates the elimination game over cch's order to build the
// elimination tree T_G (cch->parent) and, on top of it, the chordal
// supergraph stored as the upward graph (cch->up) -- see the file-level
// comment above for what both of those mean and why the tree falls out of
// the same computation that builds the upward graph rather than needing
// its own separate pass.
//
// requires CCH_EliminationOrder to have already run (reads cch->perm,
// cch->iperm, cch->xadj, cch->adjncy). purely combinatorial -- doesn't
// touch A or any GraphBLAS object at all, matching Phase 1's
// metric-independence.
void CCH_ChordalTriangulation
(
	CCH *cch
) ;

// Phase 2 (customization): writes a concrete metric into the chordal
// supergraph, populating cch->up_w / cch->dn_w. 'W' is a GrB_FP64 matrix in
// node-id space where W[u][v] is the weight of the directed edge u -> v (the
// caller collapses parallel edges to their minimum). Every chordal arc is
// first seeded from W (original edges get their weight, shortcuts +INFINITY),
// then "basic customization" relaxes every lower triangle in increasing rank
// order so each arc ends up holding the shortest path between its endpoints
// that only routes through strictly-lower-ranked nodes.
//
// requires Phase 1 to have run (reads cch->perm/iperm/up). purely a function
// of W's values -- rerunnable for a different metric (a new weightProp, live
// traffic updates, ...) without touching Phase 1's topology-only output. safe
// to call repeatedly; frees any previous up_w/dn_w first.
void CCH_Customize
(
	CCH        *cch,
	GrB_Matrix  W    // node-id space, W[u][v] = weight of edge u -> v
) ;

// Phase 3 (query): exact point-to-point shortest distance from 'src' to 'dst'
// (both node ids). Runs the heap-free elimination-tree search -- walk src's
// and dst's ancestor paths in T_G, relaxing up-arcs (forward, via up_w) and
// down-arcs (backward, via dn_w), then combine at the shared ancestors. O(tree
// height x up-degree), independent of overall graph size.
//
// requires Phase 2 to have run. returns true and sets *weight if 'dst' is
// reachable from 'src'; returns false (leaving *weight untouched) otherwise.
// NOT reentrant (uses cch->q_df / cch->q_db scratch).
bool CCH_Query
(
	CCH     *cch,
	int64_t  src,     // source node id
	int64_t  dst,     // destination node id
	double  *weight   // [output] shortest src -> dst distance
) ;

