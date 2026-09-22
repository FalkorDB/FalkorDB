/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "GraphBLAS.h"
#include <stddef.h>
#include <stdint.h>

// forward declaration of the serializer stream (see serializers/serializer_io.h)
// -- redefining the identical typedef is allowed under C11, so this stays in
// sync without pulling the serializer header into the algorithms layer
typedef struct SerializerIO_Opaque *SerializerIO;

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
	// PHASE-1 SCRATCH: freed (-> NULL) at the end of CCH_ChordalTriangulation --
	// nothing past the chordal build (Phase 2 / query / maintenance) reads it,
	// and a rebuild reconstructs it from the graph.
	int64_t *xadj   ;    // size n+1
	int64_t *adjncy ;

	// the elimination tree T_G: parent[rank] = the elimination rank of
	// rank's parent (its lowest-ranked neighbor with a higher rank), or -1
	// if 'rank' is a root. a graph need not be connected, so more than one
	// root is possible -- each connected component gets its own.
	// PHASE-1 SCRATCH, like xadj/adjncy: freed (-> NULL) at the end of
	// CCH_ChordalTriangulation. (also recoverable as up[rank][0] if ever needed.)
	int64_t *parent ;

	// the chordal supergraph, stored as the upward graph: up[rank] is an
	// arr_t (util/arr.h) of every rank' > rank that 'rank' is adjacent to
	// once every fill-in edge from the elimination game has been added --
	// i.e. the original edge (if any) plus every shortcut Phase 2 might
	// ever need to weigh in. only the upward direction is kept because
	// that's the only direction Phase 2's sweep and Phase 3's query ever
	// walk. owned dynamic arrays, NULL until populated.
	int64_t **up ;

	// the downward adjacency: down[rank] is an arr_t of every rank' < rank that
	// 'rank' is adjacent to in the chordal supergraph -- the exact inverse of
	// 'up' (rank' in down[rank] iff rank in up[rank']), kept sorted ascending.
	// metric-independent (topology only), built alongside 'up' by
	// CCH_ChordalTriangulation. only an incremental re-customization needs it --
	// to enumerate an arc's lower common neighbours when recomputing it in
	// isolation -- so the full-sweep Customize ignores it. NULL until built.
	int64_t **down ;

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

	// per-arc "middle" node, laid out parallel to 'up' -- the elimination rank
	// of the lower-ranked apex through which the customized weight was achieved
	// (up_w[a][i] came from the triangle a -> up_mid[a][i] -> up[a][i]), or -1
	// when the arc's weight is just its original road edge (no detour, nothing
	// to unpack). this is what lets a shortcut be recursively expanded back into
	// the original road edges it stands for. NULL until CCH_Customize runs.
	int64_t **up_mid ;
	int64_t **dn_mid ;
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

// total heap bytes held by the resident hierarchy: the struct, the rank-space
// arrays (perm/iperm/xadj/adjncy/parent), the upward + downward adjacency, and
// the per-arc weight/middle arrays. returns 0 for a NULL hierarchy.
size_t CCH_MemoryUsage
(
	const CCH *cch  // hierarchy to measure
) ;

// serialize the resident hierarchy to 'io' (RDB): 'n', 'perm', and each rank's
// upward adjacency + per-arc weights/middles. the derived structures (iperm,
// down) and Phase-1 scratch (xadj/adjncy/parent) are NOT written -- they are
// rebuilt / stay NULL on load.
void CCH_RdbSave
(
	const CCH   *cch,  // hierarchy to serialize
	SerializerIO io    // stream to write to
) ;

// reconstruct a hierarchy from 'io' (inverse of CCH_RdbSave): reads n/perm and
// the per-rank arrays, then derives iperm + down. no rebuild, no METIS. caller
// owns the returned CCH.
CCH *CCH_RdbLoad
(
	SerializerIO io  // stream to read from
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
	CCH             *cch,
	const GrB_Matrix W    // node-id space, W[u][v] = weight of edge u -> v
) ;

// seed callback for CCH_RecustomizeScoped: for the directed node pair (u, v)
// write the current metric weight of the cheapest u -> v original edge to
// '*w_uv' (or +INFINITY if there is none) and the v -> u weight to '*w_vu'.
// must reproduce exactly the seeding CCH_Customize does from its weight matrix W
// (min over parallel edges / relationship types), so a scoped recustomization
// lands on the same values a full one would.
typedef void (*CCH_SeedFn)
(
	void   *ctx,     // opaque caller context
	int64_t u,       // node id (rank space perm[] value)
	int64_t v,       // node id
	double *w_uv,    // [out] cheapest u -> v weight, or +INFINITY
	double *w_vu     // [out] cheapest v -> u weight, or +INFINITY
) ;

// incrementally re-customize after the original arcs listed in 'du'[i] -> 'dv'[i]
// (node-id space, 'k' of them) changed weight -- WITHOUT rerunning the full
// Phase-2 sweep. each affected chordal arc is reset to its seed (via 'seed') and
// re-relaxed over its lower common neighbours; the change is propagated up the
// elimination structure, processing arcs in increasing lower-endpoint rank so an
// arc is only recomputed once every arc it depends on is final. this handles
// weight increases and decreases uniformly (a full recompute per arc, not a
// min-relax) and lands on exactly the values CCH_Customize would, but touches
// only the affected cone. requires Phase 1 (including cch->down) to have run and
// a prior customization to be present. topology / elimination order unchanged.
void CCH_RecustomizeScoped
(
	CCH          *cch,   // hierarchy to update in place
	CCH_SeedFn    seed,  // per-arc seed weight lookup
	void         *ctx,   // opaque context passed to 'seed'
	const int64_t *du,   // changed arc source node ids
	const int64_t *dv,   // changed arc destination node ids
	uint64_t      k      // number of changed arcs
) ;

// true if a chordal arc connects ranks 'a' and 'b' (either direction). used to
// tell an edge whose arc already exists (a weight-only change, handled by a
// scoped recustomization) from one that introduces a brand-new adjacency (which
// needs the chordal structure itself to grow -- a full rebuild). requires Phase 1.
bool CCH_HasArc
(
	const CCH *cch,  // hierarchy
	int64_t    a,    // rank
	int64_t    b     // rank
) ;

// build the improving-shortcut matrix for materialization into the graph.
// for every chordal arc, emit its customized weight into 'S' (a fresh GrB_FP64
// matrix in node-id space, S[u][v] = customized weight of the directed arc
// u -> v) IFF that weight beats the corresponding road-edge weight in 'W' (or
// W has no entry there) -- i.e. only the arcs that carry information beyond the
// original edges, exactly classic CH's shortcut set. everywhere else the
// original road edge already carries the correct weight, so no shortcut is
// emitted. requires Phase 2 (CCH_Customize) to have run against this same 'W'.
// caller owns and frees '*S' and '*M'. 'M' (node-id space GrB_INT64) is emitted
// in lockstep with 'S': M[u][v] is the node id of the middle vertex the shortcut
// u -> v routes through, i.e. the arc expands to u -> M[u][v] -> v, each half of
// which is itself a road or shortcut edge to be unpacked recursively. every
// emitted shortcut has a middle (a shortcut only exists because a detour beat
// the road edge), so M has an entry for exactly the same (u,v) pairs as S.
void CCH_ExtractShortcuts
(
	const CCH       *cch,
	const GrB_Matrix W,   // road weight matrix (same one Customize ran with)
	GrB_Matrix      *S,   // [output] improving shortcut weights, node-id space FP64
	GrB_Matrix      *M    // [output] shortcut middle node ids, node-id space INT64
) ;

// Phase 3 (query) is not part of this API: the materialized SHORTCUT edges +
// node ranks are queried by the stateless, concurrency-safe rank-pruned
// bidirectional Dijkstra in proc_cch_query.c, so many queries can run against
// the same graph in parallel without any shared CCH scratch.

