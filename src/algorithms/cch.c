/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "metis.h"
#include "../util/arr.h"
#include "../util/dict.h"
#include "../util/rmalloc.h"
#include "../serializers/serializer_io.h"
#include "utils/priority_heap.h"
#include "cch.h"

#include <math.h>
#include <string.h>

// METIS_NodeND requires idx_t and int64_t to line up (see build.sh's
// build_metis(), which builds METIS with i64=1 specifically to match
// FalkorDB's 64-bit node ids) -- catch a mismatched METIS build at compile
// time rather than silently truncating node ids at runtime.
_Static_assert (sizeof (idx_t) == sizeof (int64_t),
		"METIS must be built with i64=1 (64-bit idx_t) to match NodeID's width") ;

CCH *CCH_New
(
	int64_t n
)
{
	CCH *cch = rm_calloc (1, sizeof (CCH)) ;
	cch->n = n ;
	return cch ;
}

void CCH_Free
(
	CCH *cch
)
{
	if (cch == NULL) {
		return ;
	}

	if (cch->xadj   != NULL) rm_free (cch->xadj)   ;
	if (cch->perm   != NULL) rm_free (cch->perm)   ;
	if (cch->iperm  != NULL) rm_free (cch->iperm)  ;
	if (cch->adjncy != NULL) rm_free (cch->adjncy) ;
	if (cch->parent != NULL) rm_free (cch->parent) ;

	if (cch->up != NULL) {
		for (int64_t rank = 0 ; rank < cch->n ; rank++) {
			arr_free (cch->up [rank]) ;
		}
		rm_free (cch->up) ;
	}

	if (cch->down != NULL) {
		for (int64_t rank = 0 ; rank < cch->n ; rank++) {
			arr_free (cch->down [rank]) ;
		}
		rm_free (cch->down) ;
	}

	if (cch->up_w != NULL) {
		for (int64_t rank = 0 ; rank < cch->n ; rank++) {
			rm_free (cch->up_w [rank]) ;
		}
		rm_free (cch->up_w) ;
	}

	if (cch->dn_w != NULL) {
		for (int64_t rank = 0 ; rank < cch->n ; rank++) {
			rm_free (cch->dn_w [rank]) ;
		}
		rm_free (cch->dn_w) ;
	}

	if (cch->up_mid != NULL) {
		for (int64_t rank = 0 ; rank < cch->n ; rank++) {
			rm_free (cch->up_mid [rank]) ;
		}
		rm_free (cch->up_mid) ;
	}

	if (cch->dn_mid != NULL) {
		for (int64_t rank = 0 ; rank < cch->n ; rank++) {
			rm_free (cch->dn_mid [rank]) ;
		}
		rm_free (cch->dn_mid) ;
	}

	rm_free (cch) ;
}

size_t CCH_MemoryUsage
(
	const CCH *cch
) {
	if (cch == NULL) {
		return 0 ;
	}

	int64_t n  = cch->n ;
	size_t  sz = RedisModule_MallocSize ((void *) cch) ;

	// rank-space fixed arrays
	if (cch->perm   != NULL) sz += RedisModule_MallocSize (cch->perm)   ;
	if (cch->iperm  != NULL) sz += RedisModule_MallocSize (cch->iperm)  ;
	if (cch->xadj   != NULL) sz += RedisModule_MallocSize (cch->xadj)   ;
	if (cch->adjncy != NULL) sz += RedisModule_MallocSize (cch->adjncy) ;
	if (cch->parent != NULL) sz += RedisModule_MallocSize (cch->parent) ;

	// upward / downward adjacency: a pointer array plus one arr_ per rank
	if (cch->up != NULL) {
		sz += RedisModule_MallocSize (cch->up) ;
		for (int64_t r = 0 ; r < n ; r++) {
			if (cch->up [r] != NULL) sz += arr_bytesize (cch->up [r]) ;
		}
	}
	if (cch->down != NULL) {
		sz += RedisModule_MallocSize (cch->down) ;
		for (int64_t r = 0 ; r < n ; r++) {
			if (cch->down [r] != NULL) sz += arr_bytesize (cch->down [r]) ;
		}
	}

	// per-arc metric outputs: a pointer array plus one plain array per rank
	#define _CCH_ARC_ARR(field)                                          \
		if (cch->field != NULL) {                                        \
			sz += RedisModule_MallocSize (cch->field) ;                  \
			for (int64_t r = 0 ; r < n ; r++) {                          \
				if (cch->field [r] != NULL)                              \
					sz += RedisModule_MallocSize (cch->field [r]) ;      \
			}                                                            \
		}
	_CCH_ARC_ARR (up_w)   ;
	_CCH_ARC_ARR (dn_w)   ;
	_CCH_ARC_ARR (up_mid) ;
	_CCH_ARC_ARR (dn_mid) ;
	#undef _CCH_ARC_ARR

	return sz ;
}

// builds the CSR (xadj/adjncy) representation METIS_NodeND expects: the
// simple, undirected, self-loop-free skeleton of 'A' -- i.e. the symmetrized
// union of A's pattern and its transpose's pattern. edge weights and
// direction have no bearing on a fill-reducing order, so they're dropped
// here entirely.
//
// writes into 'cch->xadj' (size n+1) and 'cch->adjncy' (size nnz) -- both
// stay cached there for reuse by CCH_ChordalTriangulation.
static void _build_undirected_skeleton
(
	CCH        *cch,
	GrB_Matrix  A
) {
	ASSERT (A   != NULL) ;
	ASSERT (cch != NULL) ;

	int64_t n = cch->n ;

	GrB_Matrix Sym = NULL ;
	GrB_OK (GrB_Matrix_new (&Sym, GrB_BOOL, n, n)) ;

	// Sym = pattern(A) | pattern(AT) -- ONEB ignores both operands' actual
	// values, so every surviving entry (from either side) collapses to a
	// plain structural 'true'
	GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp (Sym, NULL, NULL, GrB_ONEB_BOOL,
				A, A, GrB_DESC_T1)) ;

	// drop the diagonal -- METIS_NodeND rejects self-loops
	GrB_OK (GrB_select (Sym, NULL, NULL, GrB_OFFDIAG, Sym, false, NULL)) ;

	GrB_Index nvals ;
	GrB_OK (GrB_Matrix_nvals (&nvals, Sym)) ;

	int64_t *xadj   = rm_malloc (sizeof (int64_t) * (n + 1)) ;
	int64_t *adjncy = rm_malloc (sizeof (int64_t) * nvals) ;

	GxB_Iterator it ;
	GrB_OK (GxB_Iterator_new (&it)) ;
	GrB_OK (GxB_rowIterator_attach (it, Sym, NULL)) ;

	// walk every row 0..n-1 in order, filling xadj even for rows the
	// iterator skips over (empty rows aren't materialized at all when Sym
	// happens to be stored hypersparse, so 'nextRow' can jump straight past
	// them to the next nonempty row, or to GxB_EXHAUSTED)
	int64_t pos      =  0 ;
	int64_t last_row = -1 ;

	GrB_Info info = GxB_rowIterator_seekRow (it, 0) ;
	while (info != GxB_EXHAUSTED) {
		int64_t i = (int64_t) GxB_rowIterator_getRowIndex (it) ;

		for (int64_t r = last_row + 1 ; r <= i ; r++) {
			xadj [r] = pos ;
		}
		last_row = i ;

		while (info == GrB_SUCCESS) {
			adjncy [pos++] = (int64_t) GxB_rowIterator_getColIndex (it) ;
			info = GxB_rowIterator_nextCol (it) ;
		}

		info = GxB_rowIterator_nextRow (it) ;
	}

	// fill xadj for any empty trailing rows, plus the final xadj[n]
	// sentinel
	for (int64_t r = last_row + 1 ; r <= n ; r++) {
		xadj [r] = pos ;
	}

	GrB_OK (GrB_free (&it))  ;
	GrB_OK (GrB_free (&Sym)) ;

	cch->xadj   = xadj   ;
	cch->adjncy = adjncy ;
}

// step 1: computes a nested-dissection elimination order for 'A' via
// METIS's METIS_NodeND, populating cch->perm/iperm -- and, as a side effect
// cch->xadj/adjncy, the symmetrized skeleton graph built to feed METIS, cached
// here for CCH_ChordalTriangulation to reuse. only A's topology is used
// (the symmetrized union of A's pattern and its transpose's pattern, diagonal
// dropped); edge weights and direction play no role, hence "metric-independent"
void CCH_EliminationOrder
(
	CCH        *cch,
	GrB_Matrix  A   // input graph (only its pattern is used)
)
{
	ASSERT (A   != NULL) ;
	ASSERT (cch != NULL) ;

	GrB_Index nrows ;
	GrB_Index ncols ;

	GrB_OK (GrB_Matrix_nrows (&nrows, A)) ;
	GrB_OK (GrB_Matrix_ncols (&ncols, A)) ;

	// A should be squared
	ASSERT (nrows == ncols) ;
	ASSERT ((int64_t) nrows == cch->n) ;  // node count must match A's dim

	//--------------------------------------------------------------------------
	// build METIS_NodeND input (A+AT - main diagonal)
	//--------------------------------------------------------------------------

	_build_undirected_skeleton (cch, A) ;

	idx_t nvtxs = (idx_t) cch->n ;

	// allocate METIS_NodeND outputs

	size_t s = sizeof (int64_t) * cch->n ;
	int64_t *perm  = rm_malloc (s) ;  // perm  [rank]    = node id
	int64_t *iperm = rm_malloc (s) ;  // iperm [node_id] = rank

	// default options
	idx_t options [METIS_NOPTIONS] = {0} ;
	METIS_SetDefaultOptions (options) ;

	//--------------------------------------------------------------------------
	// run METIS nested dissection
	//--------------------------------------------------------------------------

	int status = METIS_NodeND (&nvtxs, (idx_t *) cch->xadj,
			(idx_t *) cch->adjncy, NULL, options, (idx_t *) perm,
			(idx_t *) iperm) ;
	ASSERT (status == METIS_OK) ;

	// save node rankings
	cch->perm  = perm  ;
	cch->iperm = iperm ;
}

// Union-Find over elimination ranks, used only by _build_elimination_tree
// below. 'ancestor[x]', for a rank 'x' already visited at least once, is
// the highest rank known SO FAR to be connected to 'x' once fill-in is
// accounted for -- i.e. x's current representative -- or -1 if 'x' has
// never been reached yet.
//
// Find(w_rank, rank): starting from 'w_rank', walks the ancestor chain
// while it stays below 'rank', compressing every rank visited to point
// straight at the walk's endpoint (so a later Find through any of them is
// O(1) instead of re-walking the chain). if that endpoint is unclaimed
// (ancestor == -1), it is immediately claimed -- Union'd -- under 'rank',
// which also records it as that rank's parent in the elimination tree T_G:
// the walk necessarily reaches T_G's smallest still-unclaimed rank
// connected to w_rank, and 'rank' is, by construction, the first (hence
// lowest-ranked) thing ever found trying to claim it.
//
// no union-by-rank/size is needed to get the near-linear bound here (unlike
// textbook union-find) -- ranks are always processed in increasing order,
// so every union already attaches a lower rank under a strictly higher
// one, and path compression alone is enough to make the total cost across
// all n Find calls O(nnz(skeleton) * alpha(n)), i.e. effectively linear.
static void _find_and_union
(
	int64_t *ancestor,
	int64_t *parent,
	int64_t  w_rank,   // rank to start the walk from
	int64_t  rank      // rank performing the Find -- becomes the parent of
	                    // whichever still-unclaimed rank the walk reaches
) {
	while (w_rank != -1 && w_rank < rank) {
		int64_t next = ancestor [w_rank] ;
		ancestor [w_rank] = rank ;         // path compression
		if (next == -1) {
			parent [w_rank] = rank ;       // unclaimed -- Union under 'rank'
		}
		w_rank = next ;
	}
}

// step 2, part 1: builds the elimination tree T_G via the Union-Find above.
// T_G's defining property: parent(u) is u's lowest-ranked neighbor with a
// higher rank than u -- but crucially, that means u's neighbor in the
// *chordal* supergraph (original edges plus every fill-in edge the
// elimination game introduces), not just u's original neighbors. see
// CCH_ChordalTriangulation's header comment for a worked example of why
// those two can differ.
//
// this gets parent() right WITHOUT ever materializing a single fill-in
// edge: process ranks in increasing order, and for each rank's node u, run
// Find/Union starting from every one of u's *original* neighbors that's
// already eliminated (smaller rank). that's the whole trick -- fill-in
// relationships never need to be looked up explicitly, because by the time
// a later, higher rank asks Find(w_rank) for some early w_rank, path
// compression has already threaded w_rank's chain through every rank that
// previously tried (and failed) to claim it, so the walk lands exactly
// where the *chordal* graph (not just the original one) says it should.
static void _build_elimination_tree
(
	CCH *cch
) {
	int64_t n = cch->n ;

	int64_t *parent   = rm_malloc (sizeof (int64_t) * n) ;
	int64_t *ancestor = rm_malloc (sizeof (int64_t) * n) ;

	for (int64_t rank = 0 ; rank < n ; rank++) {
		parent   [rank] = -1 ;
		ancestor [rank] = -1 ;

		int64_t u = cch->perm [rank] ;   // the node eliminated at this rank

		// each original edge is visited from both endpoints over the run,
		// so only chasing already-eliminated neighbors (smaller rank) also
		// avoids ever walking the same pair twice
		for (int64_t p = cch->xadj [u] ; p < cch->xadj [u + 1] ; p++) {
			int64_t w      = cch->adjncy [p] ;    // a neighbor of u
			int64_t w_rank = cch->iperm [w] ;     // w's elimination rank

			_find_and_union (ancestor, parent, w_rank, rank) ;
		}
	}

	rm_free (ancestor) ;
	cch->parent = parent ;
}

// ascending int64 comparator for qsort'ing each rank's up-list
static int _cmp_int64
(
	const void *a,
	const void *b
) {
	int64_t x = *(const int64_t *) a ;
	int64_t y = *(const int64_t *) b ;
	return (x > y) - (x < y) ;
}

// step 2, part 2: builds the chordal supergraph on top of the elimination
// tree T_G just computed, stored as the upward graph (cch->up -- see
// cch.h). uses the Gilbert-Ng-Peyton "child absorption" formula:
//
//   up(rank) = { w in original graph : rank(w) > rank }
//              U  U{ up(child) \ {rank} : child is rank's T_G child }
//
// why this is exactly right, not just a shortcut: when a child is
// eliminated, the elimination game says all of its still-remaining (i.e.
// higher-rank) neighbors -- up(child) -- become mutually adjacent (a
// clique). 'rank' is by construction the smallest-ranked member of that
// clique (that's the definition of parent() from _build_elimination_tree
// above), so 'rank' ends up adjacent to every other member too. those
// other members are exactly up(child) minus 'rank' itself. no pairwise
// clique ever needs to be enumerated explicitly -- each rank just inherits
// its children's sets.
//
// processing ranks in increasing order is automatically a bottom-up
// (children-before-parents) sweep, since parent[child] > child always.
// total cost is O(size of the chordal supergraph) -- no wasted work
// building pairs a later absorption would just discard.
//
// worked example: a 4-node star X-A, X-B, X-C, eliminated in the (bad --
// hub-first) order rank(X)=0, rank(A)=1, rank(B)=2, rank(C)=3:
//   up(X) = {A,B,C}          (its 3 original edges)
//   up(A) = {B,C}            (no original up-neighbor; absorbs up(X)\{A})
//   up(B) = {C}              (absorbs up(A)\{B})
//   up(C) = {}                (root; absorbs up(B)\{C} = {})
// i.e. eliminating hub X first forces A, B and C into a fill-in triangle
// -- exactly the pathological case nested dissection avoids by never
// eliminating high-degree nodes early.
static void _build_chordal_supergraph
(
	CCH *cch
) {
	int64_t n = cch->n ;

	// children[rank] = arr_t of rank's T_G children (every 'c' with
	// parent[c] == rank)
	int64_t **children = rm_calloc (n, sizeof (int64_t *)) ;
	for (int64_t c = 0 ; c < n ; c++) {
		int64_t p = cch->parent [c] ;
		if (p == -1) {
			continue ;
		}
		if (children [p] == NULL) {
			children [p] = arr_new (int64_t, 2) ;
		}
		arr_append (children [p], c) ;
	}

	int64_t **up = rm_calloc (n, sizeof (int64_t *)) ;

	// mark[w] == stamp means "w already added to up[rank]" for the rank
	// currently being processed -- a monotonically increasing per-rank
	// stamp (rank+1, never 0) stands in for an O(n) reset of 'mark' before
	// every rank
	int64_t *mark = rm_calloc (n, sizeof (int64_t)) ;

	for (int64_t rank = 0 ; rank < n ; rank++) {
		up [rank] = arr_new (int64_t, 4) ;
		int64_t stamp = rank + 1 ;

		// 1. rank's own up-neighbors in the original (symmetrized) graph
		int64_t u = cch->perm [rank] ;
		for (int64_t p = cch->xadj [u] ; p < cch->xadj [u + 1] ; p++) {
			int64_t w_rank = cch->iperm [cch->adjncy [p]] ;
			if (w_rank > rank && mark [w_rank] != stamp) {
				mark [w_rank] = stamp ;
				arr_append (up [rank], w_rank) ;
			}
		}

		// 2. absorb every child's up-set (minus 'rank' itself -- that
		// edge is already implied by the T_G parent/child link, not a
		// separate up-neighbor to record)
		int64_t *children_of_rank = children [rank] ;
		for (int64_t ci = 0 ; ci < arr_len (children_of_rank) ; ci++) {
			int64_t child    = children_of_rank [ci] ;
			int64_t *up_child = up [child] ;

			for (int64_t ui = 0 ; ui < arr_len (up_child) ; ui++) {
				int64_t w_rank = up_child [ui] ;
				if (w_rank != rank && mark [w_rank] != stamp) {
					mark [w_rank] = stamp ;
					arr_append (up [rank], w_rank) ;
				}
			}
			// NOTE: up[child] is deliberately NOT freed here. Unlike a plain
			// elimination-tree/fill computation (which could drop each set the
			// moment its parent absorbs it), CCH keeps every rank's upward
			// adjacency alive for good -- it *is* the chordal supergraph that
			// Phase 2 customizes and Phase 3 queries. child sets are read (not
			// consumed) by the absorption above.
		}

		arr_free (children [rank]) ;
	}

	rm_free (mark) ;
	rm_free (children) ;

	// sort each rank's up-list ascending so Phase 2 can binary-search the
	// arc (y,z) it needs to relax, and Phase 3 walks neighbors in rank order
	for (int64_t rank = 0 ; rank < n ; rank++) {
		qsort (up [rank], arr_len (up [rank]), sizeof (int64_t), _cmp_int64) ;
	}

	cch->up = up ;
}

// build cch->down, the inverse of cch->up: down[hi] lists every lo < hi with
// hi in up[lo]. iterating lo in increasing rank and appending lo to down[hi]
// leaves each down list sorted ascending (mirroring up's sorted order), which an
// incremental recustomization relies on to intersect lower neighbourhoods.
static void _build_down_adjacency
(
	CCH *cch
) {
	int64_t n = cch->n ;

	int64_t **down = rm_calloc (n, sizeof (int64_t *)) ;
	for (int64_t rank = 0 ; rank < n ; rank++) {
		down [rank] = arr_new (int64_t, 2) ;
	}

	for (int64_t lo = 0 ; lo < n ; lo++) {
		int64_t *ulo = cch->up [lo] ;
		for (uint32_t i = 0 ; i < arr_len (ulo) ; i++) {
			int64_t hi = ulo [i] ;          // hi > lo
			arr_append (down [hi], lo) ;    // lo is a lower neighbour of hi
		}
	}

	cch->down = down ;
}

void CCH_ChordalTriangulation
(
	CCH *cch
) {
	ASSERT (cch          != NULL) ;
	ASSERT (cch->perm    != NULL) ;
	ASSERT (cch->iperm   != NULL) ;
	ASSERT (cch->xadj    != NULL) ;
	ASSERT (cch->adjncy  != NULL) ;

	_build_elimination_tree   (cch) ;
	_build_chordal_supergraph (cch) ;
	_build_down_adjacency     (cch) ;

	// xadj / adjncy / parent are Phase-1 scratch: the raw skeleton fed to METIS
	// (xadj/adjncy) and the elimination tree (parent). they are read only while
	// building the chordal supergraph, above -- never by Phase 2, the query, or
	// incremental maintenance (a topology change rebuilds them from the graph
	// anyway). free them now to keep the resident hierarchy lean. NULL so
	// CCH_Free doesn't double-free and CCH_MemoryUsage counts them as 0.
	rm_free (cch->xadj)   ; cch->xadj   = NULL ;
	rm_free (cch->adjncy) ; cch->adjncy = NULL ;
	rm_free (cch->parent) ; cch->parent = NULL ;
}

//------------------------------------------------------------------------------
// Phase 2: customization
//------------------------------------------------------------------------------

// binary-searches 'z' among rank 'y's upper neighbors (up[y], kept sorted
// ascending by _build_chordal_supergraph) and returns its index. the chordal
// property guarantees this arc exists whenever y and z share a lower common
// neighbor -- which is exactly when the triangle relaxation looks it up -- so
// a miss is a logic error, not an expected outcome.
static int64_t _find_upper
(
	const CCH *cch,
	int64_t    y,
	int64_t    z
) {
	int64_t *uy = cch->up [y] ;
	int64_t  lo = 0 ;
	int64_t  hi = (int64_t) arr_len (uy) - 1 ;

	while (lo <= hi) {
		int64_t mid = lo + ((hi - lo) >> 1) ;
		if      (uy [mid] < z) lo = mid + 1 ;
		else if (uy [mid] > z) hi = mid - 1 ;
		else                   return mid ;
	}

	ASSERT (false && "arc (y,z) missing from chordal supergraph") ;
	return -1 ;
}

void CCH_Customize
(
	CCH             *cch,
	const GrB_Matrix W
) {
	ASSERT (cch     != NULL) ;
	ASSERT (cch->up != NULL) ;   // Phase 1 must have run
	ASSERT (W       != NULL) ;

	int64_t n = cch->n ;

	// a repeat Customize (different metric) reuses Phase 1's topology but
	// discards the previous metric's arc weights
	if (cch->up_w != NULL) {
		for (int64_t r = 0 ; r < n ; r++) rm_free (cch->up_w [r]) ;
		rm_free (cch->up_w) ;
	}
	if (cch->dn_w != NULL) {
		for (int64_t r = 0 ; r < n ; r++) rm_free (cch->dn_w [r]) ;
		rm_free (cch->dn_w) ;
	}
	if (cch->up_mid != NULL) {
		for (int64_t r = 0 ; r < n ; r++) rm_free (cch->up_mid [r]) ;
		rm_free (cch->up_mid) ;
	}
	if (cch->dn_mid != NULL) {
		for (int64_t r = 0 ; r < n ; r++) rm_free (cch->dn_mid [r]) ;
		rm_free (cch->dn_mid) ;
	}

	double **up_w   = rm_malloc (sizeof (double *)  * n) ;
	double **dn_w   = rm_malloc (sizeof (double *)  * n) ;
	int64_t **up_mid = rm_malloc (sizeof (int64_t *) * n) ;
	int64_t **dn_mid = rm_malloc (sizeof (int64_t *) * n) ;

	//--------------------------------------------------------------------------
	// seed: original edges take their weight from W, shortcut arcs +INFINITY
	//--------------------------------------------------------------------------

	for (int64_t rank = 0 ; rank < n ; rank++) {
		int64_t u   = cch->perm [rank] ;                 // node id at this rank
		int64_t deg = (int64_t) arr_len (cch->up [rank]) ;

		// deg can be 0 (a root with no upper neighbors); keep a 1-slot alloc
		// so the pointer is always non-NULL and freeable
		int64_t slots = (deg > 0 ? deg : 1) ;
		up_w   [rank] = rm_malloc (sizeof (double)  * slots) ;
		dn_w   [rank] = rm_malloc (sizeof (double)  * slots) ;
		up_mid [rank] = rm_malloc (sizeof (int64_t) * slots) ;
		dn_mid [rank] = rm_malloc (sizeof (int64_t) * slots) ;

		for (int64_t i = 0 ; i < deg ; i++) {
			int64_t v = cch->perm [cch->up [rank] [i]] ; // upper neighbor node id
			double  w ;

			// up direction u -> v
			up_w [rank] [i] =
				(GrB_Matrix_extractElement_FP64 (&w, W, u, v) == GrB_SUCCESS)
				? w : INFINITY ;

			// down direction v -> u
			dn_w [rank] [i] =
				(GrB_Matrix_extractElement_FP64 (&w, W, v, u) == GrB_SUCCESS)
				? w : INFINITY ;

			// -1 => arc is (so far) just its road edge, nothing to unpack
			up_mid [rank] [i] = -1 ;
			dn_mid [rank] [i] = -1 ;
		}
	}

	//--------------------------------------------------------------------------
	// basic customization: relax every lower triangle in increasing rank order
	//--------------------------------------------------------------------------
	//
	// for each vertex x (the lowest of the triangle), every pair {y,z} of its
	// upper neighbors forms a triangle x < y < z whose top arc (y,z) can be
	// improved by routing through x. processing x in increasing rank guarantees
	// the two lower arcs (x,y) and (x,z) are already final when read: any
	// update to them comes from a triangle with an even-lower apex w < x, all
	// of which were handled in earlier iterations. so a single forward sweep
	// suffices -- no iteration to convergence.
	for (int64_t x = 0 ; x < n ; x++) {
		int64_t *ux = cch->up [x] ;
		int64_t  dx = (int64_t) arr_len (ux) ;

		for (int64_t i = 0 ; i < dx ; i++) {
			int64_t y      = ux [i] ;       // lower endpoint of the top arc
			double  wxy_up = up_w [x] [i] ; // x -> y
			double  wxy_dn = dn_w [x] [i] ; // y -> x

			for (int64_t j = 0 ; j < dx ; j++) {
				int64_t z = ux [j] ;
				if (z <= y) continue ;      // enumerate each {y,z} pair once, y<z

				double wxz_up = up_w [x] [j] ; // x -> z
				double wxz_dn = dn_w [x] [j] ; // z -> x

				int64_t k = _find_upper (cch, y, z) ; // slot of arc (y,z) in up[y]

				// improve up(y,z) via the detour y -> x -> z; remember x as the
				// arc's middle so it can later be unpacked into y -> x -> z
				double cand_up = wxy_dn + wxz_up ;
				if (cand_up < up_w [y] [k]) {
					up_w   [y] [k] = cand_up ;
					up_mid [y] [k] = x ;
				}

				// improve dn(y,z) via the detour z -> x -> y
				double cand_dn = wxz_dn + wxy_up ;
				if (cand_dn < dn_w [y] [k]) {
					dn_w   [y] [k] = cand_dn ;
					dn_mid [y] [k] = x ;
				}
			}
		}
	}

	cch->up_w   = up_w ;
	cch->dn_w   = dn_w ;
	cch->up_mid = up_mid ;
	cch->dn_mid = dn_mid ;
}

//------------------------------------------------------------------------------
// Phase 2 (scoped): incremental re-customization after a weight change
//------------------------------------------------------------------------------

// like _find_upper but returns -1 instead of asserting when the arc (y,z) is
// absent -- used where "not adjacent" is a legitimate answer (probing whether a
// lower neighbour of one endpoint is also adjacent to the other).
static int64_t _find_upper_opt
(
	const CCH *cch,
	int64_t    y,
	int64_t    z
) {
	int64_t *uy = cch->up [y] ;
	int64_t  lo = 0 ;
	int64_t  hi = (int64_t) arr_len (uy) - 1 ;

	while (lo <= hi) {
		int64_t mid = lo + ((hi - lo) >> 1) ;
		if      (uy [mid] < z) lo = mid + 1 ;
		else if (uy [mid] > z) hi = mid - 1 ;
		else                   return mid ;
	}

	return -1 ;
}

// pack an arc, identified by its lower-endpoint rank and its slot in up[lo],
// into the heap's NodeID field / the dedup set's key
#define ARC_KEY(lo, slot) \
	((void *)(uintptr_t)(((uint64_t)(lo) << 32) | (uint32_t)(slot)))

void CCH_RecustomizeScoped
(
	CCH           *cch,
	CCH_SeedFn     seed,
	void          *ctx,
	const int64_t *du,
	const int64_t *dv,
	uint64_t       k
) {
	ASSERT (cch       != NULL) ;
	ASSERT (cch->up   != NULL) ;
	ASSERT (cch->down != NULL) ;
	ASSERT (cch->up_w != NULL) ;   // a prior full customization must exist
	ASSERT (seed      != NULL) ;

	int64_t n = cch->n ;

	// 'queued' dedups arcs so each is processed at most once; the min-heap
	// (keyed by lower-endpoint rank) yields arcs in the order the forward sweep
	// would finalize them, so an arc is only recomputed once every arc it reads
	// is already final
	dict          *queued = HashTableCreate (&def_dt) ;
	NodeWeightHeap heap ;
	NodeWeightHeap_init (&heap) ;

	#define ENQUEUE(lo_, slot_)                                                 \
		do {                                                                    \
			void *_key = ARC_KEY ((lo_), (slot_)) ;                             \
			if (HashTableFetchValue (queued, _key) == NULL) {                   \
				HashTableAdd (queued, _key, (void *)(uintptr_t)1) ;             \
				NodeWeightHeap_offer (&heap,                                    \
					(NodeWeightItem){ .node = (NodeID)(uintptr_t)_key,          \
					                  .weight = (double)(lo_) }) ;              \
			}                                                                   \
		} while (0)

	// seed: each changed original edge's chordal arc becomes the initial dirty
	// front
	for (uint64_t i = 0 ; i < k ; i++) {
		int64_t a = (du [i] < (int64_t) n) ? cch->iperm [du [i]] : -1 ;
		int64_t b = (dv [i] < (int64_t) n) ? cch->iperm [dv [i]] : -1 ;
		if (a < 0 || b < 0 || a == b) continue ;
		int64_t lo   = a < b ? a : b ;
		int64_t hi   = a < b ? b : a ;
		int64_t slot = _find_upper_opt (cch, lo, hi) ;
		if (slot >= 0) ENQUEUE (lo, slot) ;
	}

	NodeWeightItem it ;
	while (NodeWeightHeap_poll (&heap, &it)) {
		uint64_t key  = (uint64_t) it.node ;
		int64_t  lo   = (int64_t) (key >> 32) ;
		int64_t  slot = (int64_t) (uint32_t) key ;
		int64_t  hi   = cch->up [lo] [slot] ;

		double old_up = cch->up_w [lo] [slot] ;
		double old_dn = cch->dn_w [lo] [slot] ;

		// reset the arc to its seed (raw original-edge weights, both directions)
		double s_lohi, s_hilo ;
		seed (ctx, cch->perm [lo], cch->perm [hi], &s_lohi, &s_hilo) ;

		double  best_up = s_lohi ; int64_t mid_up = -1 ;
		double  best_dn = s_hilo ; int64_t mid_dn = -1 ;

		// re-relax over every lower common neighbour x of lo and hi. a common
		// neighbour is exactly a rank in both lower-adjacency lists (x in down[lo]
		// and x in down[hi] => x < lo < hi, arcs (x,lo) and (x,hi) both exist), so
		// intersect the two ascending lists with a linear merge -- far cheaper for
		// high-rank arcs (large down[]) than probing up[x] for every lower
		// neighbour of lo
		int64_t *dlo = cch->down [lo] ;
		int64_t *dhi = cch->down [hi] ;
		uint32_t na = arr_len (dlo), nb = arr_len (dhi) ;
		uint32_t ai = 0, bi = 0 ;
		while (ai < na && bi < nb) {
			int64_t xa = dlo [ai], xb = dhi [bi] ;
			if      (xa < xb) { ai++ ; continue ; }
			if      (xa > xb) { bi++ ; continue ; }

			int64_t x  = xa ;                            // common lower neighbour
			int64_t iy = _find_upper_opt (cch, x, lo) ;  // slot of lo in up[x]
			int64_t iz = _find_upper_opt (cch, x, hi) ;  // slot of hi in up[x]

			// detour lo -> x -> hi : (lo->x) + (x->hi) = dn_w[x][iy] + up_w[x][iz]
			double cand_up = cch->dn_w [x] [iy] + cch->up_w [x] [iz] ;
			if (cand_up < best_up) { best_up = cand_up ; mid_up = x ; }

			// detour hi -> x -> lo : (hi->x) + (x->lo) = dn_w[x][iz] + up_w[x][iy]
			double cand_dn = cch->dn_w [x] [iz] + cch->up_w [x] [iy] ;
			if (cand_dn < best_dn) { best_dn = cand_dn ; mid_dn = x ; }

			ai++ ; bi++ ;
		}

		cch->up_w   [lo] [slot] = best_up ;
		cch->up_mid [lo] [slot] = mid_up ;
		cch->dn_w   [lo] [slot] = best_dn ;
		cch->dn_mid [lo] [slot] = mid_dn ;

		// if the arc actually changed, every arc that reads it might change too:
		// the top arcs of the apex-'lo' triangles {lo, hi, w}, w in up[lo]
		if (best_up != old_up || best_dn != old_dn) {
			int64_t *ul = cch->up [lo] ;
			for (uint32_t wi = 0 ; wi < arr_len (ul) ; wi++) {
				int64_t w = ul [wi] ;
				if (w == hi) continue ;
				int64_t d_lo   = hi < w ? hi : w ;
				int64_t d_hi   = hi < w ? w  : hi ;
				int64_t d_slot = _find_upper_opt (cch, d_lo, d_hi) ;
				if (d_slot >= 0) ENQUEUE (d_lo, d_slot) ;
			}
		}
	}

	#undef ENQUEUE

	NodeWeightHeap_free (&heap) ;
	HashTableRelease (queued) ;
}

#undef ARC_KEY

bool CCH_HasArc
(
	const CCH *cch,
	int64_t    a,
	int64_t    b
) {
	ASSERT (cch != NULL) ;

	if (a == b || a < 0 || b < 0 || a >= cch->n || b >= cch->n) {
		return false ;
	}

	int64_t lo = a < b ? a : b ;
	int64_t hi = a < b ? b : a ;
	return _find_upper_opt (cch, lo, hi) >= 0 ;
}

void CCH_ExtractShortcuts
(
	const CCH       *cch,
	const GrB_Matrix W,
	GrB_Matrix      *S,
	GrB_Matrix      *M
) {
	ASSERT (cch         != NULL) ;
	ASSERT (cch->up_w   != NULL) ;   // Phase 2 must have run
	ASSERT (cch->up_mid != NULL) ;
	ASSERT (W           != NULL) ;
	ASSERT (S           != NULL) ;
	ASSERT (M           != NULL) ;

	int64_t n = cch->n ;

	GrB_Matrix _S = NULL ;
	GrB_Matrix _M = NULL ;
	GrB_OK (GrB_Matrix_new (&_S, GrB_FP64,  n, n)) ;
	GrB_OK (GrB_Matrix_new (&_M, GrB_INT64, n, n)) ;

	for (int64_t a = 0 ; a < n ; a++) {
		int64_t  u   = cch->perm [a] ;                   // lower-rank node id
		int64_t *ua  = cch->up [a] ;
		int64_t  deg = (int64_t) arr_len (ua) ;

		for (int64_t i = 0 ; i < deg ; i++) {
			int64_t v    = cch->perm [ua [i]] ;          // higher-rank node id
			double  up_w = cch->up_w [a] [i] ;           // u -> v
			double  dn_w = cch->dn_w [a] [i] ;           // v -> u
			double  road ;

			// up arc u -> v: emit only if it improves on (or replaces a
			// missing) road edge. up_w is seeded from W and only ever
			// decreases, so up_w == road (bit-identical) means the road edge
			// is already optimal and no shortcut is needed. an improving arc
			// always got its weight from a triangle, so it has a middle.
			if (up_w != INFINITY) {
				GrB_Info info =
					GrB_Matrix_extractElement_FP64 (&road, W, u, v) ;
				if (info != GrB_SUCCESS || up_w < road) {
					int64_t mid = cch->up_mid [a] [i] ;
					ASSERT (mid != -1) ;
					GrB_OK (GrB_Matrix_setElement_FP64 (_S, up_w, u, v)) ;
					GrB_OK (GrB_Matrix_setElement_INT64 (_M,
								cch->perm [mid], u, v)) ;
				}
			}

			// down arc v -> u
			if (dn_w != INFINITY) {
				GrB_Info info =
					GrB_Matrix_extractElement_FP64 (&road, W, v, u) ;
				if (info != GrB_SUCCESS || dn_w < road) {
					int64_t mid = cch->dn_mid [a] [i] ;
					ASSERT (mid != -1) ;
					GrB_OK (GrB_Matrix_setElement_FP64 (_S, dn_w, v, u)) ;
					GrB_OK (GrB_Matrix_setElement_INT64 (_M,
								cch->perm [mid], v, u)) ;
				}
			}
		}
	}

	GrB_OK (GrB_Matrix_wait (_S, GrB_MATERIALIZE)) ;
	GrB_OK (GrB_Matrix_wait (_M, GrB_MATERIALIZE)) ;
	*S = _S ;
	*M = _M ;
}

// Phase 3 (query) is not built into this module. The materialized SHORTCUT
// edges + node ranks are queried by the stateless, concurrency-safe
// rank-pruned bidirectional Dijkstra in proc_cch_query.c instead.

//------------------------------------------------------------------------------
// RDB serialization (full hierarchy)
//------------------------------------------------------------------------------

void CCH_RdbSave
(
	const CCH   *cch,
	SerializerIO io
) {
	ASSERT (cch != NULL) ;
	ASSERT (cch->up != NULL) ;   // a built hierarchy

	int64_t n = cch->n ;
	SerializerIO_WriteSigned (io, n) ;

	// perm[n] (iperm is derived on load)
	SerializerIO_WriteBuffer (io, cch->perm, n * sizeof (int64_t)) ;

	// per rank: degree + upward adjacency + per-arc weights/middles
	// (down[] is derived on load)
	for (int64_t r = 0 ; r < n ; r++) {
		uint32_t deg = arr_len (cch->up [r]) ;
		SerializerIO_WriteUnsigned (io, deg) ;
		if (deg > 0) {
			SerializerIO_WriteBuffer (io, cch->up     [r], deg * sizeof (int64_t)) ;
			SerializerIO_WriteBuffer (io, cch->up_w   [r], deg * sizeof (double))  ;
			SerializerIO_WriteBuffer (io, cch->dn_w   [r], deg * sizeof (double))  ;
			SerializerIO_WriteBuffer (io, cch->up_mid [r], deg * sizeof (int64_t)) ;
			SerializerIO_WriteBuffer (io, cch->dn_mid [r], deg * sizeof (int64_t)) ;
		}
	}
}

// read 'nbytes' from 'io' into a freshly-allocated 'dst' array (rm-owned).
// SerializerIO_ReadBuffer returns an rm-allocated buffer we copy then free, so
// 'dst' follows the CCH allocation convention (rm_malloc / arr_ data)
static void _read_into
(
	SerializerIO io,
	void        *dst,
	size_t       nbytes
) {
	size_t len ;
	void *buf = SerializerIO_ReadBuffer (io, &len) ;
	ASSERT (len == nbytes) ;
	memcpy (dst, buf, nbytes) ;
	rm_free (buf) ;
}

CCH *CCH_RdbLoad
(
	SerializerIO io
) {
	int64_t n = SerializerIO_ReadSigned (io) ;

	CCH *cch = CCH_New (n) ;   // n set; every array NULL

	// perm[n]
	cch->perm = rm_malloc (sizeof (int64_t) * n) ;
	_read_into (io, cch->perm, n * sizeof (int64_t)) ;

	// per-rank pointer arrays
	cch->up     = rm_calloc (n, sizeof (int64_t *)) ;
	cch->up_w   = rm_malloc (sizeof (double  *) * n) ;
	cch->dn_w   = rm_malloc (sizeof (double  *) * n) ;
	cch->up_mid = rm_malloc (sizeof (int64_t *) * n) ;
	cch->dn_mid = rm_malloc (sizeof (int64_t *) * n) ;

	for (int64_t r = 0 ; r < n ; r++) {
		uint32_t deg   = SerializerIO_ReadUnsigned (io) ;
		int64_t  slots = deg > 0 ? deg : 1 ;   // match CCH_Customize's layout

		cch->up     [r] = arr_newlen (int64_t, deg) ;   // arr_ with len == deg
		cch->up_w   [r] = rm_malloc (sizeof (double)  * slots) ;
		cch->dn_w   [r] = rm_malloc (sizeof (double)  * slots) ;
		cch->up_mid [r] = rm_malloc (sizeof (int64_t) * slots) ;
		cch->dn_mid [r] = rm_malloc (sizeof (int64_t) * slots) ;

		if (deg > 0) {
			_read_into (io, cch->up     [r], deg * sizeof (int64_t)) ;
			_read_into (io, cch->up_w   [r], deg * sizeof (double))  ;
			_read_into (io, cch->dn_w   [r], deg * sizeof (double))  ;
			_read_into (io, cch->up_mid [r], deg * sizeof (int64_t)) ;
			_read_into (io, cch->dn_mid [r], deg * sizeof (int64_t)) ;
		}
	}

	// derive iperm from perm; xadj/adjncy/parent stay NULL (Phase-1 scratch)
	cch->iperm = rm_malloc (sizeof (int64_t) * n) ;
	for (int64_t r = 0 ; r < n ; r++) {
		cch->iperm [cch->perm [r]] = r ;
	}

	// derive down[] from up[]
	_build_down_adjacency (cch) ;

	return cch ;
}
