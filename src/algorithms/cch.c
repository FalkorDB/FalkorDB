/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "metis.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "cch.h"

#include <math.h>

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

	if (cch->q_df != NULL) rm_free (cch->q_df) ;
	if (cch->q_db != NULL) rm_free (cch->q_db) ;

	rm_free (cch) ;
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
	CCH        *cch,
	GrB_Matrix  W
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

	double **up_w = rm_malloc (sizeof (double *) * n) ;
	double **dn_w = rm_malloc (sizeof (double *) * n) ;

	//--------------------------------------------------------------------------
	// seed: original edges take their weight from W, shortcut arcs +INFINITY
	//--------------------------------------------------------------------------

	for (int64_t rank = 0 ; rank < n ; rank++) {
		int64_t u   = cch->perm [rank] ;                 // node id at this rank
		int64_t deg = (int64_t) arr_len (cch->up [rank]) ;

		// deg can be 0 (a root with no upper neighbors); keep a 1-slot alloc
		// so the pointer is always non-NULL and freeable
		up_w [rank] = rm_malloc (sizeof (double) * (deg > 0 ? deg : 1)) ;
		dn_w [rank] = rm_malloc (sizeof (double) * (deg > 0 ? deg : 1)) ;

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

				// improve up(y,z) via the detour y -> x -> z
				double cand_up = wxy_dn + wxz_up ;
				if (cand_up < up_w [y] [k]) up_w [y] [k] = cand_up ;

				// improve dn(y,z) via the detour z -> x -> y
				double cand_dn = wxz_dn + wxy_up ;
				if (cand_dn < dn_w [y] [k]) dn_w [y] [k] = cand_dn ;
			}
		}
	}

	cch->up_w = up_w ;
	cch->dn_w = dn_w ;
}

//------------------------------------------------------------------------------
// Phase 3: query
//------------------------------------------------------------------------------

bool CCH_Query
(
	CCH     *cch,
	int64_t  src,
	int64_t  dst,
	double  *weight
) {
	ASSERT (cch         != NULL) ;
	ASSERT (cch->up_w   != NULL) ;   // Phase 2 must have run
	ASSERT (cch->parent != NULL) ;
	ASSERT (src >= 0 && src < cch->n) ;
	ASSERT (dst >= 0 && dst < cch->n) ;

	int64_t n = cch->n ;

	// lazily allocate the two scratch distance arrays. they are held at
	// +INFINITY between queries: each query touches only the O(tree-height)
	// ancestors it walks and restores exactly those entries before returning,
	// so the O(n) initialization here happens once, not per query.
	if (cch->q_df == NULL) {
		cch->q_df = rm_malloc (sizeof (double) * n) ;
		cch->q_db = rm_malloc (sizeof (double) * n) ;
		for (int64_t i = 0 ; i < n ; i++) {
			cch->q_df [i] = INFINITY ;
			cch->q_db [i] = INFINITY ;
		}
	}

	double  *df     = cch->q_df ;
	double  *db     = cch->q_db ;
	int64_t *parent = cch->parent ;

	int64_t rs = cch->iperm [src] ;
	int64_t rt = cch->iperm [dst] ;

	// forward search: walk src's ancestor path, relaxing up-arcs (up_w). every
	// up-neighbor of a node on this path is itself an ancestor of src (the
	// child-absorption property T_G is built on), so the search never leaves
	// the path -- no priority queue needed.
	df [rs] = 0.0 ;
	for (int64_t x = rs ; x != -1 ; x = parent [x]) {
		double dfx = df [x] ;
		int64_t *ux = cch->up [x] ;
		int64_t  dx = (int64_t) arr_len (ux) ;
		for (int64_t i = 0 ; i < dx ; i++) {
			double cand = dfx + cch->up_w [x] [i] ;
			if (cand < df [ux [i]]) df [ux [i]] = cand ;
		}
	}

	// backward search: walk dst's ancestor path, relaxing down-arcs (dn_w)
	db [rt] = 0.0 ;
	for (int64_t x = rt ; x != -1 ; x = parent [x]) {
		double dbx = db [x] ;
		int64_t *ux = cch->up [x] ;
		int64_t  dx = (int64_t) arr_len (ux) ;
		for (int64_t i = 0 ; i < dx ; i++) {
			double cand = dbx + cch->dn_w [x] [i] ;
			if (cand < db [ux [i]]) db [ux [i]] = cand ;
		}
	}

	// combine: the optimal src->dst path's apex (its highest-ranked node) is a
	// common ancestor of src and dst, reached by the forward search from below
	// and the backward search from above. minimize df + db over src's ancestor
	// path; db is finite only at nodes that are also dst's ancestors, so this
	// implicitly ranges over the shared ancestors.
	double best = INFINITY ;
	for (int64_t x = rs ; x != -1 ; x = parent [x]) {
		double s = df [x] + db [x] ;
		if (s < best) best = s ;
	}

	// reset touched entries to +INFINITY. every df write lands on src's
	// ancestor path and every db write on dst's, so re-walking both paths
	// clears exactly what the two searches dirtied.
	for (int64_t x = rs ; x != -1 ; x = parent [x]) df [x] = INFINITY ;
	for (int64_t x = rt ; x != -1 ; x = parent [x]) db [x] = INFINITY ;

	if (best == INFINITY) return false ;

	*weight = best ;
	return true ;
}
