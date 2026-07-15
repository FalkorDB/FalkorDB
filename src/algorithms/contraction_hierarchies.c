/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "GraphBLAS.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "../graph/entities/graph_entity.h"
#include "utils/node_map.h"
#include "utils/priority_heap.h"
#include "contraction_hierarchies.h"

// forward declaration -- Preprocess (below) calls this before its definition
static int64_t calculate_importance (GrB_Matrix A, GrB_Matrix AT, NodeID node) ;

//------------------------------------------------------------------------------
// NodeWeightHeap: min-heap of (node, weight) candidates
//------------------------------------------------------------------------------

// per-node label used by the Dijkstra search below
typedef struct {
	double weight;    // current best known weight to reach this node
	bool   finalized; // true once popped from the heap with its optimal weight
} DijkstraLabel;

//------------------------------------------------------------------------------
// Dijkstra
//------------------------------------------------------------------------------

// bounded witness search: reports whether a path from 'src_id' to 'dst_id'
// exists in 'A' that avoids 'skip_id' entirely and whose total weight is at
// most 'max_weight'. does not reconstruct the path or report its exact
// weight -- callers only need the yes/no answer to decide whether 'skip_id'
// is safe to contract without adding a shortcut edge (see preprocess()
// below).
//
// ASSUMES every edge weight in 'A' is non-negative; Dijkstra's "finalize
// once" invariant does not hold otherwise, and this is not detected here.
static bool Dijkstra_ShortestPath
(
	GrB_Matrix A,
	NodeID src_id,
	NodeID dst_id,
	NodeID skip_id,
	double max_weight
) {
	// 'labels' holds one DijkstraLabel per node ever discovered (tentative
	// or finalized best-known distance).
	// 'label_idx' maps a node id to its 1-based slot in 'labels' (0 means
	// "not yet discovered").
	// 'heap' is the Dijkstra priority queue: pending (node, weight)
	// candidates ordered so the next NodeWeightHeap_poll always returns the
	// smallest-weight candidate discovered so far.
	NodeMap label_idx ;
	NodeMap_init (&label_idx) ;

	DijkstraLabel *labels = arr_new (DijkstraLabel, 64) ;

	NodeWeightHeap heap ;
	NodeWeightHeap_init (&heap) ;

    GxB_Iterator it ;
    GrB_OK (GxB_Iterator_new (&it)) ;
    GrB_OK (GxB_rowIterator_attach (it, A, NULL)) ;

	// initialization: seed the source node with distance 0 and no parent
	// (it parents itself, which also makes the path-reconstruction loop's
	// "cur != src_id" stop condition correct). every other node is
	// implicitly at distance +inf until first discovered below.
	DijkstraLabel src_label =
		{ .weight = 0, .finalized = false };

	arr_append (labels, src_label) ;

	uint32_t *src_slot = NodeMap_findOrInsert (&label_idx, src_id, NULL) ;
	*src_slot = arr_len (labels) ;

	// push the source onto the priority queue so the main loop below has
	// somewhere to start.
	NodeWeightItem seed = { .node = src_id, .weight = 0 };

	NodeWeightHeap_offer (&heap, seed) ;

	bool found = false ;

	// main Dijkstra loop: repeatedly extract the not-yet-finalized node
	// with the smallest tentative distance and finalize it -- that
	// distance is now guaranteed optimal, since all edge weights are
	// non-negative and every unexplored candidate is at least as large.
	// stops either when dst is finalized (found) or the heap empties
	// (dst unreachable from src).
	while (!found) {
		// extract the minimum-weight candidate. this may be a stale
		// duplicate left over from a relaxation performed after this
		// entry was queued (see the lazy-deletion note on NodeWeightItem);
		// staleness is detected below via the label's 'finalized' flag
		// rather than by removing superseded heap entries in place.
		NodeWeightItem item ;
		if (!NodeWeightHeap_poll (&heap, &item)) {
			break ;  // heap exhausted: dst is unreachable
		}

		NodeID cur = item.node ;

		uint32_t cur_idx = NodeMap_find (&label_idx, cur) ;

		ASSERT (cur_idx != 0) ;
		if (labels [cur_idx - 1].finalized) {
			continue ;  // stale duplicate entry
		}

		// finalize 'cur': its current label weight is its true shortest
		// distance from src and will never be improved again (label
		// setting -- each node is finalized exactly once).
		labels [cur_idx - 1].finalized = true ;

		// dst just got finalized, its shortest path is settled: stop
		// early instead of exploring the rest of the reachable graph.
		if (cur == dst_id) {
			found = true ;
			break ;
		}

		double cur_weight = labels [cur_idx - 1].weight ;

		if (cur_weight > max_weight) {
			break ;  // break overweight
		}

		// relaxation step: examine every edge and try to improve each
		// neighbor's tentative distance through 'cur'.
		GrB_Info info = GxB_rowIterator_seekRow (it, cur) ;
		// iterate over entries in A(i,:)
		while (info == GrB_SUCCESS) {
			// get the entry A(i,j)
			GrB_Index nid = GxB_rowIterator_getColIndex (it) ;
			double w = GxB_Iterator_get_FP64 (it) ;

			// move to the next entry in A(i,:)
			info = GxB_rowIterator_nextCol (it) ;

			if (nid == cur || nid == skip_id) {
				continue ;  // ignore self-loops
			}

			// candidate distance to 'nid' going through 'cur' via
			// this edge: cur's finalized distance plus the edge's
			// weight.
			// NOTE: weightProp is assumed non-negative here (see the
			// function-level comment above); a negative value would
			// silently make this search's result incorrect.
			double new_weight = cur_weight + w ;
			if (new_weight > max_weight) {
				continue ;  // skip overweight
			}

			// look up (or reserve) 'nid's slot in 'labels'
			bool is_new ;
			uint32_t *nslot =
				NodeMap_findOrInsert (&label_idx, nid, &is_new) ;

			if (!is_new) {
				// 'nid' already labeled: this is the relaxation
				// comparison proper. skip if it's already finalized
				// (its distance is final and can't improve) or if
				// going through 'cur' isn't strictly better than what
				// it already has.
				DijkstraLabel *nlabel = labels + (*nslot - 1);
				if(nlabel->finalized || new_weight >= nlabel->weight) {
					continue ;
				}

				// found a strictly shorter route to 'nid' through
				// 'cur': update its label in place with the new best
				// distance, parent and connecting edge.
				nlabel->weight = new_weight ;
			} else {
				// first time 'nid' is discovered: create its label
				// with 'cur' as parent and 'new_weight' as its (so
				// far unbeaten) tentative distance.
				DijkstraLabel nlabel =
				{ .weight = new_weight, .finalized = false };

				arr_append (labels, nlabel) ;
				*nslot = arr_len (labels) ;
			}

			// queue (or re-queue) 'nid' at its updated tentative
			// weight. any older, now-superseded heap entry for 'nid'
			// is left in place and simply skipped later as a stale
			// duplicate once popped.
			NodeWeightItem qi = { .node = nid, .weight = new_weight } ;
			NodeWeightHeap_offer (&heap, qi) ;
		}
	}

	// search is over (dst found or heap exhausted): entries are stored by
	// value, so there's nothing to drain, just free the heap itself.
    GrB_free (&it) ;
	arr_free (labels) ;
	NodeMap_free (&label_idx) ;
	NodeWeightHeap_free (&heap) ;

	return found ;
}

// fetches 'v's immediate neighbors on both sides: '*incoming' becomes a
// sparse vector over all nodes u with an edge u->v (weight(u,v) as the
// vector's value at u), and '*outgoing' becomes the same for v's w->
// successors (edge v->w). both are freshly allocated here -- the caller
// owns them and must free them
static void get_neighbors
(
	GrB_Vector *u,  // incoming u -> v
	GrB_Vector *w,  // outgoing v -> w
	GrB_Matrix A,
	GrB_Matrix AT,
	NodeID v
) {
	ASSERT (u != NULL) ;
	ASSERT (w != NULL) ;

	GrB_Index nrows ;
	GrB_OK (GrB_Matrix_nrows (&nrows, A)) ;

	GrB_OK (GrB_Vector_new  (u, GrB_FP64, nrows)) ;
	GrB_OK (GrB_Vector_new  (w, GrB_FP64, nrows)) ;

	// T0 (transpose first input) turns a column-v extract into a row-v
	// extract: row v of AT is v's incoming neighbors (u -> v), row v of A
	// is v's outgoing neighbors (v -> w) -- matches the row=src/col=dst
	// convention used throughout this file (e.g. Dijkstra_ShortestPath's
	// row iterator over A).
	GrB_Descriptor desc = GrB_DESC_T0 ;
	GrB_OK (GrB_Col_extract (*u, NULL, NULL, AT, GrB_ALL, nrows, v, desc)) ;
	GrB_OK (GrB_Col_extract (*w, NULL, NULL, A,  GrB_ALL, nrows, v, desc)) ;
}

// computes a contraction order for 'A': every node gets an "importance"
// score (E_shortcuts - E_incident, see below), nodes are sorted ascending
// by that score, and the result is loaded into a min-heap so the
// lowest-importance (cheapest to contract) node can always be popped next.
//
//   E_incident:  the node's in-degree + out-degree -- how many original
//                edges touch it.
//   E_shortcuts: for every pair of neighbors (u, v) with u->v->w, whether a
//                witness path from u to w avoiding v was found (see
//                Dijkstra_ShortestPath above). E_shortcuts counts the pairs
//                where NO witness exists -- i.e. the shortcut edges that
//                would have to be added if v were contracted.
//
// a low (or negative) importance score means contracting the node adds few
// or no new shortcuts relative to the edges it removes -- a good early
// contraction candidate
static NodeWeightHeap *Preprocess
(
	GrB_Matrix A,
	GrB_Matrix AT
) {
	GrB_Type  t     ;
	GrB_Index nrows ;
	GrB_Index ncols ;

	GrB_OK (GxB_Matrix_type  (&t,     A)) ;
	GrB_OK (GrB_Matrix_nrows (&nrows, A)) ;
	GrB_OK (GrB_Matrix_ncols (&ncols, A)) ;

	ASSERT (nrows == ncols) ;

	//--------------------------------------------------------------------------
	// reduce A
	// importance[i] = true if A[i:] isn't empty
	//--------------------------------------------------------------------------

	GrB_Vector importance ;
	GrB_OK (GrB_Vector_new (&importance, GrB_INT64, nrows)) ;

	GrB_OK (GrB_Matrix_reduce_Monoid (importance, NULL, NULL,
				GxB_ANY_BOOL_MONOID, A, NULL)) ;

	GrB_OK (GrB_Matrix_reduce_Monoid (importance, importance, NULL,
				GxB_ANY_BOOL_MONOID, AT, GrB_DESC_SC)) ;

    GxB_Iterator it ;
    GxB_Iterator_new (&it) ;

    // attach it to the vector importance
    GrB_OK (GxB_Vector_Iterator_attach (it, importance, NULL)) ;

    GrB_Info info = GxB_Vector_Iterator_seek (it, 0) ;

	//--------------------------------------------------------------------------
	// compute importance for each node
	//--------------------------------------------------------------------------

    while (info != GxB_EXHAUSTED) {
        // get the entry v(i)
        GrB_Index v = GxB_Vector_Iterator_getIndex (it) ;
		int64_t v_importance = calculate_importance (A, AT, v) ;

		// update importance to hold importance
		GrB_OK (GrB_Vector_setElement (importance, v_importance, v)) ;

        // move to the next entry in v
        info = GxB_Vector_Iterator_next (it) ;
    }

	//--------------------------------------------------------------------------
	// sort by importance ascending
	//--------------------------------------------------------------------------

	GrB_Vector permutation ;
	GrB_OK (GrB_Vector_new (&permutation, GrB_INT64, nrows)) ;

	GrB_OK (GxB_Vector_sort (
				NULL,            // vector of sorted values
				permutation,     // vector containing the permutation
				GrB_LT_INT64,    // comparator op (must return GrB_BOOL --
				                 // the GxB_IS* family returns the same type
				                 // as its inputs instead, which
				                 // GxB_Vector_sort rejects)
				importance,      // vector to sort
				NULL)) ;

	//--------------------------------------------------------------------------
	// populate heap
	//--------------------------------------------------------------------------

	NodeWeightHeap *heap = rm_malloc (sizeof (NodeWeightHeap)) ;
	NodeWeightHeap_init (heap) ;

    GrB_OK (GxB_Vector_Iterator_attach (it, permutation, NULL)) ;
    info = GxB_Vector_Iterator_seek (it, 0) ;
    while (info != GxB_EXHAUSTED) {
        GrB_Index n_id = GxB_Iterator_get_INT64 (it) ;

		int64_t w ;
		GrB_OK (GrB_Vector_extractElement (&w, importance, n_id)) ;

		NodeWeightItem item = { .node = n_id, .weight = w };
		NodeWeightHeap_offer (heap, item) ;

        info = GxB_Vector_Iterator_next (it) ;
	}

    GrB_OK (GrB_free (&it))          ;
    GrB_OK (GrB_free (&importance))  ;
	GrB_OK (GrB_free (&permutation)) ;

	return heap ;
}

// dual-purpose: contracts 'v' out of the graph, or (with 'dryrun' set)
// just scores what contracting it would cost right now, without changing
// anything -- 'calculate_importance' below is exactly that dry run.
//
// for every pair of 'v's neighbors (u, w) with edges u->v and v->w, checks
// whether a witness path from u to w already exists that avoids v and
// costs no more than weight(u,v) + weight(v,w) (see Dijkstra_ShortestPath
// above). if no such witness exists, contracting v would require a new
// SHORTCUT edge u->w to preserve shortest-path distances once v is
// removed.
//
// when 'dryrun' is false, those shortcuts are actually inserted into 'A',
// 'AT', and the running shortcut-only overlay 'S' (which the caller owns
// across the whole contraction run -- see Contract below), and 'v' (with
// its incident edges) is removed from the active graph. when 'dryrun' is
// true, 'S' is ignored (pass NULL) and nothing is modified -- this just
// counts how many shortcuts *would* be needed.
//
// if 'importance' is non-NULL, '*importance' is set to 'v's importance
// score (e_shortcuts - e_incident, see Preprocess above), computed either
// way regardless of 'dryrun'.
static void ContractNode
(
	NodeID v,
	GrB_Matrix A,
	GrB_Matrix AT,
	GrB_Matrix S,  // [output, only used when !dryrun] accumulates every
	               // shortcut edge inserted across the whole contraction
	               // run, on top of the ones merged into A/AT directly
	bool dryrun,
	int64_t *importance
) {

	int64_t e_shortcuts = 0 ;

	GrB_Vector incoming ;  // u -> v
	GrB_Vector outgoing ;  // v -> w
	get_neighbors (&incoming, &outgoing, A, AT, v) ;

	// u -> v -> w
    GxB_Iterator u_it ;
    GxB_Iterator w_it ;

    GxB_Iterator_new (&u_it) ;
    GxB_Iterator_new (&w_it) ;

    GrB_OK (GxB_Vector_Iterator_attach (u_it, incoming, NULL)) ;
    GrB_OK (GxB_Vector_Iterator_attach (w_it, outgoing, NULL)) ;

	GrB_Index nrows ;
	GrB_Index ncols ;
	GrB_OK (GrB_Matrix_nrows (&nrows, A)) ;
	GrB_OK (GrB_Matrix_ncols (&ncols, A)) ;

	GrB_Matrix shortcuts = NULL ;
	if (!dryrun) {
		GrB_OK (GrB_Matrix_new (&shortcuts, GrB_FP64, nrows, ncols)) ;
	}

    GrB_Info u_info = GxB_Vector_Iterator_seek (u_it, 0) ;
	while (u_info != GxB_EXHAUSTED) {
		GrB_Index u = GxB_Vector_Iterator_getIndex (u_it) ;
		double uv_weight = GxB_Iterator_get_FP64 (u_it) ;

		GrB_Info w_info = GxB_Vector_Iterator_seek (w_it, 0) ;
		while (w_info != GxB_EXHAUSTED) {
			GrB_Index w = GxB_Vector_Iterator_getIndex (w_it) ;

			// pair (u, w) u -> v -> w
			// direct path
			double vw_weight = GxB_Iterator_get_FP64 (w_it) ;
			double weight = uv_weight + vw_weight ;

			if (!Dijkstra_ShortestPath (A, u, w, v, weight)) {
				// could not find a shortest path which doesn't go through v
				e_shortcuts++ ;
				if (!dryrun) {
					// insert the shortcut edge u->w (weight 'weight')
					GrB_OK (GrB_Matrix_setElement (shortcuts, weight, u, w)) ;
				}
			}

			// move to the next entry in w
			w_info = GxB_Vector_Iterator_next (w_it) ;
		}

		// move to the next entry in u
		u_info = GxB_Vector_Iterator_next (u_it) ;
	}

	GrB_OK (GrB_free (&u_it)) ;
	GrB_OK (GrB_free (&w_it)) ;

	if (importance != NULL) {
		GrB_Index nvals ;
		int64_t e_incident  = 0 ;

		GrB_OK (GrB_Vector_nvals (&nvals, outgoing)) ;
		e_incident = nvals ;

		GrB_OK (GrB_Vector_nvals (&nvals, incoming)) ;
		e_incident += nvals ;

		*importance = e_shortcuts - e_incident ;
	}

	if (dryrun == false && shortcuts != NULL) {
		//----------------------------------------------------------------------
		// accumulate shortcuts
		//----------------------------------------------------------------------

		// S  += shortcuts
		// A  += shortcuts
		// AT += Transpose(shortcuts)

		GrB_OK (GrB_Matrix_assign (S, NULL, GrB_MIN_FP64, shortcuts, GrB_ALL, nrows,
					GrB_ALL, ncols, NULL)) ;

		GrB_OK (GrB_Matrix_assign (A, NULL, GrB_MIN_FP64, shortcuts, GrB_ALL, nrows,
					GrB_ALL, ncols, NULL)) ;

		GrB_OK (GrB_Matrix_assign (AT, NULL, GrB_MIN_FP64, shortcuts, GrB_ALL, nrows,
					GrB_ALL, ncols, GrB_DESC_T0)) ;

		//----------------------------------------------------------------------
		// clear v's entries from A & AT
		//----------------------------------------------------------------------

		GrB_Scalar x ;
		GrB_OK (GrB_Scalar_new (&x, GrB_FP64)) ;

		// clear A[v:], A[:v], AT[v:] & AT[:v]
		GrB_OK (GrB_Matrix_assign_Scalar (A,  NULL, NULL, x, &v, 1, GrB_ALL, ncols, NULL)) ;
		GrB_OK (GrB_Matrix_assign_Scalar (A,  NULL, NULL, x, GrB_ALL, nrows, &v, 1, NULL)) ;
		GrB_OK (GrB_Matrix_assign_Scalar (AT, NULL, NULL, x, &v, 1, GrB_ALL, ncols, NULL)) ;
		GrB_OK (GrB_Matrix_assign_Scalar (AT, NULL, NULL, x, GrB_ALL, nrows, &v, 1, NULL)) ;

		GrB_OK (GrB_free (&x));
	}

	// clean up
	GrB_OK (GrB_free (&outgoing)) ;
	GrB_OK (GrB_free (&incoming)) ;
	GrB_OK (GrB_free (&shortcuts)) ;
}

// scores 'node's importance without modifying the graph -- a thin wrapper
// around ContractNode's dry-run mode. see ContractNode above for what the
// score means and how it's computed.
static int64_t calculate_importance
(
	GrB_Matrix A,
	GrB_Matrix AT,
	NodeID node
) {
	int64_t importance = 0 ;
	ContractNode (node, A, AT, NULL, true, &importance) ;
	return importance ;
}

// builds a contraction hierarchy over 'A': repeatedly picks the
// least-important remaining node (per Preprocess's initial ordering) and
// contracts it via ContractNode, adding whatever shortcut edges are
// needed to preserve shortest-path distances. 'A' itself is mutated in
// place as scratch space for the run (shortcuts merged in, contracted
// nodes' edges cleared) and should not be relied on by the caller
// afterward -- the two outputs that matter are '*S_out' (every shortcut
// added along the way, across all contractions, keyed by its final
// weight) and '*rank_out' (each contracted node's 1-based contraction
// rank). caller owns both and must free them.
//
// a popped node's score can go stale by the time it's popped (an earlier
// contraction may have added/removed shortcuts touching it), so before
// committing, its importance is recomputed against the current graph
// state; if that's no better than the next candidate still waiting in the
// heap, contracting it now is still optimal and it proceeds, otherwise
// it's re-queued with the fresh score and reconsidered later. this lazy
// re-scoring avoids recomputing every remaining node's importance after
// every single contraction.
void ContractionHierarchies_Contract
(
	GrB_Matrix  A,
	GrB_Matrix *S_out,
	GrB_Vector *rank_out
) {
	ASSERT (S_out    != NULL) ;
	ASSERT (rank_out != NULL) ;

	GrB_Type  t     ;
	GrB_Index nrows ;
	GrB_Index ncols ;
	GrB_OK (GxB_Matrix_type  (&t,     A)) ;
	GrB_OK (GrB_Matrix_nrows (&nrows, A)) ;
	GrB_OK (GrB_Matrix_ncols (&ncols, A)) ;

	// compute AT
	GrB_Matrix AT = NULL ;
	GrB_OK (GrB_Matrix_new (&AT, t, ncols, nrows))    ;
	GrB_OK (GrB_transpose  (AT, NULL, NULL, A, NULL)) ;

	// per-node contraction rank, 1-based; only nodes that actually get
	// contracted (i.e. have at least one incident edge) end up with an
	// entry here
	GrB_Vector rank_vec = NULL ;
	GrB_OK (GrB_Vector_new (&rank_vec, GrB_INT64, nrows)) ;

	// step 1: node ordering (heuristic) before contracting determine the order
	// in which nodes will be contracted, avoid contracting a major
	// "highway intersection" first, start with dead ends and
	// "minor streets" first, maintain a priority queue of nodes based on an
	// importance metric
	NodeWeightHeap *heap = Preprocess (A, AT) ;

	// running shortcut-only overlay: every shortcut edge inserted across
	// the whole contraction run, on top of the ones merged into A/AT
	// directly -- see ContractNode
	GrB_Matrix S = NULL ;
	GrB_OK (GrB_Matrix_new (&S, GrB_FP64, nrows, ncols)) ;

	NodeWeightItem item ;
	int64_t rank = 0 ;

	while (NodeWeightHeap_poll (heap, &item)) {
		// 2. recalculate its actual score based on the current graph state
		int64_t actual_score = calculate_importance (A, AT, item.node) ;

		NodeWeightItem peek ;
		bool perform_contraction = true ;
		if (likely (NodeWeightHeap_peek (heap, &peek))) {
			perform_contraction = (actual_score <= peek.weight) ;
		}

		if (perform_contraction) {
			// if its actual score is still less than or equal to the next best
			// thing in the heap, it is guaranteed to be the absolute minimum!
			ContractNode (item.node, A, AT, S, false, NULL) ;

			// assign node's rank
			rank++ ;
			GrB_OK (GrB_Vector_setElement_INT64 (rank_vec, rank, item.node)) ;

			// periodic progress heartbeat -- not per-node (this loop can run
			// over millions of nodes), just often enough to confirm a long
			// run is still making progress
			if (rank % 1000 == 0) {
				RedisModule_Log (NULL, "notice",
						"algo.contractionHierarchies: %lld nodes contracted "
						"so far", (long long) rank) ;
			}
		} else {
			// if its score went up and is now worse than the new top of the heap
			// update its score and throw it back in
			item.weight = actual_score ;
			NodeWeightHeap_offer (heap, item) ;
		}
	}

	GrB_OK (GrB_free (&AT)) ;
	NodeWeightHeap_free (heap) ;
	rm_free (heap) ;

	*S_out    = S ;
	*rank_out = rank_vec ;
}

