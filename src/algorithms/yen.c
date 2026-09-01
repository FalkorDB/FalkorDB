/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "yen.h"
#include "Dijkstra.h"
#include "../value.h"
#include "../util/arr.h"
#include "../util/heap.h"
#include "../util/dict.h"
#include "../util/rmalloc.h"

// a Yen candidate path waiting in the candidate heap
typedef struct {
	Path   *path;    // the candidate path
	double  weight;  // its total weight
} YenCandidate;

// candidate-heap comparator: the heap keeps the *greatest* element (per this
// cmp) on top, so invert the natural order -- smallest weight (then shortest
// path) becomes the top, i.e. the next path to accept.
static int _yen_cand_cmp
(
	const void *a,
	const void *b,
	void *udata
) {
	const YenCandidate *ca = (const YenCandidate *) a;
	const YenCandidate *cb = (const YenCandidate *) b;

	if(ca->weight != cb->weight) {
		return (ca->weight < cb->weight) ? 1 : -1;
	}

	size_t la = Path_Len (ca->path);
	size_t lb = Path_Len (cb->path);
	if(la != lb) {
		return (la < lb) ? 1 : -1;
	}

	return 0;
}

// 64-bit FNV-1a hash of a path's edge-id sequence -- its identity for dedup.
//
// deliberately not SIPath_HashCode: that hashes node+edge SIValues and would
// require wrapping each candidate Path in an SIValue (an allocation/ownership
// dance) per lookup. the edge-id sequence is exactly the identity Yen needs --
// it distinguishes parallel edges between the same node pair -- and hashing it
// is a few multiplies with no allocation.
static uint64_t _path_key
(
	const Path *p
) {
	uint64_t h = 1469598103934665603ULL;  // FNV offset basis

	uint ec = Path_EdgeCount (p);
	for(uint i = 0; i < ec; i++) {
		uint64_t e = ENTITY_GET_ID (Path_GetEdge (p, i));
		h ^= e;
		h *= 1099511628211ULL;  // FNV prime
	}

	return h;
}

// record path 'p' as seen; returns true if it was new, false if already known.
static bool _mark_seen
(
	dict *seen,
	const Path *p
) {
	uint64_t h = _path_key(p);

	if(HashTableFind(seen, (void *)(uintptr_t)h) != NULL) {
		return false;
	}

	HashTableAdd(seen, (void *)(uintptr_t)h, (void *)(uintptr_t)1);
	return true;
}

// does 'p' share prev's root prefix -- its first 'i' edges -- and have an edge
// at index 'i' to block? (for i == 0 every path trivially shares the empty
// prefix, so any path with a first edge qualifies.)
static bool _shares_root
(
	const Path *p,
	const Path *prev,
	uint i
) {
	if(Path_EdgeCount (p) <= i) {
		return false;  // no edge at index i to block
	}

	for(uint j = 0; j < i; j++) {
		if(ENTITY_GET_ID (Path_GetEdge (p, j)) !=
			ENTITY_GET_ID (Path_GetEdge (prev, j))) {
			return false;
		}
	}

	return true;
}

// total weight of prev's first 'i' edges (the root path up to the spur node)
static double _root_weight
(
	const Path *prev,
	uint i,
	AttributeID weight_prop
) {
	double w = 0.0;

	for(uint j = 0; j < i; j++) {
		SIValue v = GraphEntity_GetNumericPropertyOrDefault(
				(GraphEntity *) Path_GetEdge (prev, j),
				weight_prop, SI_LongVal(1));
		w += SI_GET_NUMERIC(v);
	}

	return w;
}

// build root(prev, i) ++ spur: prev's first i edges/i+1 nodes, then the spur
// path (whose first node is prev.nodes[i], already included, so it is skipped).
static Path *_concat
(
	const Path *prev,
	uint i,
	const Path *spur
) {
	uint spur_nodes = Path_NodeCount(spur);
	uint spur_edges = Path_EdgeCount(spur);
	Path *total = Path_New(i + 1 + spur_nodes);

	// root nodes [0..i] and root edges [0..i-1]
	for(uint j = 0; j <= i; j++) {
		Path_AppendNode(total, *Path_GetNode(prev, j));
	}
	for(uint j = 0; j < i; j++) {
		Path_AppendEdge(total, *Path_GetEdge(prev, j));
	}

	// spur nodes [1..end] (node 0 is the spur node, already appended) and all
	// spur edges
	for(uint j = 1; j < spur_nodes; j++) {
		Path_AppendNode(total, *Path_GetNode(spur, j));
	}
	for(uint j = 0; j < spur_edges; j++) {
		Path_AppendEdge(total, *Path_GetEdge(spur, j));
	}

	return total;
}

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
) {
	ASSERT(g       != NULL);
	ASSERT(paths   != NULL);
	ASSERT(weights != NULL);

	// A: accepted paths (ascending weight); AW: their weights (parallel)
	Path   **A  = arr_new (Path *, 0);
	double  *AW = arr_new (double, 0);

	*paths   = A;
	*weights = AW;

	// nothing requested, or degenerate src == dst (caller guards, but be safe)
	if(k == 0 || src == dst) {
		return 0;
	}

	DijkstraCtx *dc = DijkstraCtx_New (g, dir, relationIDs, relationMatrices,
			relationCount, weight_prop);

	// A[0]: the global shortest path. if dst is unreachable there are none.
	if(!DijkstraCtx_Run (dc, src, dst, NULL, NULL)) {
		DijkstraCtx_Free (dc);
		return 0;
	}

	Path  *p0 = DijkstraCtx_Path(dc, src, dst);
	double w0;
	DijkstraCtx_Distance(dc, dst, &w0);
	arr_append(A, p0);
	arr_append(AW, w0);

	// B: candidate min-heap; seen: dedup set of path keys (covers A and B).
	heap_t *B    = Heap_new (_yen_cand_cmp, NULL);
	dict   *seen = HashTableCreate (&def_dt);
	_mark_seen (seen, p0);

	// blocked node/edge sets, reused (emptied) across spur searches.
	dict *blocked_nodes = HashTableCreate (&def_dt);
	dict *blocked_edges = HashTableCreate (&def_dt);

	while (arr_len (A) < k) {
		Path *prev       = arr_tail (A);
		uint  prev_nodes = Path_NodeCount (prev);

		// spur node ranges over prev's nodes [0 .. prev_nodes-2] (its last node
		// is dst, which can't spur).
		for(uint i = 0; i + 1 < prev_nodes; i++) {
			NodeID spur = ENTITY_GET_ID (Path_GetNode (prev, i));

			// block the edge leaving the spur node in every already-found path
			// that shares prev's root prefix -- so the spur search can't
			// re-derive one of them.
			HashTableEmpty (blocked_edges, NULL);
			for(uint a = 0; a < arr_len(A); a++) {
				if(_shares_root(A[a], prev, i)) {
					EdgeID be = ENTITY_GET_ID (Path_GetEdge (A[a], i));
					HashTableAdd (blocked_edges, (void *)(uintptr_t)be,
							(void *)(uintptr_t)1);
				}
			}

			// block the root nodes strictly before the spur node, so the spur
			// path can't loop back through the root (keeps the total loopless).
			HashTableEmpty (blocked_nodes, NULL);
			for(uint j = 0; j < i; j++) {
				NodeID bn = ENTITY_GET_ID (Path_GetNode (prev, j));
				HashTableAdd(blocked_nodes, (void *)(uintptr_t)bn,
						(void *)(uintptr_t)1);
			}

			// search spurNode -> dst on the graph minus the blocked sets.
			if(!DijkstraCtx_Run (dc, spur, dst, blocked_nodes, blocked_edges)) {
				continue;  // no spur path from here
			}

			Path  *spur_path = DijkstraCtx_Path (dc, spur, dst);
			double spur_w ;
			DijkstraCtx_Distance (dc, dst, &spur_w);

			// candidate = root(prev, i) ++ spur_path
			Path   *total   = _concat (prev, i, spur_path);
			double  total_w = _root_weight (prev, i, weight_prop) + spur_w;
			Path_Free (spur_path);

			if (_mark_seen (seen, total)) {
				YenCandidate *c = rm_malloc (sizeof(YenCandidate));
				c->path   = total;
				c->weight = total_w;
				Heap_offer(&B, c);
			} else {
				Path_Free(total);  // already generated before
			}
		}

		// accept the lightest candidate; if none remain, we're done.
		YenCandidate *best = Heap_poll(B);
		if(best == NULL) {
			break;
		}

		arr_append(A, best->path);
		arr_append(AW, best->weight);
		rm_free(best);
	}

	// drain and free any candidates left unaccepted.
	YenCandidate *c;
	while((c = Heap_poll(B)) != NULL) {
		Path_Free(c->path);
		rm_free(c);
	}

	Heap_free(B);
	HashTableRelease(seen);
	HashTableRelease(blocked_nodes);
	HashTableRelease(blocked_edges);
	DijkstraCtx_Free(dc);

	*paths   = A;
	*weights = AW;
	return arr_len(A);
}
