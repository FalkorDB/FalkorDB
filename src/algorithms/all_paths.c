/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "all_paths.h"
#include "all_shortest_paths.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"

// make sure context level array have 'cap' available entries
static void _AllPathsCtx_EnsureLevelArrayCap
(
	AllPathsCtx *ctx,
	uint level,
	uint cap
) {
	if (cap == 0) {
		return ;
	}

	uint len = arr_len (ctx->levels) ;

	if (level < len) {
		LevelConnection *current = ctx->levels [level] ;
		ctx->levels [level] = arr_ensure_cap (current, arr_len (current) + cap) ;
		return ;
	}

	ASSERT (level == len) ;
	arr_append (ctx->levels, arr_new (LevelConnection, cap)) ;
}

// append given 'node' to given 'level' array
static void _AllPathsCtx_AddConnectionToLevel
(
	AllPathsCtx *ctx,
	uint level,
	Node *node,
	Edge *edge
) {
	ASSERT (level < arr_len (ctx->levels)) ;

	LevelConnection connection ;
	connection.node = *node ;

	if (edge) {
		connection.edge = *edge ;
	}

	arr_append (ctx->levels [level], connection) ;
}

// check to see if context levels array has entries at position 'level'
static bool _AllPathsCtx_LevelNotEmpty
(
	const AllPathsCtx *ctx,
	uint level
) {
	return (level < arr_len (ctx->levels) && arr_len (ctx->levels [level]) > 0) ;
}

// finalize a discovered edge (optionally hydrating it from the graph) and
// stage it in ctx->neighbors for filtering / expansion
static inline void _AllPathsCtx_AppendEdge
(
	AllPathsCtx *ctx,
	Edge e
) {
	if (ctx->fetch_edges) {
		bool res = Graph_GetEdge (ctx->g, e.id, &e) ;
		ASSERT (res == true) ;
	}

	arr_append (ctx->neighbors, e) ;
}

// collect every edge connecting 'id' to a neighbor in the requested
// direction, appending discovered edges to ctx->neighbors
//
// two matrix representations are handled per relation type:
// - a plain Delta_Matrix, when the relation never has parallel edges between
//   the same pair of nodes, its single UINT64 entry stores the edge id
// - a Tensor, once a relation has parallel edges, entries fan out to more
//   than one id, so a dedicated TensorIterator is required
//
// there's no transposed tensor, incoming traversal over a Tensor is done via
// a reversed scan instead of a transpose lookup
static void _AllPathsCtx_CollectEdges
(
	AllPathsCtx *ctx,
	NodeID id,
	bool incoming
) {
	for (int i = 0; i < ctx->relationCount; i++) {
		Edge e = {.relationID = ctx->relationIDs [i]} ;
		if (incoming) e.dest_id = id ; else e.src_id = id ;

		if (!ctx->multi_edge [i]) {
			Delta_MatrixTupleIter dmi ;

			if (incoming) {
				// the matrix itself holds the edge id, its transpose is
				// pattern-only, so the id is recovered with a follow-up
				// lookup on the original matrix
				Delta_Matrix TT = Delta_Matrix_getTranspose (ctx->matrices [i]) ;
				Delta_MatrixTupleIter_AttachRange (&dmi, TT, id, id) ;

				GrB_Index src_j ;
				while (Delta_MatrixTupleIter_next_BOOL (&dmi, NULL, &src_j,
							NULL) == GrB_SUCCESS) {
					e.src_id = src_j ;
					Delta_Matrix_extractElement_UINT64 (&e.id,
							ctx->matrices [i], src_j, id) ;
					_AllPathsCtx_AppendEdge (ctx, e) ;
				}
			} else {
				Delta_MatrixTupleIter_AttachRange (&dmi, ctx->matrices [i],
						id, id) ;

				while (Delta_MatrixTupleIter_next_UINT64 (&dmi, NULL,
							&e.dest_id, &e.id) == GrB_SUCCESS) {
					_AllPathsCtx_AppendEdge (ctx, e) ;
				}
			}

			Delta_MatrixTupleIter_detach (&dmi) ;
		} else {
			TensorIterator it ;
			TensorIterator_ScanRange (&it, ctx->matrices [i], id, id, incoming) ;

			if (incoming) {
				while (TensorIterator_next (&it, &e.src_id, NULL, &e.id, NULL)) {
					_AllPathsCtx_AppendEdge (ctx, e) ;
				}
			} else {
				while (TensorIterator_next (&it, NULL, &e.dest_id, &e.id, NULL)) {
					_AllPathsCtx_AppendEdge (ctx, e) ;
				}
			}
		}
	}
}

// drop every edge in ctx->neighbors that fails the traversal filter tree
static void _AllPathsCtx_FilterEdges
(
	AllPathsCtx *ctx
) {
	if (!ctx->ft) {
		return ;
	}

	uint32_t n = arr_len (ctx->neighbors) ;
	for (uint32_t i = 0; i < n; i++) {
		// update the record with the current edge
		Record_AddEdge (ctx->r, ctx->edge_idx, ctx->neighbors [i]) ;

		// drop the edge if it doesn't pass the filter
		if (FilterTree_applyFilters (ctx->ft, ctx->r) != FILTER_PASS) {
			arr_del_fast (ctx->neighbors, i) ;
			i-- ;
			n-- ;
		}
	}
}

// resolve the unvisited endpoint of each edge left in ctx->neighbors and
// place it, alongside the edge leading to it, in the next level
static void _AllPathsCtx_ExpandToLevel
(
	AllPathsCtx *ctx,
	uint32_t depth,
	bool incoming
) {
	uint32_t n = arr_len (ctx->neighbors) ;
	_AllPathsCtx_EnsureLevelArrayCap (ctx, depth, n) ;

	for (uint32_t i = 0; i < n; i++) {
		Edge *e = ctx->neighbors + i ;
		NodeID neighborID = incoming ? Edge_GetSrcNodeID (e)
									  : Edge_GetDestNodeID (e) ;

		Node neighbor = GE_NEW_NODE () ;
		Graph_GetNode (ctx->g, neighborID, &neighbor) ;

		_AllPathsCtx_AddConnectionToLevel (ctx, depth, &neighbor, e) ;
	}

	arr_clear (ctx->neighbors) ;
}

// traverse the frontier node's outgoing edges and add all encountered
// unvisited neighbors to 'depth'
static void addOutgoingNeighbors
(
	AllPathsCtx *ctx,
	LevelConnection *frontier,
	uint32_t depth
) {
	NodeID src_id = ENTITY_GET_ID (&frontier->node) ;

	_AllPathsCtx_CollectEdges  (ctx, src_id, false) ;
	_AllPathsCtx_FilterEdges   (ctx) ;
	_AllPathsCtx_ExpandToLevel (ctx, depth, false) ;
}

// traverse the frontier node's incoming edges and add all encountered
// unvisited neighbors to 'depth'
static void addIncomingNeighbors
(
	AllPathsCtx *ctx,
	LevelConnection *frontier,
	uint32_t depth
) {
	NodeID dest_id = ENTITY_GET_ID (&frontier->node) ;

	_AllPathsCtx_CollectEdges  (ctx, dest_id, true) ;
	_AllPathsCtx_FilterEdges   (ctx) ;
	_AllPathsCtx_ExpandToLevel (ctx, depth, true) ;
}

// traverse from the frontier node in the specified direction and add all
// encountered nodes and edges
void addNeighbors
(
	AllPathsCtx *ctx,
	LevelConnection *frontier,
	uint32_t depth,
	GRAPH_EDGE_DIR dir
) {
	switch (dir) {
		case GRAPH_EDGE_DIR_OUTGOING:
			addOutgoingNeighbors (ctx, frontier, depth) ;
			break ;
		case GRAPH_EDGE_DIR_INCOMING:
			addIncomingNeighbors (ctx, frontier, depth) ;
			break ;
		case GRAPH_EDGE_DIR_BOTH:
			addIncomingNeighbors (ctx, frontier, depth) ;
			addOutgoingNeighbors (ctx, frontier, depth) ;
			break ;
		default:
			ASSERT (false && "encountered unexpected traversal direction in AllPaths") ;
			break ;
	}
}

AllPathsCtx *AllPathsCtx_New
(
	Node *src,
	Node *dst,
	Graph *g,
	RelationID *relationIDs,
	int relationCount,
	GRAPH_EDGE_DIR dir,
	uint minLen,
	uint maxLen,
	Record r,
	FT_FilterNode *ft,
	int edge_idx,
	bool shortest_paths
) {
	ASSERT (src != NULL) ;

	AllPathsCtx *ctx = rm_malloc (sizeof (AllPathsCtx)) ;
	ctx->g           = g ;
	ctx->r           = r ;
	ctx->ft          = ft ;
	ctx->dir         = dir ;
	ctx->edge_idx    = edge_idx ;
	ctx->fetch_edges = (edge_idx >= 0) ;

	// Cypher variable path "[:*min..max]"" specifies edge count
	// While the path constructed here contains only nodes.
	// As such a path which require min..max edges
	// should contain min+1..max+1 nodes.
	ctx->minLen = minLen + 1 ;
	ctx->maxLen = maxLen + 1 ;

	// take ownership of relation IDs; expand GRAPH_NO_RELATION to all actual types
	if (relationCount == 1 && relationIDs [0] == GRAPH_NO_RELATION) {
		ctx->relationCount = Graph_RelationTypeCount (g) ;
		ctx->relationIDs   = rm_malloc (sizeof (RelationID) * ctx->relationCount) ;
		for (int i = 0; i < ctx->relationCount; i++) ctx->relationIDs [i] = i ;
	} else {
		ctx->relationCount = relationCount ;
		ctx->relationIDs   = rm_malloc (sizeof (RelationID) * relationCount) ;
		for (int i = 0; i < relationCount; i++) ctx->relationIDs [i] = relationIDs [i] ;
	}

	// cache relation matrices to avoid repeated Graph_GetRelationMatrix calls per expansion
	ctx->matrices   = rm_malloc (sizeof (Tensor) * ctx->relationCount) ;
	ctx->multi_edge = rm_malloc (sizeof (bool)   * ctx->relationCount) ;
	for (int i = 0; i < ctx->relationCount; i++) {
		ctx->matrices [i]   = Graph_GetRelationMatrix (g, ctx->relationIDs [i], false) ;
		ctx->multi_edge [i] = Graph_RelationshipContainsMultiEdge (g, ctx->relationIDs [i]) ;
	}

	ctx->levels         = arr_new (LevelConnection *, 1) ;
	ctx->path           = Path_New (1) ;
	ctx->neighbors      = arr_new (Edge, 32) ;
	ctx->dst            = dst ;
	ctx->shortest_paths = shortest_paths ;
	ctx->visited        = NULL ;

	_AllPathsCtx_EnsureLevelArrayCap  (ctx, 0, 1) ;
	_AllPathsCtx_AddConnectionToLevel (ctx, 0, src, NULL) ;

	if (ctx->shortest_paths) {
		if (dst == NULL) {
			// If the destination is NULL due to a scenario like a
			// failed optional match, no results will be produced
			ctx->maxLen = 0 ;
			return ctx ;
		}

		// get the the minimum length between src and dst
		// then start the traversal from dst to src
		int min_path_len = AllShortestPaths_FindMinimumLength (ctx, src, dst) ;
		ctx->minLen = min_path_len ;
		ctx->maxLen = min_path_len ;
		ctx->dst    = src ;

		if (dir == GRAPH_EDGE_DIR_INCOMING) {
			ctx->dir = GRAPH_EDGE_DIR_OUTGOING ;
		} else if (dir == GRAPH_EDGE_DIR_OUTGOING) {
			ctx->dir = GRAPH_EDGE_DIR_INCOMING ;
		}
		_AllPathsCtx_AddConnectionToLevel (ctx, 0, dst, NULL) ;
	}

	// in case we have filter tree validate that we can access the filtered edge
	ASSERT (!ctx->ft || (ctx->edge_idx >= 0 && (uint)ctx->edge_idx < Record_length (ctx->r))) ;
	return ctx ;
}

static Path *_AllPathsCtx_NextPath
(
	AllPathsCtx *ctx
) {
	// as long as path is not empty OR there are neighbors to traverse
	while (Path_NodeCount (ctx->path) > 0 ||
		   _AllPathsCtx_LevelNotEmpty (ctx, 0)) {

		uint32_t depth = Path_NodeCount (ctx->path) ;

		// can we advance?
		if (_AllPathsCtx_LevelNotEmpty (ctx, depth)) {
			// get a new frontier
			LevelConnection frontierConnection = arr_pop (ctx->levels [depth]);
			Edge frontierEdge = frontierConnection.edge ;

			// path is not valid if the next edge is a duplicate of the edges
			// that came before it
			if (depth > 1 &&
				Path_ContainsEdge (ctx->path, ENTITY_GET_ID (&frontierEdge))) {
				continue ;
			}

			// add frontier to path
			Path_AppendNode (ctx->path, frontierConnection.node) ;

			// if depth is 0 this is the source node
			// there is no leading edge to it
			// for depth > 0 for each frontier node, there is a leading edge
			if (depth > 0) {
				Path_AppendEdge (ctx->path, frontierEdge) ;
			}

			// update path depth
			depth++ ;

			// introduce neighbors only if path depth < maximum path length
			if (depth < ctx->maxLen) {
				addNeighbors (ctx, &frontierConnection, depth, ctx->dir) ;
			}

			// see if we can return path
			// TODO: note that further calls to this function will continue to
			// operate on this path, so it is essential that the caller does not
			// modify it (or creates a copy beforehand)
			// if future features like an algorithm API use this routine  they
			// should either be responsible for memory safety or a memory-safe
			// boolean/routine should be offered
			if (depth >= ctx->minLen && depth <= ctx->maxLen) {

				// looking for a specific destination node?
				if (ctx->dst != NULL) {
					Node dst = Path_Head (ctx->path) ;
					if (ENTITY_GET_ID (ctx->dst) != ENTITY_GET_ID (&dst)) {
						continue ;
					}
				}

				return ctx->path ;
			}
		} else {
			// no way to advance, backtrack
			Path_PopNode (ctx->path) ;
			if (Path_EdgeCount (ctx->path)) {
				Path_PopEdge (ctx->path) ;
			}
		}
	}

	// couldn't find a path
	return NULL ;
}

Path *AllPathsCtx_NextPath
(
	AllPathsCtx *ctx
) {
	if (!ctx || ctx->maxLen == 0) {
		return NULL ;
	}

	if (ctx->shortest_paths) {
		return AllShortestPaths_NextPath (ctx) ;
	}
	else {
		return _AllPathsCtx_NextPath (ctx) ;
	}
}

void AllPathsCtx_Reset
(
	AllPathsCtx *ctx,
	Node *src,
	Node *dst,
	Record r
) {
	ASSERT (ctx != NULL) ;
	ASSERT (src != NULL) ;
	ASSERT (!ctx->shortest_paths) ;

	ctx->r   = r ;
	ctx->dst = dst ;

	Path_Clear (ctx->path) ;
	arr_clear  (ctx->neighbors) ;

	uint levelsCount = arr_len (ctx->levels) ;
	for (uint i = 0; i < levelsCount; i++) {
		arr_clear (ctx->levels[i]) ;
	}

	_AllPathsCtx_EnsureLevelArrayCap  (ctx, 0, 1) ;
	_AllPathsCtx_AddConnectionToLevel (ctx, 0, src, NULL) ;
}

void AllPathsCtx_Free (AllPathsCtx *ctx) {
	if (!ctx) {
		return ;
	}

	uint32_t levelsCount = arr_len (ctx->levels) ;
	for (int i = 0; i < levelsCount; i++) {
		arr_free (ctx->levels [i]) ;
	}

	Path_Free (ctx->path)        ;
	arr_free  (ctx->levels)      ;
	rm_free   (ctx->matrices)    ;
	arr_free  (ctx->neighbors)   ;
	rm_free   (ctx->multi_edge)  ;
	rm_free   (ctx->relationIDs) ;

	if (ctx->visited) {
		GrB_Vector_free (&ctx->visited) ;
	}
	rm_free (ctx) ;
	ctx = NULL ;
}
