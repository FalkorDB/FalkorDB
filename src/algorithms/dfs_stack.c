
/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

/*
 * Finds all paths starting at given source node
 * We're computing one path at a time, this is done
 * to take advantage of scenarios where a query specifies LIMIT.
 * To implement this kind of iterative path finding using DFS
 * we're keeping track after:
 * 1. the last path computed, which we'll try to expand
 * 2. neighboring nodes discovered, each placed within a "level"
 * array containing all nodes discovered at a specific level.
 * */

#include "RG.h"
#include "dfs_stack.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "datatypes/path/path.h"
#include "graph/entities/graph_entity.h"

void dfs_stack_new (
	Graph_dfs_stack *stk,
	const Graph *g,
	const RelationID *rels,
	int n_rels,
	GRAPH_EDGE_DIR dir,
	int cap
) {
	ASSERT (stk != NULL) ;
	ASSERT (g != NULL) ;
	ASSERT (n_rels >= 0) ;
	ASSERT (rels != NULL || n_rels == 0) ;

	stk->levels    = arr_new (LevelIt, cap > 0 ? cap : 1) ;
	stk->rel_count = n_rels ;
	stk->rels      = rels ;

	int ndirs = (dir == GRAPH_EDGE_DIR_BOTH) ? 2 : 1 ;
	stk->it_n = n_rels * ndirs ;
	stk->it_arr = rm_calloc (stk->it_n, sizeof (TensorIterator)) ;

	int idx = 0 ;
	if (dir == GRAPH_EDGE_DIR_INCOMING || dir == GRAPH_EDGE_DIR_BOTH) {
		for (int r = 0; r < n_rels; r++, idx++) {
			Tensor T = Graph_GetRelationMatrix (g, rels [r], false) ;
			TensorIterator_Attach (&stk->it_arr [idx], T, true) ;
		}
	}
	if (dir == GRAPH_EDGE_DIR_OUTGOING || dir == GRAPH_EDGE_DIR_BOTH) {
		for (int r = 0; r < n_rels; r++, idx++) {
			Tensor T = Graph_GetRelationMatrix (g, rels [r], false) ;
			TensorIterator_Attach (&stk->it_arr [idx], T, false) ;
		}
	}

	stk->cached_path = Path_New (cap) ;
	stk->n_cached = 0 ;
}

void dfs_stack_push_neighbors (
	Graph_dfs_stack *stk,
	NodeID n
) {
	ASSERT (stk != NULL) ;
	if (stk->it_n == 0) {
		return ;
	}

	// copy an iterator that is already attached
	// WARNING: I copying internal GraphBLAS GB_Iterator structs, which
	// breaks opacity and is not recommended. However, iterator attachment
	// is very costly, and explicitly copying an iterator works as of now.
	LevelIt it = {.it = stk->it_arr[0], .src = n, .relnum = 0};
	TensorIterator_IterateRow (&it.it, it.src) ;
	arr_append(stk->levels, it);
}

static bool _levelit_next (
	Graph_dfs_stack *stk,
	GrB_Index *row,      // [optional out] source id
	GrB_Index *col,      // [optional out] dest id
	uint64_t *x,         // [optional out] edge id
	RelationID *rel,     // [optional out] rel id
	NodeID *nid          // [optional out] neighbor id
) {
	LevelIt *it = &arr_tail (stk->levels);
	while (it->relnum < stk->it_n) {
		GrB_Index _row, _col;
		uint64_t _x;
		GrB_Index *row_out = row ? row : &_row;
		GrB_Index *col_out = col ? col : &_col;
		uint64_t *x_out = x ? x : &_x;

		if (TensorIterator_next (&it->it, row_out, col_out, x_out, NULL)) {
			// keep current edge tuple on the level iterator so the stack can
			// be converted to a Path using iterator state.
			it->it.row = *row_out ;
			it->it.col = *col_out ;
			it->it.x   = *x_out ;

			if (rel) {
				ASSERT (stk->rel_count > 0) ;
				*rel = stk->rels [it->relnum % stk->rel_count] ;
			}
			if (nid) {
				ASSERT (*row_out == it->src || *col_out == it->src) ;
				*nid = *row_out ^ *col_out ^ it->src ;
			}
			return true;
		}
		// copy an iterator that is already attached
		// WARNING: I copying internal GraphBLAS GB_Iterator structs, which
		// breaks opacity and is not recommended. However, iterator attachment
		// is very costly, and explicitly copying an iterator works as of now.
		if (++it->relnum < stk->it_n) {
			it->it = stk->it_arr [it->relnum];
			TensorIterator_IterateRow (&it->it, it->src) ;
		}
	}

	return false;
}

bool dfs_stack_pop (
	Graph_dfs_stack *stk,
	NodeID *frontier,
	Edge *e
) {
	e->attributes = NULL ;
	while (arr_len(stk->levels) > 0) {
		if (_levelit_next (
			stk, &e->src_id, &e->dest_id, &e->id, &e->relationID, frontier))
		{
			stk->n_cached = MIN (stk->n_cached, arr_len(stk->levels) - 1);
			return true;
		}
		arr_pop (stk->levels) ;
	}
	stk->n_cached = 0;
	return false;
}

Path *dfs_stack_to_path (
	Graph_dfs_stack *stk,
	const Graph *g
) {
	uint depth = arr_len (stk->levels) ;
	Path *p = stk->cached_path ;
	if (depth == 0) {
		Path_Clear (p) ;
		return p;
	}
	Path_Truncate (p, stk->n_cached);
	for (uint i = stk->n_cached; i < depth; i++) {
		Node n = GE_NEW_NODE () ;

		Graph_GetNode (g, stk->levels [i].src, &n) ;
		Path_AppendNode (p, n) ;

		LevelIt *lvl = stk->levels + i ;
		ASSERT (stk->rel_count > 0) ;
		Edge e = {
			.id         = lvl->it.x,
			.src_id     = lvl->it.row,
			.dest_id    = lvl->it.col,
			.relationID = stk->rels [lvl->relnum % stk->rel_count],
			.attributes = NULL
		};

		Graph_GetEdge (g, lvl->it.x, &e) ;
		Path_AppendEdge (p, e) ;
	}
	stk->n_cached = depth;

	Node n = GE_NEW_NODE () ;
	LevelIt *lvl = &arr_tail (stk->levels);
	// the node that is NOT src
	NodeID nid = lvl->src ^ lvl->it.row ^ lvl->it.col;
	Graph_GetNode (g, nid, &n) ;
	Path_AppendNode (p, n) ;
	return p ;
}

// Checks if the stack contains the given edge (excludes the last edge)
bool dfs_stack_contains_edge
(
	const Graph_dfs_stack *stk,
	EdgeID eid
) {
	uint n = arr_len (stk->levels) ;
	for (uint i = 0; i < n - 1; i++) {
		if (stk->levels [i].it.x == eid) {
			return true;
		}
	}
	return false;
}

// Checks if the stack contains the given node (excludes the last node)
bool dfs_stack_contains_node
(
	const Graph_dfs_stack *stk,
	NodeID nid
) {
	uint n = arr_len (stk->levels) ;
	for (uint i = 0; i < n; i++) {
		if (stk->levels [i].src == nid) {
			return true;
		}
	}

	return false;
}

void dfs_stack_clear
(
	Graph_dfs_stack *stk
) {
	ASSERT (stk != NULL) ;
	arr_clear (stk->levels) ;
}

bool dfs_stack_empty
(
	const Graph_dfs_stack *stk
) {
	ASSERT (stk != NULL) ;
	return arr_len (stk->levels) == 0 ;
}

void dfs_stack_free (
	Graph_dfs_stack stk
) {
	arr_free (stk.levels) ;
	rm_free (stk.it_arr) ;
}
