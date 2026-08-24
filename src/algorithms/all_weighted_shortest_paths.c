/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "Dijkstra.h"
#include "all_weighted_shortest_paths.h"
#include "../value.h"
#include "../util/arr.h"

// get numeric attribute value of an entity otherwise return default value
static inline SIValue _get_value_or_default
(
	GraphEntity *ge,
	AttributeID id,
	SIValue default_value
) {
	SIValue v;

	if(!GraphEntity_GetProperty(ge, id, &v)) {
		return default_value;
	}

	if(SI_TYPE(v) & SI_NUMERIC) {
		return v;
	}

	return default_value;
}

// reverse a traversal direction (OUTGOING <-> INCOMING, BOTH unchanged), so a
// backward search from dst measures distance *to* dst in the original graph.
static inline GRAPH_EDGE_DIR _reverse_dir
(
	GRAPH_EDGE_DIR dir
) {
	if(dir == GRAPH_EDGE_DIR_OUTGOING) return GRAPH_EDGE_DIR_INCOMING;
	if(dir == GRAPH_EDGE_DIR_INCOMING) return GRAPH_EDGE_DIR_OUTGOING;
	return GRAPH_EDGE_DIR_BOTH;
}

// a node reached at some depth of the enumeration, and the edge that led to it
typedef struct {
	Node node;  // the node
	Edge edge;  // edge from its parent on the current path (unused at depth 0)
} DagFrontier;

uint AllWeightedShortestPaths
(
	Graph *g,
	NodeID src,
	NodeID dst,
	GRAPH_EDGE_DIR dir,
	RelationID *relationIDs,
	Tensor *relationMatrices,
	int relationCount,
	AttributeID weight_prop,
	Path ***paths,
	double *min_weight
) {
	*paths = arr_new(Path *, 0);

	// degenerate: a path needs at least one edge (the caller guards src == dst
	// and routes it through the exhaustive DFS, but stay safe here too).
	if(src == dst) {
		return 0;
	}

	// forward search: shortest-path weight from src to every reachable node.
	DijkstraCtx *fwd = DijkstraCtx_New(g, dir, relationIDs, relationMatrices,
			relationCount, weight_prop);
	DijkstraCtx_Run(fwd, src, INVALID_ENTITY_ID, NULL, NULL);

	double D;
	if(!DijkstraCtx_Distance(fwd, dst, &D)) {
		// dst unreachable from src: no paths.
		DijkstraCtx_Free(fwd);
		return 0;
	}

	// backward search: shortest-path weight from every node to dst, obtained by
	// searching from dst along reversed edges. used to prune enumeration to
	// nodes that can still reach dst.
	DijkstraCtx *bwd = DijkstraCtx_New(g, _reverse_dir(dir), relationIDs,
			relationMatrices, relationCount, weight_prop);
	DijkstraCtx_Run(bwd, dst, INVALID_ENTITY_ID, NULL, NULL);

	// expansion directions for the (forward) enumeration below
	GRAPH_EDGE_DIR dirs[2];
	int ndirs = 0;
	if(dir == GRAPH_EDGE_DIR_OUTGOING || dir == GRAPH_EDGE_DIR_BOTH) {
		dirs[ndirs++] = GRAPH_EDGE_DIR_OUTGOING;
	}
	if(dir == GRAPH_EDGE_DIR_INCOMING || dir == GRAPH_EDGE_DIR_BOTH) {
		dirs[ndirs++] = GRAPH_EDGE_DIR_INCOMING;
	}

	// iterative DFS over the shortest-path DAG. levels[d] holds the not-yet-
	// explored frontier candidates for depth d; 'cur' is the path being built.
	DagFrontier **levels = arr_new(DagFrontier *, 1);
	arr_append(levels, arr_new(DagFrontier, 1));

	Node srcNode = GE_NEW_NODE();
	Graph_GetNode(g, src, &srcNode);
	DagFrontier srcFrontier = { .node = srcNode };
	arr_append(levels[0], srcFrontier);

	Path *cur       = Path_New(8);
	Edge *neighbors = arr_new(Edge, 32);

	while(Path_NodeCount(cur) > 0 || arr_len(levels[0]) > 0) {
		uint depth = Path_NodeCount(cur);

		// can we advance at this depth?
		if(depth < arr_len(levels) && arr_len(levels[depth]) > 0) {
			DagFrontier f = arr_pop(levels[depth]);
			NodeID uid = ENTITY_GET_ID(&f.node);

			// cycle guard: with strictly positive weights the DAG is acyclic
			// and this never fires; it covers zero-weight-edge cycles.
			if(Path_ContainsNode(cur, uid)) {
				continue;
			}

			Path_AppendNode(cur, f.node);
			if(depth > 0) {
				Path_AppendEdge(cur, f.edge);
			}

			if(uid == dst) {
				// reached dst: emit a copy. dst is a sink in the enumeration,
				// so don't expand it -- the next iteration finds an empty
				// level here and backtracks.
				arr_append(*paths, Path_Clone(cur));
				continue;
			}

			// expand this node's tight, dst-reaching outgoing DAG edges into
			// the next level.
			double d_src_u;
			DijkstraCtx_Distance(fwd, uid, &d_src_u);

			// make sure a level array exists for depth+1
			while(arr_len(levels) <= depth + 1) {
				arr_append(levels, arr_new(DagFrontier, 4));
			}

			for(int d = 0; d < ndirs; d++) {
				for(int r = 0; r < relationCount; r++) {
					Graph_GetNodeEdgesFromMatrix(g, &f.node, dirs[d],
							relationMatrices[r], relationIDs[r], &neighbors);
				}

				uint32_t n = arr_len(neighbors);
				for(uint32_t j = 0; j < n; j++) {
					Edge *e = neighbors + j;
					NodeID vid = (dirs[d] == GRAPH_EDGE_DIR_OUTGOING)
						? Edge_GetDestNodeID(e)
						: Edge_GetSrcNodeID(e);

					if(vid == uid) {
						continue;  // ignore self-loops
					}

					// v must be able to reach dst (discovered by backward run)
					if(!DijkstraCtx_Distance(bwd, vid, NULL)) {
						continue;
					}

					// (u->v) must be tight: d_src(u)+w == d_src(v). exact for
					// integer weights (see header note). a strictly-greater sum
					// means this edge isn't on any shortest path to v.
					double d_src_v;
					DijkstraCtx_Distance(fwd, vid, &d_src_v);

					SIValue w = _get_value_or_default((GraphEntity *)e,
							weight_prop, SI_LongVal(1));
					if(d_src_u + SI_GET_NUMERIC(w) != d_src_v) {
						continue;
					}

					Node vnode = GE_NEW_NODE();
					Graph_GetNode(g, vid, &vnode);
					DagFrontier nf = { .node = vnode, .edge = *e };
					arr_append(levels[depth + 1], nf);
				}

				arr_clear(neighbors);
			}
		} else {
			// no way forward from the current path: backtrack one step.
			Path_PopNode(cur);
			if(Path_EdgeCount(cur) > 0) {
				Path_PopEdge(cur);
			}
		}
	}

	*min_weight = D;

	// cleanup
	uint levelsCount = arr_len(levels);
	for(uint i = 0; i < levelsCount; i++) {
		arr_free(levels[i]);
	}
	arr_free(levels);
	arr_free(neighbors);
	Path_Free(cur);
	DijkstraCtx_Free(fwd);
	DijkstraCtx_Free(bwd);

	return arr_len(*paths);
}
