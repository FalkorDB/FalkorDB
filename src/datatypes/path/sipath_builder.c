/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "sipath_builder.h"
#include "../../RG.h"
#include "../../util/rmalloc.h"

/**
 * @brief  Reverse the direction of an edge.
 * @note   A new copy of the edge is created, in order not to invalidate any data in the original edge
 *         in case it required elsewhere.
 * @param  *e: Pointer to the original edge.
 * @retval New edge with inverse direction.
 */
static Edge _Edge_ReverseDirection(Edge *e) {
	Edge edge = *e;
	edge.src_id = e->dest_id;
	edge.dest_id = e->src_id;
	return edge;
}

static void _SIPath_Reverse(SIValue p) {
	Path *path = (Path *) p.ptrval;
	Path_Reverse(path);
}

SIValue SIPathBuilder_New(uint entity_count) {
	SIValue path;
	path.ptrval = Path_New(entity_count / 2);
	path.type = T_PATH;
	path.allocation = M_SELF;
	return path;
}

void SIPathBuilder_AppendNode(SIValue p, SIValue n) {
	Path *path = (Path *) p.ptrval;
	Node *node = (Node *) n.ptrval;
	Path_AppendNode(path, *node);
}

void SIPathBuilder_AppendEdge(SIValue p, SIValue e, bool RTLEdge) {
	Path *path = (Path *) p.ptrval;
	Edge *edge = (Edge *) e.ptrval;
	ASSERT(Path_NodeCount(path) > 0);
	// The edge should connect nodes[edge_count] to nodes[edge_count+1]
	uint edge_count = Path_EdgeCount(path);
	Node *n = Path_GetNode(path, edge_count);
	EntityID nId = ENTITY_GET_ID(n);
	// Validate source node is in the right place.
	ASSERT(nId == edge->src_id || nId == edge->dest_id);
	/* Reverse direction if needed. A direction change is needed if the last node in the path, reading
	 * RTL is the source node in the edge, and the edge direction in the query is LTR.
	 * path =[(a)]
	 * e = (a)->(b)
	 * Query: MATCH p=(a)<-[]-(b)
	 * e direction needs to be change. */

	Edge edge_to_append = (RTLEdge && nId == edge->src_id) ? _Edge_ReverseDirection(edge) : *edge;
	Path_AppendEdge(path, edge_to_append);
}

void SIPathBuilder_AppendPath(SIValue path, SIValue new_path, bool RTLEdge) {
	uint path_node_count = SIPath_NodeCount(path);
	ASSERT(path_node_count > 0);
	// No need to append empty paths.
	uint new_path_node_count = SIPath_NodeCount(new_path);

	if(new_path_node_count <= 1) return;

	SIValue last_LTR_node = SIPath_GetNode(path, path_node_count - 1);
	EntityID last_LTR_node_id = ENTITY_GET_ID((Node *)last_LTR_node.ptrval);
	SIValue new_path_node_0 = SIPath_Head(new_path);
	EntityID new_path_node_0_id = ENTITY_GET_ID((Node *)new_path_node_0.ptrval);
	SIValue new_path_last_node = SIPath_Last(new_path);
	EntityID new_path_last_node_id = ENTITY_GET_ID((Node *)new_path_last_node.ptrval);
	// Validate current last LTR node is in either edges of the path.
	ASSERT(last_LTR_node_id == new_path_node_0_id || last_LTR_node_id == new_path_last_node_id);
	int new_path_edge_count = SIPath_Length(new_path);

	// Check if path needs to be rverse inserated or not.
	if(last_LTR_node_id == new_path_last_node_id) _SIPath_Reverse(new_path);

	for(uint i = 0; i < new_path_edge_count - 1; i++) {
		SIPathBuilder_AppendEdge(path, SIPath_GetRelationship(new_path, i), RTLEdge);
		// Insert only nodes which are not the last and the first, since they will be added by append node specifically.
		SIPathBuilder_AppendNode(path, SIPath_GetNode(new_path, i + 1));

	}
	SIPathBuilder_AppendEdge(path, SIPath_GetRelationship(new_path, new_path_edge_count - 1), RTLEdge);
}
