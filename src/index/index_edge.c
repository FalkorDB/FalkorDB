/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "index.h"
#include "../query_ctx.h"
#include "../graph/graphcontext.h"
#include "../graph/entities/edge.h"

extern RSDoc *Index_IndexGraphEntity(Index idx,const GraphEntity *e,
		const void *key, size_t key_len, uint *doc_field_count);

// index an edge into the given index
// when 'unsafe' is true the document is added without acquiring RediSearch's
// global write lock; this is only valid during parallel population of disjoint,
// not-yet-visible indices (RDB decode) where the caller guarantees single-writer
// access to the spec
static void _index_edge
(
	Index idx,
	const Edge *e,
	bool unsafe
) {
	ASSERT(e   != NULL);
	ASSERT(idx != NULL);

	RSDoc    *doc    = NULL;
	RSIndex  *rsIdx  = Index_RSIndex(idx);
	EntityID src_id  = Edge_GetSrcNodeID(e);
	EntityID dest_id = Edge_GetDestNodeID(e);
	EntityID edge_id = ENTITY_GET_ID(e);

	EdgeIndexKey key = {.src_id = src_id, .dest_id = dest_id, .edge_id = edge_id};
	size_t key_len = sizeof(EdgeIndexKey);

	uint doc_field_count = 0;
	doc = Index_IndexGraphEntity(idx, (const GraphEntity *)e,
			(const void *)&key, key_len, &doc_field_count);

	if(doc_field_count == 0) {
		// entity doesn't possess any attributes which are indexed
		// during fresh (unsafe) population the index is empty so there is
		// nothing to remove; avoid the lock-taking delete in that path
		if(!unsafe) {
			Index_RemoveEdge(idx, e);
		}
		RediSearch_FreeDocument(doc);
		return;
	}

	// add document to active RediSearch index
	// add src_node and dest_node fields
	RediSearch_DocumentAddFieldNumber(doc, "range:_src_id",
			src_id, RSFLDTYPE_NUMERIC);
	RediSearch_DocumentAddFieldNumber(doc, "range:_dest_id",
			dest_id, RSFLDTYPE_NUMERIC);
	int res = unsafe
		? RediSearch_SpecAddDocumentUnsafe(rsIdx, doc)
		: RediSearch_SpecAddDocument(rsIdx, doc);
	ASSERT(res == REDISMODULE_OK);
}

void Index_IndexEdge
(
	Index idx,
	const Edge *e
) {
	_index_edge(idx, e, false);
}

void Index_IndexEdgeUnsafe
(
	Index idx,
	const Edge *e
) {
	_index_edge(idx, e, true);
}

void Index_RemoveEdge
(
	Index idx,     // index to update
	const Edge *e  // edge to remove from index
) {
	ASSERT(e   != NULL);
	ASSERT(idx != NULL);

	RSIndex  *rsIdx  = Index_RSIndex(idx);
	EntityID src_id  = Edge_GetSrcNodeID(e);
	EntityID dest_id = Edge_GetDestNodeID(e);
	EntityID edge_id = ENTITY_GET_ID(e);

	EdgeIndexKey key = {.src_id = src_id, .dest_id = dest_id, .edge_id = edge_id};
	size_t key_len = sizeof(EdgeIndexKey);
	RediSearch_DeleteDocument(rsIdx, &key, key_len);
}

