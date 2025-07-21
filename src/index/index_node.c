/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "RG.h"
#include "index.h"
#include "../value.h"
#include "../query_ctx.h"
#include "../graph/graphcontext.h"
#include "../graph/rg_matrix/rg_matrix_iter.h"

extern RSDoc *Index_IndexGraphEntity(Index idx, const GraphEntity *e,
		const void *key, size_t key_len, uint *doc_field_count);

void Index_IndexNode
(
	Index idx,
	const Node *n
) {
	ASSERT(n    !=  NULL);
	ASSERT(idx  !=  NULL);

	EntityID key             = ENTITY_GET_ID(n);
	RSDoc    *doc            = NULL;
	RSIndex  *rsIdx          = Index_RSIndex(idx);
	size_t   key_len         = sizeof(EntityID);
	uint     doc_field_count = 0;

	// create RediSearch document representing node
	doc = Index_IndexGraphEntity(idx, (const GraphEntity *)n,
			(const void *)&key, key_len, &doc_field_count);

	if(doc_field_count == 0) {
		// entity doesn't poses any attributes which are indexed
		// remove entity from index and delete document
		Index_RemoveNode(idx, n);
		RediSearch_FreeDocument(doc);
		return;
	}

	// add document to RediSearch index
	RediSearch_SpecAddDocument(rsIdx, doc);
}

void Index_RemoveNode
(
	Index idx,     // index to update
	const Node *n  // node to remove from index
) {
	ASSERT(n   != NULL);
	ASSERT(idx != NULL);

	EntityID id     = ENTITY_GET_ID(n);
	RSIndex  *rsIdx = Index_RSIndex(idx);

	RediSearch_DeleteDocument(rsIdx, &id, sizeof(EntityID));
}

