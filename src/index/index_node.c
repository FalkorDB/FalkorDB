/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "index.h"
#include "index_doc_key.h"
#include "../value.h"
#include "../query_ctx.h"
#include "../graph/graphcontext.h"

extern RSDoc *Index_IndexGraphEntity(Index idx, const GraphEntity *e,
		const void *key, size_t key_len, uint *doc_field_count);

void Index_IndexNode
(
	Index idx,
	const Node *n
) {
	ASSERT(n    !=  NULL);
	ASSERT(idx  !=  NULL);

	// Acquire a strong ref for the duration of this op. Returns NULL
	// if the spec was dropped concurrently; in that case there's
	// nothing to index.
	RSIndex *rsIdx = Index_AcquireRSIndex(idx);
	if(rsIdx == NULL) return;

	char     doc_key[NODE_DOC_KEY_BUF];
	RSDoc    *doc            = NULL;
	uint     doc_field_count = 0;

	IndexDocKey_EncodeNode(ENTITY_GET_ID(n), doc_key);

	// create RediSearch document representing node
	doc = Index_IndexGraphEntity(idx, (const GraphEntity *)n,
			(const void *)doc_key, NODE_DOC_KEY_LEN, &doc_field_count);

	if(doc_field_count == 0) {
		// entity doesn't poses any attributes which are indexed
		// remove entity from index and delete document. Use the
		// already-acquired ref directly to avoid re-entering
		// Index_RemoveNode (which would Acquire again).
		RediSearch_DeleteDocument(rsIdx, doc_key, NODE_DOC_KEY_LEN);
		RediSearch_FreeDocument(doc);
		Index_ReleaseRSIndex(rsIdx);
		return;
	}

	// add document to RediSearch index
	int res = RediSearch_SpecAddDocument(rsIdx, doc);
	ASSERT(res == REDISMODULE_OK);

	Index_ReleaseRSIndex(rsIdx);
}

void Index_RemoveNode
(
	Index idx,     // index to update
	const Node *n  // node to remove from index
) {
	ASSERT(n   != NULL);
	ASSERT(idx != NULL);

	RSIndex *rsIdx = Index_AcquireRSIndex(idx);
	if(rsIdx == NULL) return;

	char doc_key[NODE_DOC_KEY_BUF];
	IndexDocKey_EncodeNode(ENTITY_GET_ID(n), doc_key);
	RediSearch_DeleteDocument(rsIdx, doc_key, NODE_DOC_KEY_LEN);

	Index_ReleaseRSIndex(rsIdx);
}

