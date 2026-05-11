/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "index.h"
#include "../graph/entities/graph_entity.h"

// Hex-encoded doc-key sizes (number of chars, excluding NUL terminator).
// EntityID is 8 bytes, EdgeIndexKey is 24 bytes; each byte expands to two
// lowercase hex chars.
#define NODE_DOC_KEY_LEN  (sizeof(EntityID) * 2)
#define EDGE_DOC_KEY_LEN  (sizeof(EdgeIndexKey) * 2)
#define NODE_DOC_KEY_BUF  (NODE_DOC_KEY_LEN + 1)
#define EDGE_DOC_KEY_BUF  (EDGE_DOC_KEY_LEN + 1)

// Encode an 8-byte EntityID as NODE_DOC_KEY_LEN lowercase hex chars,
// followed by a NUL terminator.
void IndexDocKey_EncodeNode
(
	EntityID id,                   // entity id
	char out[NODE_DOC_KEY_BUF]     // [out] hex-encoded doc key
);

// Decode NODE_DOC_KEY_LEN lowercase hex chars back to an EntityID.
void IndexDocKey_DecodeNode
(
	const char *in,                // hex-encoded doc key
	EntityID *out                  // [out] decoded entity id
);

// Encode an EdgeIndexKey (src_id, dest_id, edge_id) as EDGE_DOC_KEY_LEN
// lowercase hex chars, followed by a NUL terminator.
void IndexDocKey_EncodeEdge
(
	const EdgeIndexKey *key,       // edge doc key
	char out[EDGE_DOC_KEY_BUF]     // [out] hex-encoded doc key
);

// Decode EDGE_DOC_KEY_LEN lowercase hex chars back to an EdgeIndexKey.
void IndexDocKey_DecodeEdge
(
	const char *in,                // hex-encoded doc key
	EdgeIndexKey *out              // [out] decoded edge doc key
);
