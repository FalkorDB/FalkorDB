/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "stdlib.h"
#include "stdint.h"
#include "stdbool.h"
#include "../graph/graph.h"
#include "../graph/tensor/tensor.h"
#include "../util/datablock/datablock.h"
#include "../graph/delta_matrix/delta_matrix_iter.h"
#include "../graph/entities/graph_entity.h"
#include "rax.h"

// Encoding states
typedef enum {
	ENCODE_STATE_INIT              = 0, // encoding initial state
	ENCODE_STATE_NODES             = 1, // encoding nodes
	ENCODE_STATE_DELETED_NODES     = 2, // encoding deleted nodes
	ENCODE_STATE_EDGES             = 3, // encoding edges
	ENCODE_STATE_DELETED_EDGES     = 4, // encoding deleted edges
	ENCODE_STATE_GRAPH_SCHEMA      = 5, // encoding graph schemas
	ENCODE_STATE_LABELS_MATRICES   = 6, // encoding graph label matrices
	ENCODE_STATE_RELATION_MATRICES = 7, // encoding graph relation matrices
	ENCODE_STATE_FINAL             = 8  // encoding final state [MUSTN'T BE SAVED TO RDB]
} EncodeState;

// Header information encoded for every payload
typedef struct {
	bool *multi_edge;                // true if R[i] contain a multi edge entry
	uint64_t key_count;              // number of virtual keys + primary key
	uint64_t node_count;             // number of nodes
	uint64_t edge_count;             // number of edges
	uint64_t deleted_node_count;      // number of deleted nodes
	uint64_t deleted_edge_count;     // number of deleted edges
	const char *graph_name;          // name of graph
	uint label_matrix_count;         // number of label matrices
	uint relationship_matrix_count;  // number of relation matrices
} GraphEncodeHeader;

// GraphEncodeContext maintains the state of a graph being encoded or decoded
typedef struct {
	rax *meta_keys;                         // the holds the names of meta keys representing the graph
	uint64_t offset;                        // number of encoded entities in the current state
	EncodeState state;                      // represents the current encoding state
	uint64_t keys_processed;                // count the number of procssed graph keys
	GraphEncodeHeader header;               // header replied for each vkey
	uint64_t vkey_entity_count;             // number of entities in a single virtual key
	uint current_relation_matrix_id;        // current encoded relationship matrix
	DataBlockIterator *datablock_iterator;  // datablock iterator to be saved in the context
	TensorIterator matrix_tuple_iterator;   // tensor iterator to be saved in the context
} GraphEncodeContext;

// Creates a new graph encoding context.
GraphEncodeContext *GraphEncodeContext_New();

// Reset a graph encoding context.
void GraphEncodeContext_Reset(GraphEncodeContext *ctx);

// Populates graph encode context header.
void GraphEncodeContext_InitHeader(GraphEncodeContext *ctx, const char *graph_name, Graph *g);

// Retrieve the graph current encoding phase.
EncodeState GraphEncodeContext_GetEncodeState(const GraphEncodeContext *ctx);

// Sets the graph current encoding phase.
void GraphEncodeContext_SetEncodeState(GraphEncodeContext *ctx, EncodeState phase);

// Retrieve the graph representing keys count.
uint64_t GraphEncodeContext_GetKeyCount(const GraphEncodeContext *ctx);

// Add a meta key name, required for encoding the graph.
void GraphEncodeContext_AddMetaKey(GraphEncodeContext *ctx, const char *key);

// Returns a dynamic array with copies of the meta key names.
unsigned char **GraphEncodeContext_GetMetaKeys(const GraphEncodeContext *ctx);

// Removes the stored meta key names from the context.
void GraphEncodeContext_ClearMetaKeys(GraphEncodeContext *ctx);

// Retrieve graph currently processed key count - keys processed so far.
uint64_t GraphEncodeContext_GetProcessedKeyCount(const GraphEncodeContext *ctx);

// Retrieve graph entities encoded so far in the current state.
uint64_t GraphEncodeContext_GetProcessedEntitiesOffset(const GraphEncodeContext *ctx);

// Update the graph entities encoded so far in the current state.
void GraphEncodeContext_SetProcessedEntitiesOffset(GraphEncodeContext *ctx, uint64_t offset);

// Retrieve stored datablock iterator.
DataBlockIterator *GraphEncodeContext_GetDatablockIterator(const GraphEncodeContext *ctx);

// Set graph encoding context datablock iterator - keep iterator state for further usage.
void GraphEncodeContext_SetDatablockIterator(GraphEncodeContext *ctx, DataBlockIterator *iter);

// Retrieve graph encoding context current encoded relation matrix id.
uint GraphEncodeContext_GetCurrentRelationID(const GraphEncodeContext *ctx);

// Set graph encoding context current encoded relation matrix id.
void GraphEncodeContext_SetCurrentRelationID
(
	GraphEncodeContext *ctx,
	uint current_relation_matrix_id
);

// Retrieve stored matrix tuple iterator.
TensorIterator *GraphEncodeContext_GetMatrixTupleIterator(GraphEncodeContext *ctx);

// Returns if the the number of processed keys is equal to the total number of graph keys.
bool GraphEncodeContext_Finished(const GraphEncodeContext *ctx);

// Increases the number of processed graph keys.
void GraphEncodeContext_IncreaseProcessedKeyCount(GraphEncodeContext *ctx);

// Free graph encoding context.
void GraphEncodeContext_Free(GraphEncodeContext *ctx);
