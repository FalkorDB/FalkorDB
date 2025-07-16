/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "stdlib.h"
#include "stdint.h"
#include "stdbool.h"
#include "../graph/graph.h"
#include "../util/datablock/datablock.h"
#include "../graph/rg_matrix/rg_matrix_iter.h"
#include "../graph/entities/graph_entity.h"
#include "rax.h"

// Encoding states
typedef enum {
	ENCODE_STATE_INIT,          // encoding initial state
	ENCODE_STATE_NODES,         // encoding nodes
	ENCODE_STATE_DELETED_NODES, // encoding deleted nodes
	ENCODE_STATE_EDGES,         // encoding edges
	ENCODE_STATE_DELETED_EDGES, // encoding deleted edges
	ENCODE_STATE_GRAPH_SCHEMA,  // encoding graph schemas
	ENCODE_STATE_FINAL          // encoding final state
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
	rax *meta_keys;                             // The holds the names of meta keys representing the graph.
	uint64_t offset;                            // Number of encoded entities in the current state.
	EncodeState state;                          // Represents the current encoding state.
	uint64_t keys_processed;                    // Count the number of procssed graph keys.
	GraphEncodeHeader header;                   // Header replied for each vkey
	uint64_t vkey_entity_count;                 // Number of entities in a single virtual key.
	NodeID multiple_edges_src_id;               // The current edges array sourc node id.
	NodeID multiple_edges_dest_id;              // The current edges array destination node id.
	EdgeID *multiple_edges_array;               // Multiple edges array, save in the context.
	uint current_relation_matrix_id;            // Current encoded relationship matrix.
	uint multiple_edges_current_index;          // The current index of the encoded edges array.
	DataBlockIterator *datablock_iterator;      // Datablock iterator to be saved in the context.
	RG_MatrixTupleIter matrix_tuple_iterator;   // Matrix tuple iterator to be saved in the context.
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
void GraphEncodeContext_SetCurrentRelationID(GraphEncodeContext *ctx,
											 uint current_relation_matrix_id);

// Retrieve stored matrix tuple iterator.
RG_MatrixTupleIter *GraphEncodeContext_GetMatrixTupleIterator(GraphEncodeContext *ctx);

// Sets a multiple edges array and the current index, for saving the state of multiple edges encoding.
void GraphEncodeContext_SetMutipleEdgesArray(GraphEncodeContext *ctx, EdgeID *edges,
											 uint current_index, NodeID src, NodeID dest);

// Retrive the multiple edges array, to continue array of multiple edge encoding.
EdgeID *GraphEncodeContext_GetMultipleEdgesArray(const GraphEncodeContext *ctx);

// Retrive the multiple edges array current index, to continue array of multiple edge encoding.
uint GraphEncodeContext_GetMultipleEdgesCurrentIndex(const GraphEncodeContext *ctx);

// Retrive the multiple edges array source node.
NodeID GraphEncodeContext_GetMultipleEdgesSourceNode(const GraphEncodeContext *ctx);

// Retrive the multiple edges array destination node.
NodeID GraphEncodeContext_GetMultipleEdgesDestinationNode(const GraphEncodeContext *ctx);

// Returns if the the number of processed keys is equal to the total number of graph keys.
bool GraphEncodeContext_Finished(const GraphEncodeContext *ctx);

// Increases the number of processed graph keys.
void GraphEncodeContext_IncreaseProcessedKeyCount(GraphEncodeContext *ctx);

// Free graph encoding context.
void GraphEncodeContext_Free(GraphEncodeContext *ctx);

