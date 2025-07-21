/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "encode_v13.h"
#include "../../../globals.h"

// Determine whether we are in the context of a bgsave, in which case
// the process is independent and should not acquire locks
static inline bool _shouldAcquireLocks(void) {
	return !Globals_Get_ProcessIsChild();
}

static void _RdbSaveHeader
(
	RedisModuleIO *rdb,
	GraphContext *gc
) {
	// Header format:
	// Graph name
	// Node count
	// Edge count
	// Deleted node count
	// Deleted edge count
	// Label matrix count
	// Relation matrix count - N
	// Does relationship Ri holds mutiple edges under a single entry X N 
	// Number of graph keys (graph context key + meta keys)
	// Schema

	ASSERT(gc != NULL);

	GraphEncodeHeader *header = &(gc->encoding_context->header);

	// graph name
	RedisModule_SaveStringBuffer(rdb, header->graph_name, strlen(header->graph_name) + 1);

	// node count
	RedisModule_SaveUnsigned(rdb, header->node_count);

	// edge count
	RedisModule_SaveUnsigned(rdb, header->edge_count);

	// deleted node count
	RedisModule_SaveUnsigned(rdb, header->deleted_node_count);

	// deleted edge count
	RedisModule_SaveUnsigned(rdb, header->deleted_edge_count);

	// label matrix count
	RedisModule_SaveUnsigned(rdb, header->label_matrix_count);

	// relation matrix count
	RedisModule_SaveUnsigned(rdb, header->relationship_matrix_count);

	// does relationship Ri holds mutiple edges under a single entry X N
	for(int i = 0; i < header->relationship_matrix_count; i++) {
		// true if R[i] contain a multi edge entry
		RedisModule_SaveUnsigned(rdb, header->multi_edge[i]);
	}

	// number of keys
	RedisModule_SaveUnsigned(rdb, header->key_count);

	// save graph schemas
	RdbSaveGraphSchema_v13(rdb, gc);
}

// returns a state information regarding the number of entities required
// to encode in this state
static PayloadInfo _StatePayloadInfo
(
	GraphContext *gc,
	EncodeState state,
	uint64_t offset,
	uint64_t entities_to_encode
) {
	uint64_t required_entities_count = 0;
	switch(state) {
	case ENCODE_STATE_NODES:
		required_entities_count = Graph_NodeCount(gc->g);
		break;
	case ENCODE_STATE_DELETED_NODES:
		required_entities_count = Graph_DeletedNodeCount(gc->g);
		break;
	case ENCODE_STATE_EDGES:
		required_entities_count = Graph_EdgeCount(gc->g);
		break;
	case ENCODE_STATE_DELETED_EDGES:
		required_entities_count = Graph_DeletedEdgeCount(gc->g);
		break;
	case ENCODE_STATE_GRAPH_SCHEMA:
		required_entities_count = 1;
		break;
	default:
		ASSERT(false && "Unknown encoding state in _CurrentStatePayloadInfo");
		break;
	}

	PayloadInfo payload_info;
	payload_info.state = state;
	// when this state will be encoded, the number of entities to encode
	// is the minimum between the number of entities to encode and
	// the remaining entities left to encode from the same type
	payload_info.entities_count = MIN(entities_to_encode, required_entities_count - offset);
	return payload_info;
}

// this function saves the key content schema
// and returns it so the encoder can know how to encode the key
static PayloadInfo *_RdbSaveKeySchema
(
	RedisModuleIO *rdb,
	GraphContext *gc
) {
	//  Format:
	//  #Number of payloads info - N
	//  N * Payload info:
	//      Encode state
	//      Number of entities encoded in this state

	PayloadInfo *payloads = array_new(PayloadInfo, 1);

	// get current encoding state
	EncodeState current_state = GraphEncodeContext_GetEncodeState(gc->encoding_context);

	// if this is the start of the encodeing, set the state to be NODES
	if(current_state == ENCODE_STATE_INIT) current_state = ENCODE_STATE_NODES;

	uint64_t remaining_entities;
	Config_Option_get(Config_VKEY_MAX_ENTITY_COUNT, &remaining_entities);

	// check if this is the last key
	bool last_key = GraphEncodeContext_GetProcessedKeyCount(gc->encoding_context) ==
					(GraphEncodeContext_GetKeyCount(gc->encoding_context) - 1);
	if(last_key) remaining_entities = VKEY_ENTITY_COUNT_UNLIMITED;

	// get the current state encoded entities count
	uint64_t offset = GraphEncodeContext_GetProcessedEntitiesOffset(gc->encoding_context);

	// while there are still remaining entities to encode in this key
	// and the state is valid
	while(remaining_entities > 0 && current_state < ENCODE_STATE_FINAL) {
		// get the current state payload info, with respect to offset
		PayloadInfo current_state_payload_info = _StatePayloadInfo(gc,
				current_state, offset, remaining_entities);
		array_append(payloads, current_state_payload_info);
		if(!last_key) remaining_entities -= current_state_payload_info.entities_count;
		if(remaining_entities > 0) {
			offset = 0; // new state offset is 0
			current_state++; // advance in the states
		}
	}

	// save the number of payloads
	uint payloads_count = array_len(payloads);
	RedisModule_SaveUnsigned(rdb, payloads_count);
	for(uint i = 0; i < payloads_count; i++) {
		// for each payload
		// save its type and the number of entities it contains
		PayloadInfo payload_info = payloads[i];
		RedisModule_SaveUnsigned(rdb, payload_info.state);
		RedisModule_SaveUnsigned(rdb, payload_info.entities_count);
	}

	return payloads;
}

void RdbSaveGraph_v13
(
	RedisModuleIO *rdb,
	void *value
) {
	// Encoding format for graph context and graph meta key:
	//  Header
	//  Payload(s) count: N
	//  Key content X N:
	//      Payload type (Nodes / Edges / Deleted nodes/ Deleted edges/ Graph schema)
	//      Entities in payload
	//  Payload(s) X N
	//
	// This function will encode each payload type (if needed) in the following order:
	// 1. Nodes
	// 2. Deleted nodes
	// 3. Edges
	// 4. Deleted edges
	// 5. Graph schema
	//
	// Each payload type can spread over one or more keys. For example:
	// A graph with 200,000 nodes, and the number of entities per payload
	// is 100,000 then there will be two nodes payloads,
	// each containing 100,000 nodes, encoded into two different RDB meta keys

	GraphContext *gc = value;

	// TODO: remove, no need, as GIL is taken

	// acquire a read lock if we're not in a thread-safe context
	if(_shouldAcquireLocks()) Graph_AcquireReadLock(gc->g);

	EncodeState current_state = GraphEncodeContext_GetEncodeState(gc->encoding_context);

	if(current_state == ENCODE_STATE_INIT) {
		// inital state, populate encoding context header
		GraphEncodeContext_InitHeader(gc->encoding_context, gc->graph_name, gc->g);
	}

	// save header
	_RdbSaveHeader(rdb, gc);

	// save payloads info for this key and retrive the key schema
	PayloadInfo *key_schema = _RdbSaveKeySchema(rdb, gc);

	uint payloads_count = array_len(key_schema);
	for(uint i = 0; i < payloads_count; i++) {
		// if the current key encoding more than one payload type,
		// payloads count >1 and we are in a new state, zero the entities count
		if(i > 0) GraphEncodeContext_SetProcessedEntitiesOffset(gc->encoding_context, 0);
		PayloadInfo payload = key_schema[i];
		switch(payload.state) {
		case ENCODE_STATE_NODES:
			RdbSaveNodes_v13(rdb, gc, payload.entities_count);
			break;
		case ENCODE_STATE_DELETED_NODES:
			RdbSaveDeletedNodes_v13(rdb, gc, payload.entities_count);
			break;
		case ENCODE_STATE_EDGES:
			RdbSaveEdges_v13(rdb, gc, payload.entities_count);
			break;
		case ENCODE_STATE_DELETED_EDGES:
			RdbSaveDeletedEdges_v13(rdb, gc, payload.entities_count);
			break;
		case ENCODE_STATE_GRAPH_SCHEMA:
			// skip, handled in _RdbSaveHeader
			break;
		default:
			ASSERT(false && "Unknown encoding phase");
			break;
		}

		// save the current state and the number of encoded entities
		GraphEncodeContext_SetEncodeState(gc->encoding_context, payload.state);
		uint64_t offset = GraphEncodeContext_GetProcessedEntitiesOffset(gc->encoding_context);
		GraphEncodeContext_SetProcessedEntitiesOffset(gc->encoding_context,
													  payload.entities_count + offset);
	}
	array_free(key_schema);

	// increase processed key count
	// if finished encoding, reset context
	GraphEncodeContext_IncreaseProcessedKeyCount(gc->encoding_context);
	if(GraphEncodeContext_Finished(gc->encoding_context)) {
		GraphEncodeContext_Reset(gc->encoding_context);
		RedisModuleCtx *ctx = RedisModule_GetContextFromIO(rdb);
		RedisModule_Log(ctx, "notice", "Done encoding graph %s", gc->graph_name);
	}

	// if a lock was acquired, release it
	if(_shouldAcquireLocks()) Graph_ReleaseLock(gc->g);
}

