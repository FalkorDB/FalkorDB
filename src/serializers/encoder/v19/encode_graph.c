/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "encode_v19.h"
#include "../../../globals.h"

// Determine whether we are in the context of a bgsave, in which case
// the process is independent and should not acquire locks
static inline bool _shouldAcquireLocks(void) {
	return !Globals_Get_ProcessIsChild();
}

static void _RdbSaveHeader
(
	SerializerIO rdb,
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
	SerializerIO_WriteBuffer(rdb, header->graph_name, strlen(header->graph_name) + 1);

	// node count
	SerializerIO_WriteUnsigned(rdb, header->node_count);

	// edge count
	SerializerIO_WriteUnsigned(rdb, header->edge_count);

	// deleted node count
	SerializerIO_WriteUnsigned(rdb, header->deleted_node_count);

	// deleted edge count
	SerializerIO_WriteUnsigned(rdb, header->deleted_edge_count);

	// label matrix count
	SerializerIO_WriteUnsigned(rdb, header->label_matrix_count);

	// relation matrix count
	SerializerIO_WriteUnsigned(rdb, header->relationship_matrix_count);

	// does relationship Ri holds mutiple edges under a single entry X N
	for(int i = 0; i < header->relationship_matrix_count; i++) {
		// true if R[i] contain a multi edge entry
		SerializerIO_WriteUnsigned(rdb, header->multi_edge[i]);
	}

	// number of keys
	SerializerIO_WriteUnsigned(rdb, header->key_count);

	// save graph schemas
	RdbSaveGraphSchema_v19(rdb, gc);
}

// returns a state information regarding the number of entities required
// to encode in this state
static PayloadInfo _StatePayloadInfo
(
	GraphContext *gc,
	EncodeState state,
	uint64_t offset,
	uint64_t capacity
) {
	Graph *g = gc->g;
	uint64_t required_entities_count = 0;

	switch(state) {
		case ENCODE_STATE_NODES:
			required_entities_count = Graph_NodeCount(g);
			break;
		case ENCODE_STATE_DELETED_NODES:
			required_entities_count = Graph_DeletedNodeCount(g);
			break;
		case ENCODE_STATE_EDGES:
			required_entities_count = Graph_EdgeCount(g);
			break;
		case ENCODE_STATE_DELETED_EDGES:
			required_entities_count = Graph_DeletedEdgeCount(g);
			break;
		case ENCODE_STATE_GRAPH_SCHEMA:
			// here for historical reasons
			// can be removed once encoder / decoder version 15 is removed.
			break;
		case ENCODE_STATE_LABELS_MATRICES:
			required_entities_count = 1;  // all matrices resides in a one key
			break;
		case ENCODE_STATE_RELATION_MATRICES:
			required_entities_count = 1;  // all matrices resides in a one key
			break;
		case ENCODE_STATE_ADJ_MATRIX:
			required_entities_count = 1;
			break;
		case ENCODE_STATE_LBLS_MATRIX:
			required_entities_count = 1;
			break;
		default:
			ASSERT(false && "Unknown encoding state in _CurrentStatePayloadInfo");
			break;
	}

	PayloadInfo payload_info;

	payload_info.state  = state;
	payload_info.offset = offset;

	// when this state will be encoded, the number of entities to encode
	// is the minimum between the number of entities to encode and
	// the remaining entities left to encode from the same type
	payload_info.entities_count =
		MIN(capacity, required_entities_count - offset);

	return payload_info;
}

// this function saves the key content schema
// and returns it so the encoder can know how to encode the key
static PayloadInfo *_RdbSaveKeySchema
(
	SerializerIO rdb,
	GraphContext *gc
) {
	//  Format:
	//  #Number of payloads info - N
	//  N * Payload info:
	//      Encode state
	//      Number of entities encoded in this state

	uint32_t payloads_count = 0;
	PayloadInfo *payloads = array_new(PayloadInfo, 1);

	// get current encoding state
	EncodeState current_state =
		GraphEncodeContext_GetEncodeState(gc->encoding_context);

	// if this is the start of the encodeing, set the state to be NODES
	if(current_state == ENCODE_STATE_INIT) current_state = ENCODE_STATE_NODES;

	// number of "elements" this encoded key can hold
	uint64_t capacity;
	Config_Option_get(Config_VKEY_MAX_ENTITY_COUNT, &capacity);

	// check if this is the last key
	bool last_key =
		GraphEncodeContext_GetProcessedKeyCount(gc->encoding_context) ==
		(GraphEncodeContext_GetKeyCount(gc->encoding_context) - 1);

	// remove capacity limitation on last key
	if(last_key) capacity = VKEY_ENTITY_COUNT_UNLIMITED;

	// get the current state encoded entities count
	uint64_t offset =
		GraphEncodeContext_GetProcessedEntitiesOffset(gc->encoding_context);

	// while there are still capacity in this key and the state is valid
	while(capacity > 0 && current_state < ENCODE_STATE_FINAL) {
		// get the current state payload info, with respect to offset
		PayloadInfo payload =
			_StatePayloadInfo(gc, current_state, offset, capacity);

		// only include non empty states
		if(payload.entities_count > 0) {
			array_append(payloads, payload);
			if(!last_key) capacity -= payload.entities_count;
		}

		// if there's still room in this key
		// meaning all entities of the current type are encoded
		// reset offset for the next entity type
		if(capacity > 0) {
			offset = 0;       // new state offset is 0
			current_state++;  // advance to next state
		}
	}

	// save the number of payloads
	payloads_count = array_len(payloads);
	SerializerIO_WriteUnsigned(rdb, payloads_count);

	// save paylopads
	for(uint i = 0; i < payloads_count; i++) {
		// for each payload
		// save its type and the number of entities it contains
		PayloadInfo payload_info = payloads[i];
		SerializerIO_WriteUnsigned(rdb, payload_info.state);
		SerializerIO_WriteUnsigned(rdb, payload_info.entities_count);
	}

	return payloads;
}

void RdbSaveGraph_latest
(
	SerializerIO rdb,
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
	//
	// Each payload type can spread over one or more keys. For example:
	// A graph with 200,000 nodes, and the number of entities per payload
	// is 100,000 then there will be two nodes payloads,
	// each containing 100,000 nodes, encoded into two different RDB meta keys

	GraphContext *gc = value;
	Graph        *g = gc->g;

	// TODO: remove, no need, as GIL is taken

	// acquire a read lock if we're not in a thread-safe context
	if(_shouldAcquireLocks()) Graph_AcquireReadLock(gc->g);

	// get last encoded state
	EncodeState current_state =
		GraphEncodeContext_GetEncodeState(gc->encoding_context);

	if(current_state == ENCODE_STATE_INIT) {
		// inital state, populate encoding context header
		GraphEncodeContext_InitHeader(gc->encoding_context, gc->graph_name, g);
	}

	// save header
	_RdbSaveHeader(rdb, gc);

	// save payloads info for this key and retrive the key schema
	PayloadInfo *payloads = _RdbSaveKeySchema(rdb, gc);

	PayloadInfo *payload = NULL;
	uint32_t payloads_count = array_len(payloads);

	for(uint i = 0; i < payloads_count; i++) {
		payload = payloads+i;

		switch(payload->state) {
			case ENCODE_STATE_NODES:
				RdbSaveNodes_v19(rdb, gc, payload->offset,
						payload->entities_count);
				break;

			case ENCODE_STATE_DELETED_NODES:
				RdbSaveDeletedNodes_v19(rdb, gc, payload->offset,
						payload->entities_count);
				break;

			case ENCODE_STATE_EDGES:
				RdbSaveEdges_v19(rdb, gc, payload->offset,
						payload->entities_count);
				break;

			case ENCODE_STATE_DELETED_EDGES:
				RdbSaveDeletedEdges_v19(rdb, gc, payload->offset,
						payload->entities_count);
				break;

			case ENCODE_STATE_LABELS_MATRICES:
				RdbSaveLabelMatrices_v19(rdb, g);
				break;

			case ENCODE_STATE_RELATION_MATRICES:
				RdbSaveRelationMatrices_v19(rdb, g);
				break;

			case ENCODE_STATE_ADJ_MATRIX:
				RdbSaveAdjMatrix_v19(rdb, g);
				break;

			case ENCODE_STATE_LBLS_MATRIX:
				RdbSaveLblsMatrix_v19(rdb, g);
				break;

			default:
				ASSERT(false && "Unknown encoding phase");
				break;
		}
	}

	// update encoding state for next virtual key
	if(payload != NULL) {
		// save the current state
		GraphEncodeContext_SetEncodeState(gc->encoding_context, payload->state);

		// save offset
		GraphEncodeContext_SetProcessedEntitiesOffset(gc->encoding_context,
				payload->offset + payload->entities_count);
	}

	// free payloads
	array_free(payloads);

	// increase processed key count
	// if finished encoding, reset context
	GraphEncodeContext_IncreaseProcessedKeyCount(gc->encoding_context);
	if(GraphEncodeContext_Finished(gc->encoding_context)) {
		GraphEncodeContext_Reset(gc->encoding_context);
	}

	// if a lock was acquired, release it
	if(_shouldAcquireLocks()) Graph_ReleaseLock(g);
}

