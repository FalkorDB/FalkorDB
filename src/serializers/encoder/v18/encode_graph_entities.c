/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "encode_v18.h"
#include "../../../datatypes/datatypes.h"

// forword decleration
static void _RdbSaveSIValue
(
	SerializerIO rdb,
	const SIValue *v
);

static void _RdbSaveSIArray
(
	SerializerIO rdb,
	const SIValue list
) {
	// saves array as
	// unsigned : array legnth
	// array[0]
	// .
	// .
	// .
	// array[array length -1]

	uint arrayLen = SIArray_Length(list);
	SerializerIO_WriteUnsigned(rdb, arrayLen);
	for(uint i = 0; i < arrayLen; i ++) {
		SIValue value = SIArray_Get(list, i);
		_RdbSaveSIValue(rdb, &value);
	}
}

static void _RdbSaveSIVector
(
	SerializerIO rdb, 
	SIValue v
) {
	// saves a vector
	// unsigned : vector dimension
	// vector[0]
	// .
	// .
	// .
	// vector[vector dimension -1]

	void   *vec = v.ptrval;
	size_t n    = SIVector_ByteSize(v);

	SerializerIO_WriteBuffer(rdb, vec, n);
}

static void _RdbSaveSIValue
(
	SerializerIO rdb,
	const SIValue *v
) {
	// Format:
	// SIType
	// Value

	SerializerIO_WriteUnsigned(rdb, v->type);

	switch(v->type) {
		case T_BOOL:
		case T_INT64:
			SerializerIO_WriteSigned(rdb, v->longval);
			break;

		case T_DOUBLE:
			SerializerIO_WriteDouble(rdb, v->doubleval);
			break;

		case T_STRING:
		case T_INTERN_STRING:
			SerializerIO_WriteBuffer(rdb, v->stringval,
					strlen(v->stringval) + 1);
			break;

		case T_ARRAY:
			_RdbSaveSIArray(rdb, *v);
			break;

		case T_POINT:
			SerializerIO_WriteDouble(rdb, Point_lat(*v));
			SerializerIO_WriteDouble(rdb, Point_lon(*v));
			break;

		case T_VECTOR_F32:
			_RdbSaveSIVector(rdb, *v);
			break;

		case T_TIME:
		case T_DATE:
		case T_DATETIME:
		case T_DURATION:
			SerializerIO_WriteSigned(rdb, v->datetimeval);
			break;

		case T_NULL:
			break;  // no data beyond type needs to be encoded for NULL

		default:
			ASSERT(0 && "Attempted to serialize value of invalid type.");
	}
}

// encode deleted entities IDs
static inline void _RdbSaveDeletedEntities_v18
(
	SerializerIO rdb,          // RDB
	uint64_t n,                // number of deleted entities IDs to encode
	uint64_t offset,           // offset into deleted_id_list
	uint64_t *deleted_id_list  // list of deleted IDs
) {
	ASSERT(n > 0);
	ASSERT(deleted_id_list != NULL);

	// dump the entire list[offset..offset+n] into the RDB as a buffer
	SerializerIO_WriteBuffer(rdb, (const char*)(deleted_id_list + offset),
			n * sizeof(uint64_t));
}

// encode deleted node IDs
void RdbSaveDeletedNodes_v18
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	const uint64_t n   // number of deleted nodes to encode
) {
	// Format:
	// node id X N

	ASSERT(n > 0);

	// get deleted nodes list
	uint64_t *deleted_nodes_list = Serializer_Graph_GetDeletedNodesList(gc->g);
	_RdbSaveDeletedEntities_v18(rdb, n, offset, deleted_nodes_list);
}

// encode deleted edges IDs
void RdbSaveDeletedEdges_v18
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	const uint64_t n   // number of deleted edges to encode
) {
	// Format:
	// edge id X N

	ASSERT(n > 0);

	// get deleted edges list
	uint64_t *deleted_edges_list = Serializer_Graph_GetDeletedEdgesList(gc->g);
	_RdbSaveDeletedEntities_v18(rdb, n, offset, deleted_edges_list);
}

// encode graph entities
static void _SaveEntities_v18
(
	SerializerIO rdb,         // RDB
	GraphContext *gc,         // graph context
	DataBlockIterator *iter,  // entity iterator
	const uint64_t n          // number of entities to encode
) {
	// format:
	//  ID
	//  #properties N
	//  (name, value type, value) X N */

	for(uint64_t i = 0; i < n; i++) {
		GraphEntity e;
		e.attributes = (AttributeSet *)DataBlockIterator_Next(iter, &e.id);

		// save ID
		EntityID id = ENTITY_GET_ID(&e);
		SerializerIO_WriteUnsigned(rdb, id);

		// properties N
		// (name, value type, value) X N
		const AttributeSet set = GraphEntity_GetAttributes(&e);
		uint16_t attr_count = AttributeSet_Count(set);

		SerializerIO_WriteUnsigned(rdb, attr_count);

		for(int j = 0; j < attr_count; j++) {
			SIValue value ;
			AttributeID attr_id ;
			AttributeSet_GetIdx (set, j, &attr_id, &value) ;

			SerializerIO_WriteUnsigned (rdb, attr_id) ;
			_RdbSaveSIValue (rdb, &value) ;
		}
	}
}

// encode nodes
void RdbSaveNodes_v18
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // iterator offset
	const uint64_t n   // number of nodes to encode
) {
	// format:
	//  node ID
	//  #properties N
	//  (name, value type, value) X N

	// make sure there's capacity
	ASSERT(n > 0);

	// get graph's node count
	uint64_t graph_nodes = Graph_NodeCount(gc->g);

	// get datablock iterator from context,
	// already set to offset by a previous encodeing of nodes, or create new one
	DataBlockIterator *iter =
		GraphEncodeContext_GetDatablockIterator(gc->encoding_context);
	if(!iter) {
		iter = Graph_ScanNodes(gc->g);
		GraphEncodeContext_SetDatablockIterator(gc->encoding_context, iter);
	}

	_SaveEntities_v18(rdb, gc, iter, n);

	// check if done encodeing nodes
	if(offset + n == graph_nodes) {
		DataBlockIterator_Free(iter);
		iter = NULL;
		GraphEncodeContext_SetDatablockIterator(gc->encoding_context, iter);
	}
}

// encode edges
void RdbSaveEdges_v18
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	const uint64_t n   // number of edges to encode
) {
	// format:
	//  edge id
	//  #properties N
	//  (name, value type, value) X N

	// make sure there's capacity
	ASSERT(n > 0);

	// get graph's edge count
	uint64_t graph_edges = Graph_EdgeCount(gc->g);

	// get datablock iterator from context,
	// already set to offset by a previous encodeing of nodes, or create new one
	DataBlockIterator *iter =
		GraphEncodeContext_GetDatablockIterator(gc->encoding_context);
	if(!iter) {
		iter = Graph_ScanEdges(gc->g);
		GraphEncodeContext_SetDatablockIterator(gc->encoding_context, iter);
	}

	_SaveEntities_v18(rdb, gc, iter, n);

	// check if done encodeing edges
	if(offset + n == graph_edges) {
		DataBlockIterator_Free(iter);
		iter = NULL;
		GraphEncodeContext_SetDatablockIterator(gc->encoding_context, iter);
	}
}

