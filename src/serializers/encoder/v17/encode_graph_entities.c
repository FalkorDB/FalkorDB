/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "encode_v17.h"
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

	uint32_t dim = SIVector_Dim(v);
	SerializerIO_WriteUnsigned(rdb, dim);

	// get vector elements
	void *elements = SIVector_Elements(v);

	// save individual elements
	float *values = (float*)elements;
	for(uint32_t i = 0; i < dim; i ++) {
		SerializerIO_WriteFloat(rdb, values[i]);
	}
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
		case T_NULL:
			break;  // no data beyond type needs to be encoded for NULL
		default:
			ASSERT(0 && "Attempted to serialize value of invalid type.");
	}
}

static void _RdbSaveEntity
(
	SerializerIO rdb,
	const GraphEntity *e
) {
	// Format:
	// #attributes N
	// (name, value type, value) X N 

	const AttributeSet set = GraphEntity_GetAttributes(e);
	uint16_t attr_count = AttributeSet_Count(set);

	SerializerIO_WriteUnsigned(rdb, attr_count);

	for(int i = 0; i < attr_count; i++) {
		AttributeID attr_id;
		SIValue value = AttributeSet_GetIdx(set, i, &attr_id);
		SerializerIO_WriteUnsigned(rdb, attr_id);
		_RdbSaveSIValue(rdb, &value);
	}
}

static void _RdbSaveEdge
(
	SerializerIO rdb,
	const Graph *g,
	const Edge *e,
	int r,
	bool multi_edge
) {

	// Format:
	//  edge ID
	//  source node ID
	//  destination node ID
	//  multi-edge
	//  edge properties

	SerializerIO_WriteUnsigned(rdb, ENTITY_GET_ID(e));

	// source node ID
	SerializerIO_WriteUnsigned(rdb, Edge_GetSrcNodeID(e));

	// destination node ID
	SerializerIO_WriteUnsigned(rdb, Edge_GetDestNodeID(e));

	// relation type
	SerializerIO_WriteUnsigned(rdb, r);

	// multi-edge
	SerializerIO_WriteUnsigned(rdb, multi_edge);

	// edge properties
	_RdbSaveEntity(rdb, (GraphEntity *)e);
}

// encode a single node
static void _RdbSaveNode_v17
(
	SerializerIO rdb,
	GraphContext *gc,
	GraphEntity *n
) {
	// Format:
	//     ID
	//     #properties N
	//     (name, value type, value) X N */

	// save ID
	EntityID id = ENTITY_GET_ID(n);
	SerializerIO_WriteUnsigned(rdb, id);

	// properties N
	// (name, value type, value) X N
	_RdbSaveEntity(rdb, (GraphEntity *)n);
}

// encode deleted entities IDs
static void _RdbSaveDeletedEntities_v17
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t n,
	uint64_t offset,
	uint64_t *deleted_id_list
) {
	ASSERT(n > 0);

	// iterated over the required range in the datablock deleted items
	for(uint64_t i = offset; i < offset + n; i++) {
		SerializerIO_WriteUnsigned(rdb, deleted_id_list[i]);
	}
}

// encode deleted node IDs
void RdbSaveDeletedNodes_v17
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
	_RdbSaveDeletedEntities_v17(rdb, gc, n, offset, deleted_nodes_list);
}

// encode deleted edges IDs
void RdbSaveDeletedEdges_v17
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
	_RdbSaveDeletedEntities_v17(rdb, gc, n, offset, deleted_edges_list);
}

// encode nodes
void RdbSaveNodes_v17
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // iterator offset
	const uint64_t n   // number of nodes to encode
) {
	// Format:
	// Node Format * nodes_to_encode:
	//  ID
	//  #properties N
	//  (name, value type, value) X N

	ASSERT(n != 0);

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

	for(uint64_t i = 0; i < n; i++) {
		GraphEntity e;
		e.attributes = (AttributeSet *)DataBlockIterator_Next(iter, &e.id);
		_RdbSaveNode_v17(rdb, gc, &e);
	}

	// check if done encodeing nodes
	if(offset + n == graph_nodes) {
		DataBlockIterator_Free(iter);
		iter = NULL;
		GraphEncodeContext_SetDatablockIterator(gc->encoding_context, iter);
	}
}

// write relationship type header
static void _EncodeRelationHeader
(
	SerializerIO rdb,
	const Graph *g,
	RelationID r
) {
	// Header:
	//     relationship ID
	//     rather or not relationship contains tensors

	// encode relationship id
	SerializerIO_WriteUnsigned(rdb, r);

	bool tensor = Graph_RelationshipContainsMultiEdge(g, r);

	// encode rather or not relation contains tensors
	SerializerIO_WriteUnsigned(rdb, tensor);
}

// encode edges
static void _EncodeEdges
(
	SerializerIO rdb,    // RDB
	const Graph *g,      // graph
	TensorIterator *it,  // tensor iterator
	uint64_t *n          // max number of edges to encode
) {
	// Format:
	//  edge ID
	//  edge properties

	Edge   e;          // current edge
	EdgeID edgeID;     // edge id
	uint64_t _n = *n;  // virtual key capacity

	// as long as there's room in the virtual key
	// and iterator isn't depleted
	while(_n > 0 &&
		  (TensorIterator_next(it, &e.src_id, &e.dest_id, &edgeID, NULL))) {
		// get edge attribute set
		bool edge_found = Graph_GetEdge(g, edgeID, &e);
		ASSERT(edge_found == true);

		// encode edge ID
		SerializerIO_WriteUnsigned(rdb, edgeID);

		// encode edge properties
		_RdbSaveEntity(rdb, (GraphEntity *)&e);

		// reduce capacity
		_n--;
	}

	// update capacity
	*n = _n;
}

// encode tensors
static void _EncodeTensors
(
	SerializerIO rdb,    // RDB
	const Graph *g,      // graph
	TensorIterator *it,  // tensor iterator
	uint64_t *n          // max number of edges to encode
) {
	// format:
	// edge format:
	//     edge id
	//     edge properties
	//     multi-edge
	//     source node id       [optional]
	//     destination node id  [optional]

	Edge   e;            // current encoded edge
	bool tensor;         // rather or not the edge is part of a tensor
	EdgeID edgeID;       // edge ID
	uint64_t _n = *n;    // virtual key capacity

	// as long as there's room in the virtual key
	// and iterator isn't depleted
	while(_n > 0 &&
		  (TensorIterator_next(it, &e.src_id, &e.dest_id, &edgeID, &tensor))) {
		// get edge attribute set
		bool edge_found = Graph_GetEdge(g, edgeID, &e);
		ASSERT(edge_found == true);

		// encode edge ID
		SerializerIO_WriteUnsigned(rdb, edgeID);

		// encode edge properties
		_RdbSaveEntity(rdb, (GraphEntity *)&e);

		// encode tensor
		SerializerIO_WriteUnsigned(rdb, tensor);

		if(unlikely(tensor)) {
			// encode source node ID
			SerializerIO_WriteUnsigned(rdb, e.src_id);

			// encode destination node ID
			SerializerIO_WriteUnsigned(rdb, e.dest_id);
		}

		// reduce capacity
		_n--;
	}

	// update capacity
	*n = _n;
}

// encode edges
void RdbSaveEdges_v17
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	uint64_t offset,   // offset
	const uint64_t n   // number of edges to encode
) {
	// format:
	//
	// Header:
	//     relationship ID
	//     rather or not relationship contains tensors
	//
	// edge format:
	//     edge id
	//     edge properties
	//     multi-edge [only if relation contains tensors]
	//     source node id
	//     destination node id


	// make sure there's capacity
	ASSERT(n > 0);

	// number of relationship matrices in the graph
	int relations_count = Graph_RelationTypeCount(gc->g);

	Delta_Matrix R;  // current relation matrix

	// get matrix tuple iterator from context
	// already set to the next entry to fetch
	// for previous edge encide or create new one
	TensorIterator *it =
		GraphEncodeContext_GetMatrixTupleIterator(gc->encoding_context);

	// get current relation matrix
	uint r = GraphEncodeContext_GetCurrentRelationID(gc->encoding_context);

	// first relationship matrix
	if(r == 0) {
		R = Graph_GetRelationMatrix(gc->g, r, false);

		// attach iterator if not already attached
		if(!TensorIterator_is_attached(it, R)) {
			TensorIterator_ScanRange(it, R, 0, UINT64_MAX, false);
		}
	}

	//--------------------------------------------------------------------------
	// encode edges
	//--------------------------------------------------------------------------

	uint64_t _n = n;

	// as long as there's capacity in this virtual key
	while(_n > 0) {
		// encode relation header
		_EncodeRelationHeader(rdb, gc->g, r);

		// check if current relationship matrix contains tensors
		bool tensors = Graph_RelationshipContainsMultiEdge(gc->g, r);

		if(tensors) {
			// encode tensors
			_EncodeTensors(rdb, gc->g, it, &_n);
		} else {
			// encode edges
			_EncodeEdges(rdb, gc->g, it, &_n);
		}

		// encode end marker
		SerializerIO_WriteUnsigned(rdb, INVALID_ENTITY_ID);

		// there's still room in the VKey
		if(_n > 0) {
			// move to the next relation
			if(++r == relations_count) {
				// no more relations break
				break;
			}

			// set iterator on new relation matrix
			R = Graph_GetRelationMatrix(gc->g, r, false);
			TensorIterator_ScanRange(it, R, 0, UINT64_MAX, false);
		}
	}

	// check if done encoding edges
	if(offset + n == Graph_EdgeCount(gc->g)) {
		*it = (TensorIterator){0};
	}

	// update encoding context
	GraphEncodeContext_SetCurrentRelationID(gc->encoding_context, r);
}

