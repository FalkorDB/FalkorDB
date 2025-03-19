/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v17.h"

// forward declarations
static SIValue _RdbLoadPoint(SerializerIO rdb);
static SIValue _RdbLoadSIArray(SerializerIO rdb);
static SIValue _RdbLoadVector(SerializerIO rdb, SIType t);

static SIValue _RdbLoadSIValue
(
	SerializerIO rdb
) {
	// Format:
	// SIType
	// Value
	SIType t = SerializerIO_ReadUnsigned(rdb);
	switch(t) {
	case T_INT64:
		return SI_LongVal(SerializerIO_ReadSigned(rdb));
	case T_DOUBLE:
		return SI_DoubleVal(SerializerIO_ReadDouble(rdb));
	case T_STRING:
		// transfer ownership of the heap-allocated string to the
		// newly-created SIValue
		return SI_TransferStringVal(SerializerIO_ReadBuffer(rdb, NULL));
	case T_BOOL:
		return SI_BoolVal(SerializerIO_ReadSigned(rdb));
	case T_ARRAY:
		return _RdbLoadSIArray(rdb);
	case T_POINT:
		return _RdbLoadPoint(rdb);
	case T_VECTOR_F32:
		return _RdbLoadVector(rdb, t);
	case T_NULL:
	default: // currently impossible
		return SI_NullVal();
	}
}

static SIValue _RdbLoadPoint
(
	SerializerIO rdb
) {
	double lat = SerializerIO_ReadDouble(rdb);
	double lon = SerializerIO_ReadDouble(rdb);
	return SI_Point(lat, lon);
}

static SIValue _RdbLoadSIArray
(
	SerializerIO rdb
) {
	/* loads array as
	   unsinged : array legnth
	   array[0]
	   .
	   .
	   .
	   array[array length -1]
	 */
	uint arrayLen = SerializerIO_ReadUnsigned(rdb);
	SIValue list = SI_Array(arrayLen);
	for(uint i = 0; i < arrayLen; i++) {
		SIValue elem = _RdbLoadSIValue(rdb);
		SIArray_Append(&list, elem);
		SIValue_Free(elem);
	}
	return list;
}

static SIValue _RdbLoadVector
(
	SerializerIO rdb,
	SIType t
) {
	ASSERT(t & T_VECTOR);

	// loads vector
	// unsigned : vector length
	// vector[0]
	// .
	// .
	// .
	// vector[vector length -1]

	SIValue vector;

	uint32_t dim = SerializerIO_ReadUnsigned(rdb);

	vector = SI_Vectorf32(dim);
	float *values = SIVector_Elements(vector);

	for(uint32_t i = 0; i < dim; i++) {
		values[i] = SerializerIO_ReadFloat(rdb);
	}

	return vector;
}

static void _RdbLoadEntity
(
	SerializerIO rdb,
	GraphEntity *e
) {
	// format:
	// #properties N
	// (name, value type, value) X N

	uint64_t n = SerializerIO_ReadUnsigned(rdb);

	if(n == 0) return;

	SIValue     vals[n];
	AttributeID ids [n];

	for(uint64_t i = 0; i < n; i++) {
		ids[i]  = SerializerIO_ReadUnsigned(rdb);
		vals[i] = _RdbLoadSIValue(rdb);
	}

	AttributeSet_AddNoClone(e->attributes, ids, vals, n, false);
}

// decode nodes
void RdbLoadNodes_v17
(
	SerializerIO rdb,          // RDB
	Graph *g,                  // graph context
	const uint64_t node_count  // number of nodes to decode
) {
	// node format:
	//      ID
	//      #properties N
	//      (name, value type, value) X N

	uint64_t prev_graph_node_count = Graph_NodeCount(g);

	for(uint64_t i = 0; i < node_count; i++) {
		Node n;
		NodeID id = SerializerIO_ReadUnsigned(rdb);

		Serializer_Graph_SetNode(g, id, NULL, 0, &n);

		_RdbLoadEntity(rdb, (GraphEntity *)&n);
	}

	ASSERT(prev_graph_node_count + node_count == Graph_NodeCount(g));
}

// decode deleted nodes
void RdbLoadDeletedNodes_v17
(
	SerializerIO rdb,                  // RDB
	Graph *g,                          // graph context
	const uint64_t deleted_node_count  // number of deleted nodes
) {
	// Format:
	// node id X N

	uint64_t prev_deleted_node_count = Graph_DeletedNodeCount(g);

	for(uint64_t i = 0; i < deleted_node_count; i++) {
		NodeID id = SerializerIO_ReadUnsigned(rdb);
		Serializer_Graph_MarkNodeDeleted(g, id);
	}

	// read encoded deleted node count and validate
	ASSERT(deleted_node_count + prev_deleted_node_count ==
			Graph_DeletedNodeCount(g));
}

// decode edges
static uint64_t _DecodeEdges
(
	SerializerIO rdb,  // RDB
	Graph *g           // graph
) {
	// format:
	//  edge ID
	//  edge properties

	Edge e;                      // current decoded edge
	uint64_t decoded_edges = 0;  // number of decoded edges

	// as long as we didn't hit our END MARKER
	while(true) {
		// decode edge ID
		e.id = SerializerIO_ReadUnsigned(rdb);

		// break on END MARKER
		if(e.id == INVALID_ENTITY_ID) {
			break;
		}

		// load edge attributes
		Serializer_Graph_AllocEdgeAttributes(g, e.id, &e);
		_RdbLoadEntity(rdb, (GraphEntity *)&e);

		// update number of decode edges
		decoded_edges++;
	}

	return decoded_edges;
}

// decode edges
void RdbLoadEdges_v17
(
	SerializerIO rdb,  // RDB
	Graph *g,          // graph context
	const uint64_t n   // virtual key capacity
) {
	// format:
	// {
	//  edge ID
	//  edge properties
	// } X N

	uint64_t prev_edge_count = Graph_EdgeCount(g); // #edges in the graph

	for(uint64_t i = 0; i < n;) {
		i += _DecodeEdges(rdb, g);
	}

	// read encoded deleted edge count and validate
	ASSERT(n + prev_edge_count == Graph_EdgeCount(g));
}

// decode deleted edges
void RdbLoadDeletedEdges_v17
(
	SerializerIO rdb,                  // RDB
	Graph *g,                          // graph context
	const uint64_t deleted_edge_count  // number of deleted edges
) {
	// Format:
	// edge id X N

	uint64_t prev_deleted_edge_count = Graph_DeletedEdgeCount(g);

	for(uint64_t i = 0; i < deleted_edge_count; i++) {
		EdgeID id = SerializerIO_ReadUnsigned(rdb);
		Serializer_Graph_MarkEdgeDeleted(g, id);
	}

	// read encoded deleted edge count and validate
	ASSERT(deleted_edge_count + prev_deleted_edge_count ==
			Graph_DeletedEdgeCount(g));
}

