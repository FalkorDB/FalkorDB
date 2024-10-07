/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v14.h"

// forward declarations
static SIValue _RdbLoadPoint(SerializerIO rdb);
static SIValue _RdbLoadSIArray(SerializerIO rdb);
static SIValue _RdbLoadMap (SerializerIO rdb, SIType t);
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
	case T_MAP:
		return _RdbLoadMap(rdb, t);
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

static SIValue _RdbLoadMap
(
	SerializerIO rdb,
	SIType t
) {
	// loads map as
	// unsinged : key count
	// key:val
	// .
	// .
	// .
	// key:val

	uint n = SerializerIO_ReadUnsigned(rdb);
	char*   keys[n];
	SIValue vals[n];

	for(uint i = 0; i < n; i++) {
		keys[i] = _RdbLoadSIValue(rdb).stringval;
		vals[i] = _RdbLoadSIValue(rdb);
	}

	return Map_FromArrays(keys, vals, n);
}

static void _RdbLoadEntity
(
	SerializerIO rdb,
	GraphContext *gc,
	GraphEntity *e
) {
	// Format:
	// #properties N
	// (name, value type, value) X N

	uint64_t n = SerializerIO_ReadUnsigned(rdb);
	SIValue vals[n];
	AttributeID ids[n];

	for(int i = 0; i < n; i++) {
		ids[i]  = SerializerIO_ReadUnsigned(rdb);
		vals[i] = _RdbLoadSIValue(rdb);
	}

	AttributeSet_AddNoClone(e->attributes, ids, vals, n, false);
}

void RdbLoadNodes_v14
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t node_count
) {
	// Node Format:
	//      ID
	//      #labels M
	//      (labels) X M
	//      #properties N
	//      (name, value type, value) X N

	for(uint64_t i = 0; i < node_count; i++) {
		Node n;
		NodeID id = SerializerIO_ReadUnsigned(rdb);

		// #labels M
		uint64_t nodeLabelCount = SerializerIO_ReadUnsigned(rdb);

		// * (labels) x M
		LabelID labels[nodeLabelCount];
		for(uint64_t i = 0; i < nodeLabelCount; i ++){
			labels[i] = SerializerIO_ReadUnsigned(rdb);
		}

		Serializer_Graph_SetNode(gc->g, id, labels, nodeLabelCount, &n);

		_RdbLoadEntity(rdb, gc, (GraphEntity *)&n);

		// introduce n to each relevant index
		for (int i = 0; i < nodeLabelCount; i++) {
			Schema *s = GraphContext_GetSchemaByID(gc, labels[i], SCHEMA_NODE);
			ASSERT(s != NULL);

			if(PENDING_IDX(s)) Index_IndexNode(PENDING_IDX(s), &n);
		}
	}
}

void RdbLoadDeletedNodes_v14
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t deleted_node_count
) {
	// Format:
	// node id X N
	for(uint64_t i = 0; i < deleted_node_count; i++) {
		NodeID id = SerializerIO_ReadUnsigned(rdb);
		Serializer_Graph_MarkNodeDeleted(gc->g, id);
	}
}

void RdbLoadEdges_v14
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t edge_count
) {
	// Format:
	// {
	//  edge ID
	//  source node ID
	//  destination node ID
	//  relation type
	// } X N
	// edge properties X N

	// construct connections
	for(uint64_t i = 0; i < edge_count; i++) {
		Edge e;
		EdgeID    edgeId   = SerializerIO_ReadUnsigned(rdb);
		NodeID    srcId    = SerializerIO_ReadUnsigned(rdb);
		NodeID    destId   = SerializerIO_ReadUnsigned(rdb);
		uint64_t  relation = SerializerIO_ReadUnsigned(rdb);

		Serializer_Graph_SetEdge(gc->g, gc->decoding_context->multi_edge[relation],
			edgeId, srcId, destId, relation, &e);
		_RdbLoadEntity(rdb, gc, (GraphEntity *)&e);

		// index edge
		Schema *s = GraphContext_GetSchemaByID(gc, relation, SCHEMA_EDGE);
		ASSERT(s != NULL);

		if(PENDING_IDX(s)) Index_IndexEdge(PENDING_IDX(s), &e);
	}
}

void RdbLoadDeletedEdges_v14
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t deleted_edge_count
) {
	// Format:
	// edge id X N
	for(uint64_t i = 0; i < deleted_edge_count; i++) {
		EdgeID id = SerializerIO_ReadUnsigned(rdb);
		Serializer_Graph_MarkEdgeDeleted(gc->g, id);
	}
}
