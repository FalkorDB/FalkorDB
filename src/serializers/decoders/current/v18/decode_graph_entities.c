/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v18.h"
#include "util/datablock/oo_datablock.h"

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

	SIValue v;
	char *str;
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

	case T_INTERN_STRING:
		// create intern string and free loaded buffer
		str = SerializerIO_ReadBuffer(rdb, NULL);
		v = SI_InternStringVal(str);
		rm_free(str);
		return v;

	case T_BOOL:
		return SI_BoolVal(SerializerIO_ReadSigned(rdb));

	case T_ARRAY:
		return _RdbLoadSIArray(rdb);

	case T_POINT:
		return _RdbLoadPoint(rdb);

	case T_VECTOR_F32:
		return _RdbLoadVector(rdb, t);

	case T_TIME:
		return SI_Time(SerializerIO_ReadSigned(rdb));

	case T_DATE:
		return SI_Date(SerializerIO_ReadSigned(rdb));

	case T_DATETIME:
		return SI_DateTime(SerializerIO_ReadSigned(rdb));

	case T_DURATION:
		return SI_Duration(SerializerIO_ReadSigned(rdb));

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
		SIArray_AppendAsOwner (&list, &elem) ;
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

	size_t buffer_size;
	void *buffer = SerializerIO_ReadBuffer(rdb, &buffer_size);

	// validate buffer size is divisible by float
	ASSERT (buffer_size % sizeof(float) == 0) ;

	SIValue vector = { .type       = T_VECTOR_F32,
					   .ptrval     = buffer,
					   .allocation = M_SELF };
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

	AttributeSet_Add (e->attributes, ids, vals, n, false) ;
}

// decode nodes
void RdbLoadNodes_v18
(
	SerializerIO rdb, // RDB
	Graph *g,         // graph context
	const uint64_t n  // number of nodes to decode
) {
	// format:
	//  ID
	//  #properties N
	//  (name, value type, value) X N

	uint64_t prev_graph_node_count = Graph_NodeCount(g);

	for(uint64_t i = 0; i < n; i++) {
		Node n;
		NodeID id = SerializerIO_ReadUnsigned(rdb);

		AttributeSet *set = DataBlock_AllocateItemOutOfOrder(g->nodes, id);
		*set = NULL;

		n.id = id;
		n.attributes = set;

		_RdbLoadEntity(rdb, (GraphEntity *)&n);
	}

	// read encoded node count and validate
	ASSERT(n + prev_graph_node_count == Graph_NodeCount(g));
}

// decode deleted nodes
void RdbLoadDeletedNodes_v18
(
	SerializerIO rdb,                  // RDB
	Graph *g,                          // graph context
	const uint64_t deleted_node_count  // number of deleted nodes
) {
	// Format:
	// node ids

	uint64_t prev_deleted_node_count = Graph_DeletedNodeCount(g);

	// read node deleted IDs list from the RDB
	size_t n;
	NodeID *deleted_nodes_list = (NodeID*)SerializerIO_ReadBuffer(rdb, &n);

	// validate buffer alignment
	if (n % sizeof(NodeID) != 0) {
		rm_free (deleted_nodes_list) ;
		ASSERT (false && "corrupted deleted nodes buffer") ;
	}

	ASSERT((n / sizeof(NodeID)) == deleted_node_count);

	// mark each node id as deleted
	for(uint64_t i = 0; i < deleted_node_count; i++) {
		NodeID id = deleted_nodes_list[i];
		Serializer_Graph_MarkNodeDeleted(g, id);
	}
	rm_free(deleted_nodes_list);

	// validate deleted node count is as expected
	ASSERT(deleted_node_count + prev_deleted_node_count ==
			Graph_DeletedNodeCount(g));
}

// decode edges
void RdbLoadEdges_v18
(
	SerializerIO rdb,  // RDB
	Graph *g,          // graph context
	const uint64_t n   // number of edges to decode
) {
	// format:
	//  ID
	//  #properties N
	//  (name, value type, value) X N

	uint64_t prev_edge_count = Graph_EdgeCount(g); // #edges in the graph

	for(uint64_t i = 0; i < n; i++) {
		Edge e;

		EdgeID id = SerializerIO_ReadUnsigned(rdb);

		AttributeSet *set = DataBlock_AllocateItemOutOfOrder(g->edges, id);
		*set = NULL;

		e.id = id;
		e.attributes = set;

		_RdbLoadEntity(rdb, (GraphEntity *)&e);
	}

	// read encoded edge count and validate
	ASSERT(n + prev_edge_count == Graph_EdgeCount(g));
}

// decode deleted edges
void RdbLoadDeletedEdges_v18
(
	SerializerIO rdb,                  // RDB
	Graph *g,                          // graph context
	const uint64_t deleted_edge_count  // number of deleted edges
) {
	// Format:
	// edge ids

	uint64_t prev_deleted_edge_count = Graph_DeletedEdgeCount(g);

	// read edge deleted IDs list from the RDB
	size_t n;
	EdgeID *deleted_edges_list = (EdgeID*)SerializerIO_ReadBuffer(rdb, &n);

	// validate buffer alignment
	if (n % sizeof(EdgeID) != 0) {
		rm_free(deleted_edges_list);
		ASSERT(false && "corrupted deleted edges buffer");
	}

	ASSERT((n / sizeof(EdgeID)) == deleted_edge_count);

	// mark each edge id as deleted
	for(uint64_t i = 0; i < deleted_edge_count; i++) {
		EdgeID id = deleted_edges_list[i];
		Serializer_Graph_MarkEdgeDeleted(g, id);
	}
	rm_free(deleted_edges_list);

	// validate deleted edge count is as expected
	ASSERT(deleted_edge_count + prev_deleted_edge_count ==
			Graph_DeletedEdgeCount(g));
}

