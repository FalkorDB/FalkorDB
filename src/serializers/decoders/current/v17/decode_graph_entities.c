/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v17.h"
#include "util/datablock/oo_datablock.h"
#include "util/rocksdb.h"

// forward declarations
static SIValue _RdbLoadPoint(SerializerIO rdb);
static SIValue _RdbLoadSIArray(SerializerIO rdb);
static SIValue _RdbLoadVector(SerializerIO rdb, SIType t);

static SIValue _RdbLoadSIValue
(
	SerializerIO rdb,
	NodeID node_id,
	AttributeID attr_id,
	rocksdb_writebatch_t *writebatch
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
	case T_STRING: {
		// transfer ownership of the heap-allocated string to the
		// newly-created SIValue
		char *str = SerializerIO_ReadBuffer(rdb, NULL);
		if(writebatch && strnlen(str, ROCKSDB_MIN_STR_LEN) == ROCKSDB_MIN_STR_LEN) {
			char node_key[ROCKSDB_KEY_SIZE];
			RocksDB_set_key(node_key, node_id, attr_id);
			RocksDB_put(writebatch, node_key, str);
			rm_free(str);
			return (SIValue){.type = T_STRING, .stringval = NULL, .allocation = M_DISK};
		}
		return SI_TransferStringVal(str);
	}
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
		SIValue elem = _RdbLoadSIValue(rdb, -1, ATTRIBUTE_ID_NONE, NULL);
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

	SIValue vector = { .type       = T_VECTOR_F32,
					   .ptrval     = SerializerIO_ReadBuffer(rdb, NULL),
					   .allocation = M_SELF };
	return vector;
}

static void _RdbLoadEntity
(
	SerializerIO rdb,
	GraphEntity *e,
	rocksdb_writebatch_t *writebatch
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
		vals[i] = _RdbLoadSIValue(rdb, ENTITY_GET_ID(e), ids[i], writebatch);
	}

	AttributeSet_AddNoClone(e->attributes, ids, vals, n, false);
}

// decode nodes
void RdbLoadNodes_v17
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
	rocksdb_writebatch_t *batch = RocksDB_create_batch();
	for(uint64_t i = 0; i < n; i++) {
		Node n;
		NodeID id = SerializerIO_ReadUnsigned(rdb);

		AttributeSet *set = DataBlock_AllocateItemOutOfOrder(g->nodes, id);
		*set = NULL;

		n.id = id;
		n.attributes = set;

		_RdbLoadEntity(rdb, (GraphEntity *)&n, batch);
	}
	RocksDB_put_batch(batch);

	// read encoded node count and validate
	ASSERT(n + prev_graph_node_count == Graph_NodeCount(g));
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
void RdbLoadEdges_v17
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

		_RdbLoadEntity(rdb, (GraphEntity *)&e, NULL);
	}

	// read encoded edge count and validate
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

