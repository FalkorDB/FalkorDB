/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v17.h"
#include "util/datablock/oo_datablock.h"

// forward declarations
static bool _RdbLoadPoint(SerializerIO io, SIValue *value);
static bool _RdbLoadSIArray(SerializerIO io, SIValue *value);
static bool _RdbLoadVector(SerializerIO rdb, SIValue *value, SIType t);

static bool _RdbLoadSIValue
(
	SerializerIO io,
	SIValue *value
) {
	// Format:
	// SIType
	// Value

	SIType t;
	uint64_t type;

	TRY_READ(io, type);
	t = type;

	switch(t) {
		case T_INT64: {
				int64_t x;
				TRY_READ(io, x);
				*value = SI_LongVal(x);
				break;
			}

		case T_DOUBLE: {
				double x;
				TRY_READ(io, x);
				*value = SI_DoubleVal(x);
				break;
			}

		case T_STRING: {
				// transfer ownership of the heap-allocated string to the
				// newly-created SIValue
				char *x;
				if(!SerializerIO_ReadBuffer(io, &x, NULL)) {
					return false;
				}
				*value = SI_TransferStringVal(x);
				break;
			}

		case T_INTERN_STRING: {
				// create intern string and free loaded buffer
				char *x;
				if(!SerializerIO_ReadBuffer(io, &x, NULL)) {
					return false;
				}
				*value = SI_InternStringVal(x);
				rm_free(x);
				break;
			}

		case T_BOOL: {
				bool x;
				int64_t b;

				TRY_READ(io, b);
				x = b;

				*value = SI_BoolVal(x);
				break;
			}

		case T_ARRAY:
			if (!_RdbLoadSIArray(io, value)) {
				return false;
			}
			break;

		case T_POINT:
			if (!_RdbLoadPoint(io, value)) {
				return false;
			}
			break;

		case T_VECTOR_F32:
			if (!_RdbLoadVector(io, value, t)) {
				return false;
			}
			break;

		case T_NULL:
			*value = SI_NullVal();
			break;

		default: // currently impossible
			assert(false && "unknown value type");
			return false;
	}

	return true;
}

static bool _RdbLoadPoint
(
	SerializerIO io,
	SIValue *value
) {
	double lat;
	double lon;

	TRY_READ(io, lat);
	TRY_READ(io, lon);

	*value = SI_Point(lat, lon);
	return true;
}

static bool _RdbLoadSIArray
(
	SerializerIO io,
	SIValue *value
) {
	/* loads array as
	   unsinged : array legnth
	   array[0]
	   .
	   .
	   .
	   array[array length -1]
	 */
	uint64_t arrayLen;
	TRY_READ(io, arrayLen);
	*value = SI_Array(arrayLen);
	for (uint i = 0; i < arrayLen; i++) {
		SIValue elem;
		if (!_RdbLoadSIValue(io, &elem)) {
			SIArray_Free(*value);
			return false;
		}

		SIArray_AppendAsOwner(value, &elem);
	}
	return true;
}

static bool _RdbLoadVector
(
	SerializerIO io,
	SIValue *value,
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

	char *elements;
	if (!SerializerIO_ReadBuffer(io, &elements, NULL)) {
		return false;
	}

	*value = (SIValue) { .type       = T_VECTOR_F32,
						 .ptrval     = elements,
						 .allocation = M_SELF };

	return true;
}

static bool _RdbLoadEntity
(
	SerializerIO io,
	GraphEntity *e
) {
	// format:
	// #properties N
	// (name, value type, value) X N

	uint64_t n;
	TRY_READ(io, n);

	if(n == 0) return true;  // no attributes

	SIValue     vals[n];
	AttributeID ids [n];

	for(uint64_t i = 0; i < n; i++) {
		uint64_t id;
		if (!SerializerIO_ReadUnsigned(io, &id)) {
			return false;
		}
		ids[i] = id;

		if (!_RdbLoadSIValue(io, &vals[i])) {
			return false;
		}
	}

	AttributeSet_AddNoClone(e->attributes, ids, vals, n, false);

	return true;
}

// decode nodes
bool RdbLoadNodes_v17
(
	SerializerIO io,  // RDB
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
		NodeID id;

		if (!SerializerIO_ReadUnsigned(io, &id)) {
			return false;
		}

		AttributeSet *set = DataBlock_AllocateItemOutOfOrder(g->nodes, id);
		*set = NULL;

		n.id = id;
		n.attributes = set;

		if (!_RdbLoadEntity(io, (GraphEntity *)&n)) {
			return false;
		}
	}

	// read encoded node count and validate
	ASSERT(n + prev_graph_node_count == Graph_NodeCount(g));

	return true;
}

// decode deleted nodes
bool RdbLoadDeletedNodes_v17
(
	SerializerIO io,                   // RDB
	Graph *g,                          // graph context
	const uint64_t deleted_node_count  // number of deleted nodes
) {
	// Format:
	// node ids

	uint64_t prev_deleted_node_count = Graph_DeletedNodeCount(g);

	// read node deleted IDs list from the RDB
	size_t n;
	NodeID *deleted_nodes_list;
	if (!SerializerIO_ReadBuffer(io, (char**)&deleted_nodes_list, &n)) {
		return false;
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

	return true;
}

// decode edges
bool RdbLoadEdges_v17
(
	SerializerIO io,  // RDB
	Graph *g,         // graph context
	const uint64_t n  // number of edges to decode
) {
	// format:
	//  ID
	//  #properties N
	//  (name, value type, value) X N

	uint64_t prev_edge_count = Graph_EdgeCount(g); // #edges in the graph

	for (uint64_t i = 0; i < n; i++) {
		Edge e;
		EdgeID id;

		if (!SerializerIO_ReadUnsigned(io, &id)) {
			return false;
		}

		AttributeSet *set = DataBlock_AllocateItemOutOfOrder(g->edges, id);
		*set = NULL;

		e.id = id;
		e.attributes = set;

		if (!_RdbLoadEntity(io, (GraphEntity *)&e)) {
			return false;
		}
	}

	// read encoded edge count and validate
	ASSERT(n + prev_edge_count == Graph_EdgeCount(g));
	return true;
}

// decode deleted edges
bool RdbLoadDeletedEdges_v17
(
	SerializerIO io,                   // RDB
	Graph *g,                          // graph context
	const uint64_t deleted_edge_count  // number of deleted edges
) {
	// Format:
	// edge ids

	uint64_t prev_deleted_edge_count = Graph_DeletedEdgeCount(g);

	// read edge deleted IDs list from the RDB
	size_t n;
	EdgeID *deleted_edges_list;
	if (!SerializerIO_ReadBuffer(io, (char**)&deleted_edges_list, &n)) {
		return false;
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

	return true;
}

