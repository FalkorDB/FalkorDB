/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v19.h"
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
		// stop appending on a short read; a partial array is a valid SIValue and
		// the whole graph is torn down once the abort propagates
		if(SerializerIO_Error(rdb)) {
			break;
		}
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
	if (unlikely (buffer_size % sizeof(float) != 0)) {
		rm_free (buffer) ;
		SerializerIO_SetError (rdb, "vector size is not a multiple of float") ;
		return SI_NullVal () ;
	}

	SIValue vector = { .type       = T_VECTOR_F32,
					   .ptrval     = buffer,
					   .allocation = M_SELF };
	return vector;
}

#define ENTITY_PROP_STACK_THRESHOLD 256

// every decoded property value must be of a storable type (an unknown type
// tag decodes as NULL); fails the decode otherwise
static bool _ValidValues
(
	SerializerIO rdb,     // RDB
	const SIValue *vals,  // decoded values
	uint64_t n            // number of values
) {
	for (uint64_t i = 0 ; i < n ; i++) {
		if (unlikely (!(SI_TYPE (vals [i]) & SI_VALID_PROPERTY_VALUE))) {
			SerializerIO_SetError (rdb, "invalid property value type") ;
			return false ;
		}
	}
	return true ;
}

// free decoded property values that will not be attached to an entity
static void _FreeValues
(
	SIValue *vals,  // values to free
	uint64_t n      // number of values
) {
	for (uint64_t i = 0 ; i < n ; i++) {
		SIValue_Free (vals [i]) ;
	}
}

static void _RdbLoadEntity
(
	SerializerIO rdb,
	GraphEntity *e
) {
	// format:
	// #properties N
	// (name, value type, value) X N

	uint64_t n = SerializerIO_ReadUnsigned (rdb) ;

	if (n == 0) {
		return ;
	}

	if (unlikely (n > UINT16_MAX)) {
		SerializerIO_SetError (rdb, "entity has too many properties") ;
		return ;
	}

	// small path: all storage lives on the stack, no allocation needed
	if (likely (n <= ENTITY_PROP_STACK_THRESHOLD)) {
		SIValue     vals [n] ;
		AttributeID ids  [n] ;

		for (uint64_t i = 0 ; i < n ; i++) {
			ids  [i] = SerializerIO_ReadUnsigned (rdb) ;
			vals [i] = _RdbLoadSIValue (rdb) ;
		}

		// short read mid-entity: the remaining slots hold zeroed ids and
		// NULL values, which AttributeSet_Add rejects; drop the entity
		if (unlikely (SerializerIO_Error (rdb) || !_ValidValues (rdb, vals, n))) {
			_FreeValues (vals, n) ;
			return ;
		}

		AttributeSet_Add (e->attributes, ids, vals, n, false) ;
		return ;
	}

	// large path: heap allocation required
	SIValue     *vals = rm_malloc (n * sizeof (SIValue)) ;
	AttributeID *ids  = rm_malloc (n * sizeof (AttributeID)) ;

	// rm_malloc aborts on failure, so no null-check is needed;
	// remove this comment if that assumption ever changes

	for (uint64_t i = 0 ; i < n ; i++) {
		ids  [i] = SerializerIO_ReadUnsigned (rdb) ;
		vals [i] = _RdbLoadSIValue (rdb) ;
	}

	// short read mid-entity, see above
	if (unlikely (SerializerIO_Error (rdb) || !_ValidValues (rdb, vals, n))) {
		_FreeValues (vals, n) ;
		rm_free (ids) ;
		rm_free (vals) ;
		return ;
	}

	AttributeSet_Add (e->attributes, ids, vals, n, false) ;

	rm_free (ids) ;
	rm_free (vals) ;
}

// decode nodes
void RdbLoadNodes_v19
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

		// abort on a short read before mutating the datablock with a bogus id
		if(SerializerIO_Error(rdb)) {
			return;
		}

		// the datablock was sized from the header's node counts; a valid id
		// never makes it grow
		if(id >= g->nodes->itemCap) {
			SerializerIO_SetError(rdb, "node id out of range");
			return;
		}

		AttributeSet *set = DataBlock_AllocateItemOutOfOrder(g->nodes, id);
		*set = NULL;

		n.id = id;
		n.attributes = set;

		_RdbLoadEntity(rdb, (GraphEntity *)&n);
	}

	// validate node count (meaningless after a short read)
	if(!SerializerIO_Error(rdb) &&
	   n + prev_graph_node_count != Graph_NodeCount(g)) {
		SerializerIO_SetError(rdb, "decoded node count mismatch");
	}
}

// decode deleted nodes
void RdbLoadDeletedNodes_v19
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

	// abort on a short read before validating the (empty) buffer against the
	// expected count, which would otherwise trip the assert below
	if (SerializerIO_Error(rdb)) {
		rm_free(deleted_nodes_list);
		return;
	}

	// validate buffer: must be aligned and match expected count
	if (n % sizeof(NodeID) != 0 ||
		n / sizeof(NodeID) != deleted_node_count) {
		rm_free(deleted_nodes_list);
		RedisModule_Log(NULL, "warning",
			"Malformed RDB: deleted nodes buffer size %zu "
			"is not aligned or does not match expected count %" PRIu64,
			n, deleted_node_count);
		SerializerIO_SetError(rdb, "deleted nodes buffer size mismatch");
		return;
	}

	// mark each node id as deleted
	for(uint64_t i = 0; i < deleted_node_count; i++) {
		NodeID id = deleted_nodes_list[i];
		if(id >= g->nodes->itemCap) {
			rm_free(deleted_nodes_list);
			SerializerIO_SetError(rdb, "deleted node id out of range");
			return;
		}
		Serializer_Graph_MarkNodeDeleted(g, id);
	}
	rm_free(deleted_nodes_list);

	// validate deleted node count is as expected
	if(deleted_node_count + prev_deleted_node_count !=
	   Graph_DeletedNodeCount(g)) {
		SerializerIO_SetError(rdb, "deleted node count mismatch");
	}
}

// decode edges
void RdbLoadEdges_v19
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

		// abort on a short read before mutating the datablock with a bogus id
		if(SerializerIO_Error(rdb)) {
			return;
		}

		// the datablock was sized from the header's edge counts; a valid id
		// never makes it grow
		if(id >= g->edges->itemCap) {
			SerializerIO_SetError(rdb, "edge id out of range");
			return;
		}

		AttributeSet *set = DataBlock_AllocateItemOutOfOrder(g->edges, id);
		*set = NULL;

		e.id = id;
		e.attributes = set;

		_RdbLoadEntity(rdb, (GraphEntity *)&e);
	}

	// validate edge count (meaningless after a short read)
	if(!SerializerIO_Error(rdb) &&
	   n + prev_edge_count != Graph_EdgeCount(g)) {
		SerializerIO_SetError(rdb, "decoded edge count mismatch");
	}
}

// decode deleted edges
void RdbLoadDeletedEdges_v19
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

	// abort on a short read before validating the (empty) buffer against the
	// expected count, which would otherwise trip the assert below
	if (SerializerIO_Error(rdb)) {
		rm_free(deleted_edges_list);
		return;
	}

	// validate buffer: must be aligned and match expected count
	if (n % sizeof(EdgeID) != 0 ||
		n / sizeof(EdgeID) != deleted_edge_count) {
		rm_free(deleted_edges_list);
		RedisModule_Log(NULL, "warning",
			"Malformed RDB: deleted edges buffer size %zu "
			"is not aligned or does not match expected count %" PRIu64,
			n, deleted_edge_count);
		SerializerIO_SetError(rdb, "deleted edges buffer size mismatch");
		return;
	}

	// mark each edge id as deleted
	for(uint64_t i = 0; i < deleted_edge_count; i++) {
		EdgeID id = deleted_edges_list[i];
		if(id >= g->edges->itemCap) {
			rm_free(deleted_edges_list);
			SerializerIO_SetError(rdb, "deleted edge id out of range");
			return;
		}
		Serializer_Graph_MarkEdgeDeleted(g, id);
	}
	rm_free(deleted_edges_list);

	// validate deleted edge count is as expected
	if(deleted_edge_count + prev_deleted_edge_count !=
	   Graph_DeletedEdgeCount(g)) {
		SerializerIO_SetError(rdb, "deleted edge count mismatch");
	}
}

