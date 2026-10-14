/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v20.h"
#include "util/datablock/oo_datablock.h"

// forward declarations
static SIValue _RdbLoadPoint(SerializerIO rdb);
static SIValue _RdbLoadSIArray(SerializerIO rdb);
static SIValue _RdbLoadVector(SerializerIO rdb, SIType t);

// decode a single SIValue
// 'allowed' is the set of value types acceptable in the current context:
// entity properties accept SI_VALID_PROPERTY_VALUE, array elements additionally
// accept T_NULL; any other tag fails the decode
static SIValue _RdbLoadSIValue
(
	SerializerIO rdb,  // RDB
	SIType allowed     // accepted value types
) {
	// Format:
	// SIType
	// Value

	SIValue v;
	char *str;
	SIType t = SerializerIO_TryReadUnsigned(rdb);

	// reject any tag not accepted in this context (also rejects the 0 tag a short
	// read would leave behind); a known type that simply isn't storable here is
	// distinguished from a tag no SIType has
	if(unlikely (!(t & allowed))) {
		if(t & SI_ALL) {
			SerializerIO_SetError(rdb, "invalid property value type");
		} else {
			SerializerIO_SetError(rdb, EMSG_VALUE_UNKNOWN_TYPE,
					(unsigned long long) t);
		}
		return SI_NullVal();
	}

	switch(t) {
		case T_INT64:
			return SI_LongVal(SerializerIO_TryReadSigned(rdb));

		case T_DOUBLE:
			return SI_DoubleVal(SerializerIO_TryReadDouble(rdb));

		case T_STRING:
			// transfer ownership of the heap-allocated string to the
			// newly-created SIValue
			str = SerializerIO_TryReadCString(rdb);
			// unarmed (RESTORE/COPY/offload): a short read latches the error and
			// returns NULL instead of jumping; don't hand NULL to the SIValue
			if(unlikely(str == NULL)) return SI_NullVal();
			return SI_TransferStringVal(str);

		case T_INTERN_STRING:
			// create intern string and free loaded buffer
			str = SerializerIO_TryReadCString(rdb);
			// unarmed: a short read returns NULL; SI_InternStringVal would crash
			if(unlikely(str == NULL)) return SI_NullVal();
			v = SI_InternStringVal(str);
			rm_free(str);
			return v;

		case T_BOOL:
			return SI_BoolVal(SerializerIO_TryReadSigned(rdb));

		case T_ARRAY:
			return _RdbLoadSIArray(rdb);

		case T_POINT:
			return _RdbLoadPoint(rdb);

		case T_VECTOR_F32:
			return _RdbLoadVector(rdb, t);

		case T_TIME:
			return SI_Time(SerializerIO_TryReadSigned(rdb));

		case T_DATE:
			return SI_Date(SerializerIO_TryReadSigned(rdb));

		case T_DATETIME:
			return SI_DateTime(SerializerIO_TryReadSigned(rdb));

		case T_DURATION:
			return SI_Duration(SerializerIO_TryReadSigned(rdb));

		case T_NULL:
			return SI_NullVal();

		default:
			SerializerIO_SetError(rdb, EMSG_VALUE_UNKNOWN_TYPE,
					(unsigned long long) t);
			return SI_NullVal();
	}
}

static SIValue _RdbLoadPoint
(
	SerializerIO rdb
) {
	double lat = SerializerIO_TryReadDouble(rdb);
	double lon = SerializerIO_TryReadDouble(rdb);
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
	uint64_t arrayLen = SerializerIO_TryReadUnsigned(rdb);

	// guard an absurd element count before it drives the loop below; a genuinely
	// truncated payload would short-read before this many elements. SetError
	// longjmps to the landing pad, so the returned placeholder is unreachable
	if(unlikely(arrayLen > GRAPH_MAX_SIARRAY_LEN)) {
		SerializerIO_SetError(rdb, "array length %llu exceeds limit",
				(unsigned long long) arrayLen);
		return SI_NullVal();
	}

	// reserve only a modest initial capacity, never the full claimed length, so a
	// bogus-large length can't force a huge up-front allocation; the array grows
	// as real elements arrive, and a short read aborts to the landing pad
	SIValue list = SI_Array(arrayLen < 1024 ? arrayLen : 1024);
	for(uint64_t i = 0; i < arrayLen; i++) {
		SIValue elem = _RdbLoadSIValue(rdb, SI_VALID_PROPERTY_VALUE | T_NULL);
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
	void *buffer = SerializerIO_TryReadBuffer(rdb, &buffer_size);

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

static void _RdbLoadEntity
(
	SerializerIO rdb,
	GraphEntity *e
) {
	// format:
	// #properties N
	// (name, value type, value) X N

	uint64_t n = SerializerIO_TryReadUnsigned (rdb) ;

	if (n == 0) {
		return ;
	}

	// an entity can't hold more distinct properties than the graph has attributes
	if (unlikely (n > GRAPH_MAX_ATTRIBUTES)) {
		SerializerIO_SetError (rdb, "entity has too many properties") ;
		return ;  // unreachable: SetError longjmps
	}

	// property ids and values are read through the TryRead API with the abort
	// handler armed: a short read or a malformed value (SetError) longjmps
	// straight to the landing pad. the in-progress ids/vals buffers and the
	// values decoded so far are abandoned by that jump; this is a one-shot leak
	// on a fatal, aborting load, accepted to keep the handler armed throughout

	// small path: all storage lives on the stack, no allocation needed
	if (likely (n <= ENTITY_PROP_STACK_THRESHOLD)) {
		SIValue     vals [n] ;
		AttributeID ids  [n] ;

		for (uint64_t i = 0 ; i < n ; i++) {
			ids  [i] = SerializerIO_TryReadUnsigned (rdb) ;
			vals [i] = _RdbLoadSIValue (rdb, SI_VALID_PROPERTY_VALUE) ;
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
		ids  [i] = SerializerIO_TryReadUnsigned (rdb) ;
		vals [i] = _RdbLoadSIValue (rdb, SI_VALID_PROPERTY_VALUE) ;
	}

	AttributeSet_Add (e->attributes, ids, vals, n, false) ;

	rm_free (ids) ;
	rm_free (vals) ;
}

// decode nodes
void RdbLoadNodes_v20
(
	SerializerIO rdb,        // RDB
	Graph *g,                // graph context
	const uint64_t n,        // number of nodes to decode
	const uint64_t id_limit  // node ids are below this (header counts)
) {
	// format:
	//  ID
	//  #properties N
	//  (name, value type, value) X N

	uint64_t prev_graph_node_count = Graph_NodeCount(g);

	for(uint64_t i = 0; i < n; i++) {
		Node node;
		NodeID id = SerializerIO_TryReadUnsigned(rdb);

		// ids run below the header's live + deleted node count; a short read
		// would have already jumped to the landing pad
		if (unlikely (id >= id_limit)) {
			SerializerIO_SetError(rdb, "node id %llu out of range",
					(unsigned long long) id);
			return;  // unreachable: SetError longjmps
		}

		AttributeSet *set = DataBlock_AllocateItemOutOfOrder(g->nodes, id);
		*set = NULL;

		node.id = id;
		node.attributes = set;

		_RdbLoadEntity(rdb, (GraphEntity *)&node);
	}

	// every node decoded (a short read would have jumped); the count must match
	// the header
	if(n + prev_graph_node_count != Graph_NodeCount(g)) {
		SerializerIO_SetError(rdb, "decoded node count mismatch");
	}
}

// decode deleted nodes
void RdbLoadDeletedNodes_v20
(
	SerializerIO rdb,                   // RDB
	Graph *g,                           // graph context
	const uint64_t deleted_node_count,  // number of deleted nodes
	const uint64_t id_limit             // node ids are below this
) {
	// Format:
	// node ids

	uint64_t prev_deleted_node_count = Graph_DeletedNodeCount(g);

	// read node deleted IDs list from the RDB
	size_t n;
	NodeID *deleted_nodes_list = (NodeID*)SerializerIO_TryReadBuffer(rdb, &n);

	// a short read would have jumped to the landing pad; the buffer read here is
	// complete and can be validated against the expected count

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
		if(unlikely(id >= id_limit)) {
			rm_free(deleted_nodes_list);
			SerializerIO_SetError(rdb, "deleted node id %llu out of range",
					(unsigned long long) id);
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
void RdbLoadEdges_v20
(
	SerializerIO rdb,        // RDB
	Graph *g,                // graph context
	const uint64_t n,        // number of edges to decode
	const uint64_t id_limit  // edge ids are below this (header counts)
) {
	// format:
	//  ID
	//  #properties N
	//  (name, value type, value) X N

	uint64_t prev_edge_count = Graph_EdgeCount(g); // #edges in the graph

	for(uint64_t i = 0; i < n; i++) {
		Edge e;

		EdgeID id = SerializerIO_TryReadUnsigned(rdb);

		// ids run below the header's live + deleted edge count; a short read
		// would have already jumped to the landing pad
		if(unlikely (id >= id_limit)) {
			SerializerIO_SetError(rdb, "edge id %llu out of range",
					(unsigned long long) id);
			return;  // unreachable: SetError longjmps
		}

		AttributeSet *set = DataBlock_AllocateItemOutOfOrder(g->edges, id);
		*set = NULL;

		e.id = id;
		e.attributes = set;

		_RdbLoadEntity(rdb, (GraphEntity *)&e);
	}

	// every edge decoded (a short read would have jumped); the count must match
	// the header
	if(n + prev_edge_count != Graph_EdgeCount(g)) {
		SerializerIO_SetError(rdb, "decoded edge count mismatch");
	}
}

// decode deleted edges
void RdbLoadDeletedEdges_v20
(
	SerializerIO rdb,                   // RDB
	Graph *g,                           // graph context
	const uint64_t deleted_edge_count,  // number of deleted edges
	const uint64_t id_limit             // edge ids are below this
) {
	// Format:
	// edge ids

	uint64_t prev_deleted_edge_count = Graph_DeletedEdgeCount(g);

	// read edge deleted IDs list from the RDB
	size_t n;
	EdgeID *deleted_edges_list = (EdgeID*)SerializerIO_TryReadBuffer(rdb, &n);

	// a short read would have jumped to the landing pad; the buffer read here is
	// complete and can be validated against the expected count

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
		if(unlikely(id >= id_limit)) {
			rm_free(deleted_edges_list);
			SerializerIO_SetError(rdb, "deleted edge id %llu out of range",
					(unsigned long long) id);
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

