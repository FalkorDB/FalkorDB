/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v16.h"

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
	SIType t = SerializerIO_TryReadUnsigned(rdb);

	// reject any tag not accepted in this context (also rejects the 0 tag a short
	// read would leave behind)
	if(unlikely (!(t & allowed))) {
		if(t & SI_ALL) {
			SerializerIO_SetError(rdb, EMSG_DECODE_INVALID_PROPERTY_VALUE);
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
	case T_STRING: {
		// transfer ownership of the heap-allocated string to the
		// newly-created SIValue
		char *str = SerializerIO_TryReadCString(rdb);
		// unarmed (offload): a short read latches the error and returns NULL
		// instead of jumping; don't hand NULL to the SIValue
		if(unlikely(str == NULL)) return SI_NullVal();
		return SI_TransferStringVal(str);
	}
	case T_BOOL:
		return SI_BoolVal(SerializerIO_TryReadSigned(rdb));
	case T_ARRAY:
		return _RdbLoadSIArray(rdb);
	case T_POINT:
		return _RdbLoadPoint(rdb);
	case T_VECTOR_F32:
		return _RdbLoadVector(rdb, t);
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

	// guard an absurd element count before it drives the loop below
	if(unlikely(arrayLen > GRAPH_MAX_SIARRAY_LEN)) {
		SerializerIO_SetError(rdb, EMSG_DECODE_LIMIT, "array length",
				(unsigned long long) arrayLen);
		return SI_NullVal();
	}

	// reserve only a modest initial capacity, never the full claimed length
	SIValue list = SI_Array(arrayLen < 1024 ? arrayLen : 1024);
	for(uint64_t i = 0; i < arrayLen; i++) {
		SIValue elem = _RdbLoadSIValue(rdb, SI_VALID_PROPERTY_VALUE | T_NULL);
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

	uint64_t dim = SerializerIO_TryReadUnsigned(rdb);

	// guard an absurd dimension before it sizes the allocation / drives the loop
	if(unlikely(dim > GRAPH_MAX_SIARRAY_LEN)) {
		SerializerIO_SetError(rdb, EMSG_DECODE_LIMIT, "vector dimension",
				(unsigned long long) dim);
		return SI_NullVal();
	}

	vector = SI_Vectorf32(dim);
	float *values = SIVector_Elements(vector);

	for(uint32_t i = 0; i < dim; i++) {
		values[i] = SerializerIO_TryReadFloat(rdb);
	}

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
		SerializerIO_SetError (rdb, EMSG_DECODE_ENTITY_PROPS) ;
		return ;  // unreachable: SetError longjmps
	}

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
void RdbLoadNodes_v16
(
	SerializerIO rdb,           // RDB
	GraphContext *gc,           // graph context
	const uint64_t node_count,  // number of nodes to decode
	const uint64_t id_limit     // node ids are below this (header counts)
) {
	// Node Format:
	//      ID
	//      #labels M
	//      (labels) X M
	//      #properties N
	//      (name, value type, value) X N

	Graph *g = GraphContext_GetGraph (gc) ;
	uint64_t label_count = Graph_LabelTypeCount (g) ;

	// get delay indexing configuration
	bool delay_indexing;
	Config_Option_get(Config_DELAY_INDEXING, &delay_indexing);

	uint64_t prev_graph_node_count = Graph_NodeCount (g) ;

	for(uint64_t i = 0; i < node_count; i++) {
		Node n;
		NodeID id = SerializerIO_TryReadUnsigned(rdb);

		// ids run below the header's live + deleted node count
		if(unlikely(id >= id_limit)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "node id",
					(unsigned long long) id);
			return;  // unreachable: SetError longjmps
		}

		// #labels M
		uint64_t nodeLabelCount = SerializerIO_TryReadUnsigned(rdb);

		// guard the per-node label count before it sizes the VLA below; a node
		// can't carry more labels than the graph defines
		if(unlikely(nodeLabelCount > label_count)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_LIMIT, "node label count",
					(unsigned long long) nodeLabelCount,
					(unsigned long long) label_count);
			return;  // unreachable: SetError longjmps
		}

		// (labels) x M
		LabelID labels[nodeLabelCount];
		for(uint64_t j = 0; j < nodeLabelCount; j ++){
			uint64_t l = SerializerIO_TryReadUnsigned(rdb);
			// labels index the schema array; reject an out-of-range id
			if(unlikely(l >= label_count)) {
				SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "node label",
						(unsigned long long) l);
				return;
			}
			labels[j] = l;
		}

		Serializer_Graph_SetNode (g, id, labels, nodeLabelCount, &n) ;

		_RdbLoadEntity (rdb, (GraphEntity *)&n) ;

		// introduce n to each relevant index
		if(!delay_indexing) {
			for (int j = 0; j < nodeLabelCount; j++) {
				Schema *s = GraphContext_GetSchemaByID(gc, labels[j],
						SCHEMA_NODE);
				ASSERT(s != NULL);

				// index node
				if(PENDING_IDX(s)) Index_IndexNode(PENDING_IDX(s), &n);
			}
		}
	}

	if(prev_graph_node_count + node_count != Graph_NodeCount (g)) {
		SerializerIO_SetError(rdb, EMSG_DECODE_COUNT_MISMATCH, "decoded node");
	}
}

// decode deleted nodes
void RdbLoadDeletedNodes_v16
(
	SerializerIO rdb,                   // RDB
	GraphContext *gc,                   // graph context
	const uint64_t deleted_node_count,  // number of deleted nodes
	const uint64_t id_limit             // node ids are below this
) {
	// Format:
	// node id X N

	Graph *g = GraphContext_GetGraph (gc) ;

	uint64_t prev_deleted_node_count = Graph_DeletedNodeCount (g) ;

	for(uint64_t i = 0; i < deleted_node_count; i++) {
		NodeID id = SerializerIO_TryReadUnsigned(rdb);
		if(unlikely(id >= id_limit)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "deleted node id",
					(unsigned long long) id);
			return;
		}
		Serializer_Graph_MarkNodeDeleted (g, id) ;
	}

	// read encoded deleted node count and validate
	if(deleted_node_count + prev_deleted_node_count !=
	   Graph_DeletedNodeCount (g)) {
		SerializerIO_SetError(rdb, EMSG_DECODE_COUNT_MISMATCH, "deleted node");
	}
}

// decode edge relationship header
static void _DecodeRelationHeader
(
	SerializerIO rdb,  // RDB
	RelationID *r,     // [output] relation id
	bool *tensor       // [output] tensor
) {
	// format:
	//
	// Header:
	//     relationship ID
	//     rather or not relationship contains tensors
	//

	ASSERT(r      != NULL);
	ASSERT(tensor != NULL);

	*r = SerializerIO_TryReadUnsigned(rdb);
	ASSERT(*r != GRAPH_NO_RELATION);

	*tensor = SerializerIO_TryReadUnsigned(rdb);
}

// decode tensors
static uint64_t _DecodeTensors
(
	SerializerIO rdb,              // RDB
	GraphContext *gc,             // graph context
	RelationID r,                 // relationship type
	const uint64_t node_id_limit, // src/dest node ids are below this
	const uint64_t edge_id_limit  // edge ids are below this
) {
	// format:
	// edge format:
	//     edge id
	//     source node id
	//     destination node id
	//     multi-edge
	//     edge properties

	Graph *g = GraphContext_GetGraph (gc) ;

	Edge e;                       // current decoded edge
	int       idx            = 0; // batch index
	uint64_t  decoded_edges  = 0; // number of decoded edges
	int       tensor_idx     = 0; // tensors batch index
	enum { BATCH_SIZE = 4096 };   // compile-time constant, avoids a VLA

	// single edge batch
	EdgeID ids  [BATCH_SIZE];
	NodeID srcs [BATCH_SIZE];
	NodeID dests[BATCH_SIZE];

	// tensors batch
	EdgeID tensors_ids  [BATCH_SIZE];
	NodeID tensors_srcs [BATCH_SIZE];
	NodeID tensors_dests[BATCH_SIZE];

	// get relationship type schema
	Schema *s = GraphContext_GetSchemaByID(gc, r, SCHEMA_EDGE);
	ASSERT(s != NULL);

	Index index = PENDING_IDX(s);

	// get delay indexing configuration
	bool delay_indexing;
	Config_Option_get(Config_DELAY_INDEXING, &delay_indexing);

	// perform indexing if there's an index and delay indexing is false
	bool perform_indexing = (!delay_indexing && index != NULL);

	// as long as we didn't hit our END MARKER
	while(true) {
		// decode edge ID
		e.id = SerializerIO_TryReadUnsigned(rdb);

		// break on END MARKER
		if(e.id == INVALID_ENTITY_ID) {
			break;
		}

		// edge id indexes the edge datablock; reject an out-of-range id before
		// it drives an unbounded allocation
		if(unlikely(e.id >= edge_id_limit)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "edge id",
					(unsigned long long) e.id);
			return 0;
		}

		// decode edge source node id
		e.src_id     = SerializerIO_TryReadUnsigned(rdb);
		// decode edge destination node id
		e.dest_id    = SerializerIO_TryReadUnsigned(rdb);
		// set edge relation id
		e.relationID = r;

		// src/dest form connections in the adjacency matrix; reject bad ids
		if(unlikely(e.src_id >= node_id_limit || e.dest_id >= node_id_limit)) {
			SerializerIO_SetError(rdb,
					EMSG_DECODE_ID_RANGE, "edge endpoint",
					(unsigned long long) e.src_id, (unsigned long long) e.dest_id);
			return 0;
		}

		// decode tensor marker
		bool tensor = SerializerIO_TryReadUnsigned(rdb);

		// load edge attributes
		Serializer_Graph_AllocEdgeAttributes (g, e.id, &e) ;
		_RdbLoadEntity (rdb, (GraphEntity *)&e) ;

		// index edge
		if(perform_indexing) {
			Index_IndexEdge(index, &e);
		}

		// batch edge
		if(tensor) {
			// batch tensor
			tensors_ids[tensor_idx]   = e.id;
			tensors_srcs[tensor_idx]  = e.src_id;
			tensors_dests[tensor_idx] = e.dest_id;
			tensor_idx++;  // advance batch index
		} else {
			// batch edge
			ids[idx]   = e.id;
			srcs[idx]  = e.src_id;
			dests[idx] = e.dest_id;
			idx++;  // advance batch index
		}

		//----------------------------------------------------------------------
		// flush batches
		//----------------------------------------------------------------------

		// flush tensors batch
		if(tensor_idx == BATCH_SIZE) {
			// flush batch
			Serializer_OptimizedFormConnections (g, r, tensors_srcs,
					tensors_dests, tensors_ids, tensor_idx, true) ;

			// reset multi-edge batch count
			tensor_idx = 0;
		}

		// flush batch
		if(idx == BATCH_SIZE) {
			// flush batch
			Serializer_OptimizedFormConnections (g, r, srcs, dests, ids, idx,
					false) ;

			// reset batch count
			idx = 0;
		}

		// update number of decode edges
		decoded_edges++;
	}

	//----------------------------------------------------------------------
	// flush batches
	//----------------------------------------------------------------------

	// flush tensors batch
	if(tensor_idx > 0) {
		// flush batch
		Serializer_OptimizedFormConnections (g, r, tensors_srcs,
				tensors_dests, tensors_ids, tensor_idx, true) ;
	}

	// flush batch
	if(idx > 0) {
		// flush batch
		Serializer_OptimizedFormConnections (g, r, srcs, dests, ids, idx,
				false) ;
	}

	return decoded_edges;
}

// decode edges
static uint64_t _DecodeEdges
(
	SerializerIO rdb,              // RDB
	GraphContext *gc,             // graph context
	RelationID r,                 // relationship type
	const uint64_t node_id_limit, // src/dest node ids are below this
	const uint64_t edge_id_limit  // edge ids are below this
) {
	// Format:
	//  edge ID
	//  source node ID
	//  destination node ID
	//  edge properties

	Graph *g = GraphContext_GetGraph (gc) ;

	Edge e;                        // current decoded edge
	uint64_t idx           = 0;    // batch index
	uint64_t decoded_edges = 0;    // number of decoded edges
	enum { BATCH_SIZE = 256 };     // compile-time constant, avoids a VLA

	// batch
	EdgeID ids  [BATCH_SIZE];
	NodeID srcs [BATCH_SIZE];
	NodeID dests[BATCH_SIZE];

	// get relationship type schema
	Schema *s = GraphContext_GetSchemaByID(gc, r, SCHEMA_EDGE);
	ASSERT(s != NULL);

	Index index = PENDING_IDX(s);

	// get delay indexing configuration
	bool delay_indexing;
	Config_Option_get(Config_DELAY_INDEXING, &delay_indexing);

	// perform indexing if there's an index and delay indexing is false
	bool perform_indexing = (!delay_indexing && index != NULL);

	// as long as we didn't hit our END MARKER
	while(true) {
		// decode edge ID
		e.id = SerializerIO_TryReadUnsigned(rdb);

		// break on END MARKER
		if(e.id == INVALID_ENTITY_ID) {
			break;
		}

		// edge id indexes the edge datablock; reject an out-of-range id before
		// it drives an unbounded allocation
		if(unlikely(e.id >= edge_id_limit)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "edge id",
					(unsigned long long) e.id);
			return 0;
		}

		// decode edge source node id
		e.src_id     = SerializerIO_TryReadUnsigned(rdb);
		// decode edge destination node id
		e.dest_id    = SerializerIO_TryReadUnsigned(rdb);
		// set edge relation id
		e.relationID = r;

		// src/dest form connections in the adjacency matrix; reject bad ids
		if(unlikely(e.src_id >= node_id_limit || e.dest_id >= node_id_limit)) {
			SerializerIO_SetError(rdb,
					EMSG_DECODE_ID_RANGE, "edge endpoint",
					(unsigned long long) e.src_id, (unsigned long long) e.dest_id);
			return 0;
		}

		// load edge attributes
		Serializer_Graph_AllocEdgeAttributes (g, e.id, &e) ;
		_RdbLoadEntity (rdb, (GraphEntity *)&e) ;

		// index edge
		if(perform_indexing) {
			Index_IndexEdge(index, &e);
		}

		// batch edge
		ids[idx]   = e.id;
		srcs[idx]  = e.src_id;
		dests[idx] = e.dest_id;
		idx++;  // advance batch index

		// flush batch
		if(idx == BATCH_SIZE) {
			Serializer_OptimizedFormConnections (g, r, srcs, dests, ids, idx,
					false) ;
			// reset batch index
			idx = 0;
		}

		// update number of decode edges
		decoded_edges++;
	}

	// flush batch
	if(idx > 0) {
		Serializer_OptimizedFormConnections (g, r, srcs, dests, ids, idx,
				false) ;
	}

	return decoded_edges;
}

// decode edges
void RdbLoadEdges_v16
(
	SerializerIO rdb,             // RDB
	GraphContext *gc,             // graph context
	const uint64_t n,             // number of edges to decode
	const uint64_t node_id_limit, // src/dest node ids are below this
	const uint64_t edge_id_limit  // edge ids are below this
) {
	// Format:
	// {
	//  edge ID
	//  source node ID
	//  destination node ID
	//  relation type
	//  multi-edge [only when relationship type has tensors]
	// } X N
	// edge properties X N

	Graph *g = GraphContext_GetGraph (gc) ;
	uint64_t rel_count = Graph_RelationTypeCount (g) ;

	bool tensor;
	RelationID r;
	uint64_t decoded_edges   = 0;
	uint64_t prev_edge_count = Graph_EdgeCount (g) ; // number of edges in the graph

	for(uint64_t i = 0; i < n;) {
		// Decode relation header
		_DecodeRelationHeader(rdb, &r, &tensor);

		// relation id indexes the schema array; reject an out-of-range id before
		// it is used to look up the edge schema
		if(unlikely(r == GRAPH_NO_RELATION || (uint64_t) r >= rel_count)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_RELATION_ID_RANGE, "relation id", r);
			return;
		}

		if(tensor) {
			decoded_edges = _DecodeTensors(rdb, gc, r, node_id_limit,
					edge_id_limit);
		} else {
			decoded_edges = _DecodeEdges(rdb, gc, r, node_id_limit,
					edge_id_limit);
		}

		i += decoded_edges;
	}

	// read encoded deleted edge count and validate
	if(n + prev_edge_count != Graph_EdgeCount (g)) {
		SerializerIO_SetError(rdb, EMSG_DECODE_COUNT_MISMATCH, "decoded edge");
	}
}

// decode deleted edges
void RdbLoadDeletedEdges_v16
(
	SerializerIO rdb,                   // RDB
	GraphContext *gc,                   // graph context
	const uint64_t deleted_edge_count,  // number of deleted edges
	const uint64_t id_limit             // edge ids are below this
) {
	// Format:
	// edge id X N

	Graph *g = GraphContext_GetGraph (gc) ;

	uint64_t prev_deleted_edge_count = Graph_DeletedEdgeCount (g) ;

	for(uint64_t i = 0; i < deleted_edge_count; i++) {
		EdgeID id = SerializerIO_TryReadUnsigned(rdb);
		if(unlikely(id >= id_limit)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "deleted edge id",
					(unsigned long long) id);
			return;
		}
		Serializer_Graph_MarkEdgeDeleted (g, id) ;
	}

	// read encoded deleted edge count and validate
	if(deleted_edge_count + prev_deleted_edge_count !=
	   Graph_DeletedEdgeCount (g)) {
		SerializerIO_SetError(rdb, EMSG_DECODE_COUNT_MISMATCH, "deleted edge");
	}
}
