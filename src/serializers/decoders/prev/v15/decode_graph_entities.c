/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v15.h"

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

void RdbLoadNodes_v15
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t node_count,
	uint64_t id_limit
) {
	// Node Format:
	//      ID
	//      #labels M
	//      (labels) X M
	//      #properties N
	//      (name, value type, value) X N

	Graph *g = GraphContext_GetGraph (gc) ;
	uint64_t label_count = Graph_LabelTypeCount (g) ;
	uint64_t prev_graph_node_count = Graph_NodeCount (g) ;

	// get delay indexing configuration
	bool delay_indexing ;
	Config_Option_get (Config_DELAY_INDEXING, &delay_indexing) ;

	for (uint64_t i = 0 ; i < node_count ; i++) {
		Node n ;
		NodeID id = SerializerIO_TryReadUnsigned (rdb) ;

		// ids run below the header's live + deleted node count
		if(unlikely(id >= id_limit)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "node id",
					(unsigned long long) id);
			return;
		}

		// #labels M
		uint64_t nodeLabelCount = SerializerIO_TryReadUnsigned (rdb) ;

		// guard the per-node label count before it sizes the VLA below
		if(unlikely(nodeLabelCount > label_count)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_LIMIT, "node label count",
					(unsigned long long) nodeLabelCount,
					(unsigned long long) label_count);
			return;
		}

		// * (labels) x M
		LabelID labels [nodeLabelCount] ;
		for (uint64_t j = 0 ; j < nodeLabelCount ; j ++){
			uint64_t l = SerializerIO_TryReadUnsigned (rdb) ;
			// labels index the schema array; reject an out-of-range id
			if(unlikely(l >= label_count)) {
				SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "node label",
						(unsigned long long) l);
				return;
			}
			labels [j] = l ;
		}

		Serializer_Graph_SetNode (g, id, labels, nodeLabelCount, &n) ;

		_RdbLoadEntity (rdb, (GraphEntity *)&n) ;

		// introduce n to each relevant index
		if (!delay_indexing) {
			for (uint64_t j = 0 ; j < nodeLabelCount ; j++) {
				Schema *s = GraphContext_GetSchemaByID (gc, labels [j],
						SCHEMA_NODE) ;
				ASSERT (s != NULL) ;

				// index node
				if (PENDING_IDX (s)) {
					Index_IndexNode (PENDING_IDX (s), &n) ;
				}
			}
		}
	}

	if(prev_graph_node_count + node_count != Graph_NodeCount (g)) {
		SerializerIO_SetError(rdb, EMSG_DECODE_COUNT_MISMATCH, "decoded node");
	}
}

void RdbLoadDeletedNodes_v15
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t deleted_node_count,
	uint64_t id_limit
) {
	// Format:
	// node id X N

	Graph *g = GraphContext_GetGraph (gc) ;
	uint64_t prev_deleted_node_count = Graph_DeletedNodeCount (g) ;

	for (uint64_t i = 0; i < deleted_node_count; i++) {
		NodeID id = SerializerIO_TryReadUnsigned (rdb) ;
		if(unlikely(id >= id_limit)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "deleted node id",
					(unsigned long long) id);
			return;
		}
		Serializer_Graph_MarkNodeDeleted (g, id) ;
	}

	if(deleted_node_count + prev_deleted_node_count !=
	   Graph_DeletedNodeCount (g)) {
		SerializerIO_SetError(rdb, EMSG_DECODE_COUNT_MISMATCH, "deleted node");
	}
}

void RdbLoadEdges_v15
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t edge_count,
	uint64_t node_id_limit,
	uint64_t edge_id_limit
) {
	// Format:
	// {
	//  edge ID
	//  source node ID
	//  destination node ID
	//  relation type
	// } X N
	// edge properties X N

	Graph *g = GraphContext_GetGraph (gc) ;
	GraphDecodeContext *decoding_context = GraphContext_GetDecodingCtx (gc) ;
	uint64_t rel_count = Graph_RelationTypeCount (g) ;
	uint64_t prev_edge_count = Graph_EdgeCount (g) ;


	Schema     *s               = NULL;
	Index      index            = NULL;
	bool       perform_indexing = false;
	NodeID     prev_src         = INVALID_ENTITY_ID;
	NodeID     prev_dest        = INVALID_ENTITY_ID;
	RelationID prev_relation    = GRAPH_UNKNOWN_RELATION;

	int       idx        = 0;                     // edge batch index
	int       tensor_idx = 0;                     // tensor batch index
	const int BATCH_SIZE = MIN(edge_count, 256);  // max batch size

	EdgeID ids         [BATCH_SIZE];
	NodeID srcs        [BATCH_SIZE];
	NodeID dests       [BATCH_SIZE];
	EdgeID tensor_ids  [BATCH_SIZE];
	NodeID tensor_srcs [BATCH_SIZE];
	NodeID tensor_dests[BATCH_SIZE];

	// get delay indexing configuration
	bool delay_indexing;
	Config_Option_get(Config_DELAY_INDEXING, &delay_indexing);

	// construct edges
	for(uint64_t i = 0; i < edge_count; i++) {
		//----------------------------------------------------------------------
		// populate edge
		//----------------------------------------------------------------------

		Edge e;

		e.id         = SerializerIO_TryReadUnsigned(rdb);
		e.src_id     = SerializerIO_TryReadUnsigned(rdb);
		e.dest_id    = SerializerIO_TryReadUnsigned(rdb);
		e.relationID = SerializerIO_TryReadUnsigned(rdb);

		// edge id indexes the edge datablock; src/dest form connections; the
		// relation id indexes the multi_edge array and the schema. reject bad
		// ids before any of them is used
		if(unlikely(e.id >= edge_id_limit)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "edge id",
					(unsigned long long) e.id);
			return;
		}
		if(unlikely(e.src_id >= node_id_limit || e.dest_id >= node_id_limit)) {
			SerializerIO_SetError(rdb,
					EMSG_DECODE_ID_RANGE, "edge endpoint",
					(unsigned long long) e.src_id, (unsigned long long) e.dest_id);
			return;
		}
		if(unlikely((uint64_t) e.relationID >= rel_count)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_RELATION_ID_RANGE, "edge relation id",
					e.relationID);
			return;
		}

		// determine if relation contains tensors
		bool tensor = decoding_context->multi_edge [e.relationID] ;

		bool relation_changed = e.relationID != prev_relation;
		if(relation_changed) {
			// reset prev src and dest node ids
			prev_src  = INVALID_ENTITY_ID;
			prev_dest = INVALID_ENTITY_ID;

			// update schema
			s = GraphContext_GetSchemaByID(gc, e.relationID, SCHEMA_EDGE);
			ASSERT(s != NULL);
			index = PENDING_IDX(s);
			perform_indexing = (!delay_indexing && index != NULL);
		}

		//----------------------------------------------------------------------
		// load edge attributes
		//----------------------------------------------------------------------

		Serializer_Graph_AllocEdgeAttributes (g, e.id, &e) ;
		_RdbLoadEntity (rdb, (GraphEntity *)&e) ;

		//----------------------------------------------------------------------
		// index edge
		//----------------------------------------------------------------------

		if(perform_indexing) {
			Index_IndexEdge(PENDING_IDX(s), &e);
		}

		//----------------------------------------------------------------------
		// flush batches
		//----------------------------------------------------------------------

		// flush batch when:
		// 1. batch is full
		// 2. relation id changed
		if(relation_changed ||
		   (idx > 0 && idx >= BATCH_SIZE) ||
		   (tensor_idx > 0 && tensor_idx >= BATCH_SIZE)) {

			if(idx > 0) {
				// flush batch
				Serializer_OptimizedFormConnections (g, prev_relation, srcs,
						dests, ids, idx, false) ;

				// reset batch state
				idx = 0;
			}

			// flush multi-edge batch when:
			if(tensor_idx > 0) {
				// flush batch
				Serializer_OptimizedFormConnections (g, prev_relation,
						tensor_srcs, tensor_dests, tensor_ids, tensor_idx, true) ;

				// reset multi-edge batch state
				tensor_idx = 0;
			}
		}

		// determine if we're dealing with a multi-edge
		// first iteration is considered multi_edge, as we don't know which edge
		// was introduced in the previous virtual key
		bool multi_edge = (tensor                                           &&
						  ((e.src_id == prev_src && e.dest_id == prev_dest) ||
						   relation_changed));

		// accumulate edge
		if(multi_edge) {
			tensor_ids[tensor_idx]   = e.id;
			tensor_srcs[tensor_idx]  = e.src_id;
			tensor_dests[tensor_idx] = e.dest_id;
			tensor_idx++;  // advance batch index
		} else {
			// batch edge src, dest and id
			ids[idx]   = e.id;
			srcs[idx]  = e.src_id;
			dests[idx] = e.dest_id;
			idx++;  // advance batch index
		}

		// update prev values
		prev_src      = e.src_id;
		prev_dest     = e.dest_id;
		prev_relation = e.relationID;
	}

	// flush last batch
	if(idx > 0) {
		// flush batch
		Serializer_OptimizedFormConnections (g, prev_relation, srcs, dests,
				ids, idx, false) ;
	}

	// flush last multi-edge batch
	if(tensor_idx > 0) {
		// flush batch
		Serializer_OptimizedFormConnections (g, prev_relation,
				tensor_srcs, tensor_dests, tensor_ids, tensor_idx, true) ;
	}

	if(edge_count + prev_edge_count != Graph_EdgeCount (g)) {
		SerializerIO_SetError(rdb, EMSG_DECODE_COUNT_MISMATCH, "decoded edge");
	}
}

void RdbLoadDeletedEdges_v15
(
	SerializerIO rdb,
	GraphContext *gc,
	uint64_t deleted_edge_count,
	uint64_t id_limit
) {
	// Format:
	// edge id X N

	Graph *g = GraphContext_GetGraph (gc) ;
	uint64_t prev_deleted_edge_count = Graph_DeletedEdgeCount (g) ;

	for (uint64_t i = 0; i < deleted_edge_count; i++) {
		EdgeID id = SerializerIO_TryReadUnsigned (rdb) ;
		if(unlikely(id >= id_limit)) {
			SerializerIO_SetError(rdb, EMSG_DECODE_ID_RANGE, "deleted edge id",
					(unsigned long long) id);
			return;
		}
		Serializer_Graph_MarkEdgeDeleted (g, id) ;
	}

	if(deleted_edge_count + prev_deleted_edge_count !=
	   Graph_DeletedEdgeCount (g)) {
		SerializerIO_SetError(rdb, EMSG_DECODE_COUNT_MISMATCH, "deleted edge");
	}
}
