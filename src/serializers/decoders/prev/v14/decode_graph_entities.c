/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v14.h"

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
	GraphContext *gc,
	GraphEntity *e
) {
	// Format:
	// #properties N
	// (name, value type, value) X N

	uint64_t n = SerializerIO_ReadUnsigned(rdb);
	if(n == 0) return;

	SIValue vals[n];
	AttributeID ids[n];

	for(int i = 0; i < n; i++) {
		ids[i]  = SerializerIO_ReadUnsigned(rdb);
		vals[i] = _RdbLoadSIValue(rdb);
	}

	AttributeSet_Add (e->attributes, ids, vals, n, false) ;
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

	// get delay indexing configuration
	bool delay_indexing;
	Config_Option_get(Config_DELAY_INDEXING, &delay_indexing);

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
		// introduce n to each relevant index
		if(!delay_indexing) {
			for (int i = 0; i < nodeLabelCount; i++) {
				Schema *s = GraphContext_GetSchemaByID(gc, labels[i],
						SCHEMA_NODE);
				ASSERT(s != NULL);

				// index node
				if(PENDING_IDX(s)) Index_IndexNode(PENDING_IDX(s), &n);
			}
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

	Schema     *s               = NULL;
	Index      index            = NULL;
	bool       perform_indexing = false;
	NodeID     prev_src      = INVALID_ENTITY_ID;
	NodeID     prev_dest     = INVALID_ENTITY_ID;
	RelationID prev_relation = GRAPH_UNKNOWN_RELATION;

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

		e.id         = SerializerIO_ReadUnsigned(rdb);
		e.src_id     = SerializerIO_ReadUnsigned(rdb);
		e.dest_id    = SerializerIO_ReadUnsigned(rdb);
		e.relationID = SerializerIO_ReadUnsigned(rdb);

		// determine if relation contains tensors
		bool tensor = gc->decoding_context->multi_edge[e.relationID];

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

		Serializer_Graph_AllocEdgeAttributes(gc->g, e.id, &e);
		_RdbLoadEntity(rdb, gc, (GraphEntity *)&e);

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
				Serializer_OptimizedFormConnections(gc->g, prev_relation, srcs,
						dests, ids, idx, false);

				// reset batch state
				idx = 0;
			}

			// flush multi-edge batch when:
			if(tensor_idx > 0) {
				// flush batch
				Serializer_OptimizedFormConnections(gc->g, prev_relation,
						tensor_srcs, tensor_dests, tensor_ids, tensor_idx, true);

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
		Serializer_OptimizedFormConnections(gc->g, prev_relation, srcs, dests,
				ids, idx, false);
	}

	// flush last multi-edge batch
	if(tensor_idx > 0) {
		// flush batch
		Serializer_OptimizedFormConnections(gc->g, prev_relation,
				tensor_srcs, tensor_dests, tensor_ids, tensor_idx, true);
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
