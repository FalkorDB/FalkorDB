/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v16.h"

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

	AttributeSet_AddNoClone(e->attributes, ids, vals, n, false);
}

void RdbLoadNodes_v16
(
	SerializerIO rdb,
	GraphContext *gc
) {
	// Node Format:
	//      ID
	//      #labels M
	//      (labels) X M
	//      #properties N
	//      (name, value type, value) X N

	uint64_t decoded_node_count = 0;
	uint64_t prev_graph_node_count = Graph_NodeCount(gc->g);

	while(true) {
		Node n;
		NodeID id = SerializerIO_ReadUnsigned(rdb);

		// break on end marker
		if(id == INVALID_ENTITY_ID) break;

		// #labels M
		uint64_t nodeLabelCount = SerializerIO_ReadUnsigned(rdb);

		// (labels) x M
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

			// index node
			if(PENDING_IDX(s)) Index_IndexNode(PENDING_IDX(s), &n);
		}

		decoded_node_count++;
	}

	// read encoded node count and validate
	uint64_t encoded_node_count = SerializerIO_ReadUnsigned(rdb);
	ASSERT(encoded_node_count == decoded_node_count);
	ASSERT(prev_graph_node_count + decoded_node_count == Graph_NodeCount(gc->g));
}

void RdbLoadDeletedNodes_v16
(
	SerializerIO rdb,
	GraphContext *gc
) {
	// Format:
	// node id X N

	uint64_t decoded_deleted_node_count = 0;
	uint64_t prev_deleted_node_count = Graph_DeletedNodeCount(gc->g);

	while(true) {
		NodeID id = SerializerIO_ReadUnsigned(rdb);

		// break on end marker
		if(id == INVALID_ENTITY_ID) break;

		Serializer_Graph_MarkNodeDeleted(gc->g, id);

		decoded_deleted_node_count++;
	}

	// read encoded deleted node count and validate
	uint64_t encoded_deleted_node_count = SerializerIO_ReadUnsigned(rdb);
	ASSERT(encoded_deleted_node_count == decoded_deleted_node_count);
	ASSERT(decoded_deleted_node_count + prev_deleted_node_count ==
			Graph_DeletedNodeCount(gc->g));
}

void RdbLoadEdges_v16
(
	SerializerIO rdb,
	GraphContext *gc
) {
	// Format:
	// {
	//  edge ID
	//  source node ID
	//  destination node ID
	//  relation type
	// } X N
	// edge properties X N

	Edge          e;
	Schema*       s;
	AttributeSet* set;
	bool          multi_edge_relation;

	NodeID     prev_src      = INVALID_ENTITY_ID;
	NodeID     prev_dest     = INVALID_ENTITY_ID;
	NodeID     max_node_id   = 0;
	RelationID prev_relation = GRAPH_UNKNOWN_RELATION;

	uint64_t  prev_edge_count    = Graph_EdgeCount(gc->g); // number of edges in the graph
	uint64_t  decoded_edge_count = 0;    // number of edges decoded
	int       idx                = 0;    // batch next element position
	int       multi_edge_idx     = 0;    // multi-edge batch next element position
	const int MAX_BATCH_SIZE     = 256;  // max batch size
	//const int MAX_BATCH_SIZE = 16384;  // max batch size

	// single edge batch
	EdgeID ids  [MAX_BATCH_SIZE];
	NodeID srcs [MAX_BATCH_SIZE];
	NodeID dests[MAX_BATCH_SIZE];

	// multi-edge batch
	EdgeID multi_edge_ids  [MAX_BATCH_SIZE];
	NodeID multi_edge_srcs [MAX_BATCH_SIZE];
	NodeID multi_edge_dests[MAX_BATCH_SIZE];

	// construct edges
	while(true) {
		//----------------------------------------------------------------------
		// populate edge
		//----------------------------------------------------------------------

		e.id = SerializerIO_ReadUnsigned(rdb);

		// break on end marker
		if(e.id == INVALID_ENTITY_ID) break;

		decoded_edge_count++;

		e.src_id     = SerializerIO_ReadUnsigned(rdb);
		e.dest_id    = SerializerIO_ReadUnsigned(rdb);
		e.relationID = SerializerIO_ReadUnsigned(rdb);

		//----------------------------------------------------------------------
		// load edge attributes
		//----------------------------------------------------------------------

		Serializer_Graph_AllocEdgeAttributes(gc->g, e.id, &e);
		_RdbLoadEntity(rdb, gc, (GraphEntity *)&e);

		// handle transition to a new relationship type
		bool relation_changed = e.relationID != prev_relation;
		if(relation_changed) {
			s = GraphContext_GetSchemaByID(gc, e.relationID, SCHEMA_EDGE);
			ASSERT(s != NULL);

			// determine if relation contains "multi-edge"
			multi_edge_relation = gc->decoding_context->multi_edge[e.relationID];

			// reset prev
			prev_src  = INVALID_ENTITY_ID;
			prev_dest = INVALID_ENTITY_ID;
		}

		//----------------------------------------------------------------------
		// index edge
		//----------------------------------------------------------------------

		if(PENDING_IDX(s)) Index_IndexEdge(PENDING_IDX(s), &e);

		//----------------------------------------------------------------------
		// flush batches
		//----------------------------------------------------------------------

		// flush batch when:
		// 1. batch is full
		// 2. relation id changed
		if(idx > 0 && (idx == MAX_BATCH_SIZE || relation_changed)) {
			// flush batch
			Serializer_OptimizedFormConnections(gc->g, prev_relation, srcs,
					dests, ids, idx, max_node_id, false);

			// reset batch count
			idx = 0;
		}

		// flush multi-edge batch when:
		// 1. batch is full
		// 2. relation id changed
		if(multi_edge_idx > 0 &&
		   (multi_edge_idx == MAX_BATCH_SIZE || relation_changed)) {
			// flush batch
			Serializer_OptimizedFormConnections(gc->g, prev_relation,
					multi_edge_srcs, multi_edge_dests, multi_edge_ids,
					multi_edge_idx, max_node_id, true);

			// reset multi-edge batch count
			multi_edge_idx = 0;
		}

		// determine if we're dealing with a multi-edge
		bool multi_edge = (multi_edge_relation  &&
						   e.src_id == prev_src &&
						   e.dest_id == prev_dest);

		// accumulate edge
		if(multi_edge) {
			// batch multi-edge src, dest and id
			multi_edge_ids[multi_edge_idx]   = e.id;
			multi_edge_srcs[multi_edge_idx]  = e.src_id;
			multi_edge_dests[multi_edge_idx] = e.dest_id;
			multi_edge_idx++;  // advance batch index
		} else {
			// batch edge src, dest and id
			ids[idx]   = e.id;
			srcs[idx]  = e.src_id;
			dests[idx] = e.dest_id;
			idx++;  // advance batch index
		}

		// maintain max src and max dest
		max_node_id = MAX(max_node_id, MAX(e.src_id, e.dest_id));

		// update prev values
		prev_src      = e.src_id;
		prev_dest     = e.dest_id;
		prev_relation = e.relationID;
	}

	// flush last batch
	if(idx > 0) {
		// flush batch
		Serializer_OptimizedFormConnections(gc->g, prev_relation, srcs,
				dests, ids, idx, max_node_id, false);
	}

	// flush last multi-edge batch
	if(multi_edge_idx > 0) {
		// flush batch
		Serializer_OptimizedFormConnections(gc->g, prev_relation,
				multi_edge_srcs, multi_edge_dests, multi_edge_ids,
				multi_edge_idx, max_node_id, true);
	}

	// read encoded deleted edge count and validate
	uint64_t encoded_edge_count = SerializerIO_ReadUnsigned(rdb);
	ASSERT(encoded_edge_count == decoded_edge_count);
	ASSERT(decoded_edge_count + prev_edge_count == Graph_EdgeCount(gc->g));
}

void RdbLoadDeletedEdges_v16
(
	SerializerIO rdb,
	GraphContext *gc
) {
	// Format:
	// edge id X N

	uint64_t decoded_deleted_edge_count = 0;
	uint64_t prev_deleted_edge_count = Graph_DeletedEdgeCount(gc->g);

	while(true) {
		EdgeID id = SerializerIO_ReadUnsigned(rdb);

		// break on end marker
		if(id == INVALID_ENTITY_ID) break;

		Serializer_Graph_MarkEdgeDeleted(gc->g, id);

		decoded_deleted_edge_count++;
	}

	// read encoded deleted edge count and validate
	uint64_t encoded_deleted_edge_count = SerializerIO_ReadUnsigned(rdb);
	ASSERT(encoded_deleted_edge_count == decoded_deleted_edge_count);
	ASSERT(decoded_deleted_edge_count + prev_deleted_edge_count ==
			Graph_DeletedNodeCount(gc->g));
}

