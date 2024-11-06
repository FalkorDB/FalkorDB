/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v15.h"
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

void RdbLoadNodes_v15
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

void RdbLoadDeletedNodes_v15
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

void RdbLoadEdges_v15
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


	Edge          e;
	AttributeSet* set;

	NodeID     prev_src      = INVALID_ENTITY_ID;
	NodeID     prev_dest     = INVALID_ENTITY_ID;
	NodeID     max_node_id   = 0;
	RelationID prev_relation = GRAPH_UNKNOWN_RELATION;

	int       idx            = 0;      // batch next element position
	int       multi_edge_idx = 0;      //
	const int MAX_BATCH_SIZE = MIN(edge_count, 16384);  // max batch size

	EdgeID ids  [MAX_BATCH_SIZE];
	NodeID srcs [MAX_BATCH_SIZE];
	NodeID dests[MAX_BATCH_SIZE];

	EdgeID multi_edge_ids  [MAX_BATCH_SIZE];
	NodeID multi_edge_srcs [MAX_BATCH_SIZE];
	NodeID multi_edge_dests[MAX_BATCH_SIZE];

	// construct edges
	for(uint64_t i = 0; i < edge_count; i++) {
		//----------------------------------------------------------------------
		// populate edge
		//----------------------------------------------------------------------

		e.id         = SerializerIO_ReadUnsigned(rdb);
		e.src_id     = SerializerIO_ReadUnsigned(rdb);
		e.dest_id    = SerializerIO_ReadUnsigned(rdb);
		e.relationID = SerializerIO_ReadUnsigned(rdb);

		// determine if relation contains "multi-edge"
		bool multi_edge_relation = gc->decoding_context->multi_edge[e.relationID];

		//----------------------------------------------------------------------
		// load edge attributes
		//----------------------------------------------------------------------

		AttributeSet *set = DataBlock_AllocateItemOutOfOrder(gc->g->edges, e.id);
		*set = NULL;
		e.attributes = set;
		_RdbLoadEntity(rdb, gc, (GraphEntity *)&e);

		//----------------------------------------------------------------------
		// index edge
		//----------------------------------------------------------------------

		Schema *s = GraphContext_GetSchemaByID(gc, e.relationID, SCHEMA_EDGE);
		ASSERT(s != NULL);
		if(PENDING_IDX(s)) Index_IndexEdge(PENDING_IDX(s), &e);

		//----------------------------------------------------------------------
		// flush batches
		//----------------------------------------------------------------------

		// flush multi-edge batch when:
		// 1. batch is full
		// 2. relation id changed
		if(multi_edge_idx > 0 &&
		   (multi_edge_idx >= MAX_BATCH_SIZE || e.relationID != prev_relation)) {
			printf("Flush multi-edge batch\n");
			printf("batch size: %d\n", multi_edge_idx);
			// flush batch
			Serializer_OptimizedFormConnections(gc->g, prev_relation,
					multi_edge_srcs, multi_edge_dests, multi_edge_ids,
					multi_edge_idx, max_node_id, true);

			// reset multi-edge batch state
			multi_edge_idx = 0;
		}

		// flush batch when:
		// 1. batch is full
		// 2. relation id changed
		if(idx > 0 && (idx >= MAX_BATCH_SIZE || e.relationID != prev_relation)) {
			// flush batch
			Serializer_OptimizedFormConnections(gc->g, prev_relation, srcs,
					dests, ids, idx, max_node_id, false);

			// reset batch state
			idx = 0;
		}

		// determine if we're dealing with a multi-edge
		bool multi_edge = (multi_edge_relation                              &&
						  ((e.src_id == prev_src && e.dest_id == prev_dest) ||
						  (prev_src == INVALID_ENTITY_ID && prev_dest == INVALID_ENTITY_ID)));

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

	// flush last multi-edge batch
	if(multi_edge_idx > 0) {
		printf("Flush multi-edge batch\n");
		printf("batch size: %d\n", multi_edge_idx);
		// flush batch
		Serializer_OptimizedFormConnections(gc->g, prev_relation,
				multi_edge_srcs, multi_edge_dests, multi_edge_ids,
				multi_edge_idx, max_node_id, true);
	}

	// flush last batch
	if(idx > 0) {
		// flush batch
		Serializer_OptimizedFormConnections(gc->g, prev_relation, srcs,
				dests, ids, idx, max_node_id, false);
	}
}

void RdbLoadDeletedEdges_v15
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
