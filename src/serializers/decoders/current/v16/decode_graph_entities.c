/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
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
	GraphContext *gc,
	const uint64_t node_count
) {
	// Node Format:
	//      ID
	//      #labels M
	//      (labels) X M
	//      #properties N
	//      (name, value type, value) X N

	uint64_t prev_graph_node_count = Graph_NodeCount(gc->g);

	for(uint64_t i = 0; i < node_count; i++) {
		Node n;
		NodeID id = SerializerIO_ReadUnsigned(rdb);

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
	}

	ASSERT(prev_graph_node_count + node_count == Graph_NodeCount(gc->g));
}

void RdbLoadDeletedNodes_v16
(
	SerializerIO rdb,
	GraphContext *gc,
	const uint64_t deleted_node_count
) {
	// Format:
	// node id X N

	uint64_t prev_deleted_node_count = Graph_DeletedNodeCount(gc->g);

	for(uint64_t i = 0; i < deleted_node_count; i++) {
		NodeID id = SerializerIO_ReadUnsigned(rdb);
		Serializer_Graph_MarkNodeDeleted(gc->g, id);
	}

	// read encoded deleted node count and validate
	ASSERT(deleted_node_count + prev_deleted_node_count ==
			Graph_DeletedNodeCount(gc->g));
}

static void _DecodeRelationHeader
(
	SerializerIO rdb,
	RelationID *r,
	bool *tensor
) {
	// format:
	//
	// Header:
	//     relationship ID
	//     rather or not relationship contains tensors
	//

	*r = SerializerIO_ReadUnsigned(rdb);

	// end marker
	if(*r == GRAPH_NO_RELATION) {
		return;
	}

	*tensor = SerializerIO_ReadUnsigned(rdb);
}

uint64_t _DecodeTensors
(
	SerializerIO rdb,
	GraphContext *gc,
	RelationID r
) {
	// format:
	// edge format:
	//     edge id
	//     source node id
	//     destination node id
	//     multi-edge
	//     edge properties

	Edge e;

	int      idx            = 0;  // batch next element position
	NodeID   max_id         = 0;  // max node id
	uint64_t decoded_edges  = 0;  // number of decoded edges
	int      multi_edge_idx = 0;  // tensors batch next element position

	Schema *s = GraphContext_GetSchemaByID(gc, r, SCHEMA_EDGE);
	ASSERT(s != NULL);

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

	while(true) {
		// decode edge id
		e.id = SerializerIO_ReadUnsigned(rdb);

		// break on end marker
		if(e.id == INVALID_ENTITY_ID) {
			break;
		}

		// decode edge src, dest and tensor marker
		e.relationID = r;
		e.src_id     = SerializerIO_ReadUnsigned(rdb);
		e.dest_id    = SerializerIO_ReadUnsigned(rdb);

		bool tensor = SerializerIO_ReadUnsigned(rdb);

		//----------------------------------------------------------------------
		// load edge attributes
		//----------------------------------------------------------------------

		Serializer_Graph_AllocEdgeAttributes(gc->g, e.id, &e);
		_RdbLoadEntity(rdb, gc, (GraphEntity *)&e);

		//----------------------------------------------------------------------
		// index edge
		//----------------------------------------------------------------------

		if(PENDING_IDX(s)) Index_IndexEdge(PENDING_IDX(s), &e);

		// update max node id
		max_id = MAX(max_id, MAX(e.src_id, e.dest_id));

		// accumulate edge
		if(tensor) {
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

		//----------------------------------------------------------------------
		// flush batches
		//----------------------------------------------------------------------

		// flush multi-edge batch when:
		if(multi_edge_idx == MAX_BATCH_SIZE) {
			// flush batch
			Serializer_OptimizedFormConnections(gc->g, r, multi_edge_srcs,
					multi_edge_dests, multi_edge_ids, multi_edge_idx,
					max_id, true);

			// reset multi-edge batch count
			multi_edge_idx = 0;
		}

		// flush batch when:
		// 1. batch is full
		// 2. relation id changed
		if(idx == MAX_BATCH_SIZE) {
			// flush batch
			Serializer_OptimizedFormConnections(gc->g, r, srcs, dests, ids,
					idx, max_id, false);

			// reset batch count
			idx = 0;
		}

		decoded_edges++;
	}

	//----------------------------------------------------------------------
	// flush batches
	//----------------------------------------------------------------------

	// flush multi-edge batch when:
	if(multi_edge_idx == MAX_BATCH_SIZE) {
		// flush batch
		Serializer_OptimizedFormConnections(gc->g, r, multi_edge_srcs,
				multi_edge_dests, multi_edge_ids, multi_edge_idx,
				max_id, true);

		// reset multi-edge batch count
		multi_edge_idx = 0;
	}

	// flush batch when:
	// 1. batch is full
	// 2. relation id changed
	if(idx == MAX_BATCH_SIZE) {
		// flush batch
		Serializer_OptimizedFormConnections(gc->g, r, srcs, dests, ids,
				idx, max_id, false);

		// reset batch count
		idx = 0;
	}

	return decoded_edges;
}

uint64_t _DecodeEdges
(
	SerializerIO rdb,
	GraphContext *gc,
	RelationID r
) {
	// Format:
	//  edge ID
	//  source node ID
	//  destination node ID
	//  edge properties

	Edge e;
	NodeID max_id = 0;
	uint64_t decoded_edges = 0;

	Schema *s = GraphContext_GetSchemaByID(gc, r, SCHEMA_EDGE);
	ASSERT(s != NULL);

	while(true) {
		e.id = SerializerIO_ReadUnsigned(rdb);

		// break on end marker
		if(e.id == INVALID_ENTITY_ID) {
			break;
		}

		e.src_id     = SerializerIO_ReadUnsigned(rdb);
		e.dest_id    = SerializerIO_ReadUnsigned(rdb);
		e.relationID = r;

		//----------------------------------------------------------------------
		// load edge attributes
		//----------------------------------------------------------------------

		Serializer_Graph_AllocEdgeAttributes(gc->g, e.id, &e);
		_RdbLoadEntity(rdb, gc, (GraphEntity *)&e);

		//----------------------------------------------------------------------
		// index edge
		//----------------------------------------------------------------------

		if(PENDING_IDX(s)) Index_IndexEdge(PENDING_IDX(s), &e);

		max_id = MAX(max_id, MAX(e.src_id, e.dest_id));
		Serializer_OptimizedFormConnections(gc->g, r, &e.src_id, &e.dest_id,
				&e.id, 1, max_id, false);

		decoded_edges++;
	}

	return decoded_edges;
}

void RdbLoadEdges_v16
(
	SerializerIO rdb,
	GraphContext *gc,
	const uint64_t n
) {
	// Format:
	// {
	//  edge ID
	//  source node ID
	//  destination node ID
	//  relation type
	//  multi-edge
	// } X N
	// edge properties X N

	bool tensor;
	RelationID r;
	uint64_t decoded_edges   = 0;
	uint64_t prev_edge_count = Graph_EdgeCount(gc->g); // number of edges in the graph

	for(uint64_t i = 0; i < n;) {
		// Decode relation header
		_DecodeRelationHeader(rdb, &r, &tensor);

		if(tensor) {
			decoded_edges = _DecodeTensors(rdb, gc, r);
		} else {
			decoded_edges = _DecodeEdges(rdb, gc, r);
		}

		i += decoded_edges;
	}

	// read encoded deleted edge count and validate
	ASSERT(n + prev_edge_count == Graph_EdgeCount(gc->g));
}

void RdbLoadDeletedEdges_v16
(
	SerializerIO rdb,
	GraphContext *gc,
	const uint64_t deleted_edge_count
) {
	// Format:
	// edge id X N

	uint64_t prev_deleted_edge_count = Graph_DeletedEdgeCount(gc->g);

	for(uint64_t i = 0; i < deleted_edge_count; i++) {
		EdgeID id = SerializerIO_ReadUnsigned(rdb);
		Serializer_Graph_MarkEdgeDeleted(gc->g, id);
	}

	// read encoded deleted edge count and validate
	ASSERT(deleted_edge_count + prev_deleted_edge_count ==
			Graph_DeletedNodeCount(gc->g));
}

