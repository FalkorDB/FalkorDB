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

	for(uint64_t i = 0; i < n; i++) {
		ids[i]  = SerializerIO_ReadUnsigned(rdb);
		vals[i] = _RdbLoadSIValue(rdb);
	}

	AttributeSet_Add (e->attributes, ids, vals, n, false) ;
}

// decode nodes
void RdbLoadNodes_v16
(
	SerializerIO rdb,          // RDB
	GraphContext *gc,          // graph context
	const uint64_t node_count  // number of nodes to decode
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

	uint64_t prev_graph_node_count = Graph_NodeCount(gc->g);

	for(uint64_t i = 0; i < node_count; i++) {
		Node n;
		NodeID id = SerializerIO_ReadUnsigned(rdb);

		// #labels M
		uint64_t nodeLabelCount = SerializerIO_ReadUnsigned(rdb);

		// (labels) x M
		LabelID labels[nodeLabelCount];
		for(uint64_t j = 0; j < nodeLabelCount; j ++){
			labels[j] = SerializerIO_ReadUnsigned(rdb);
		}

		Serializer_Graph_SetNode(gc->g, id, labels, nodeLabelCount, &n);

		_RdbLoadEntity(rdb, gc, (GraphEntity *)&n);

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

	ASSERT(prev_graph_node_count + node_count == Graph_NodeCount(gc->g));
}

// decode deleted nodes
void RdbLoadDeletedNodes_v16
(
	SerializerIO rdb,                  // RDB
	GraphContext *gc,                  // graph context
	const uint64_t deleted_node_count  // number of deleted nodes
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

	*r = SerializerIO_ReadUnsigned(rdb);
	ASSERT(*r != GRAPH_NO_RELATION);

	*tensor = SerializerIO_ReadUnsigned(rdb);
}

// decode tensors
static uint64_t _DecodeTensors
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	RelationID r       // relationship type
) {
	// format:
	// edge format:
	//     edge id
	//     source node id
	//     destination node id
	//     multi-edge
	//     edge properties

	Edge e;                           // current decoded edge
	int       idx            = 0;    // batch index
	uint64_t  decoded_edges  = 0;    // number of decoded edges
	int       tensor_idx     = 0;    // tensors batch index
	const int BATCH_SIZE     = 4096;  // batch size

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
		e.id = SerializerIO_ReadUnsigned(rdb);

		// break on END MARKER
		if(e.id == INVALID_ENTITY_ID) {
			break;
		}

		// decode edge source node id
		e.src_id     = SerializerIO_ReadUnsigned(rdb);
		// decode edge destination node id
		e.dest_id    = SerializerIO_ReadUnsigned(rdb);
		// set edge relation id
		e.relationID = r;

		// decode tensor marker
		bool tensor = SerializerIO_ReadUnsigned(rdb);

		// load edge attributes
		Serializer_Graph_AllocEdgeAttributes(gc->g, e.id, &e);
		_RdbLoadEntity(rdb, gc, (GraphEntity *)&e);

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
			Serializer_OptimizedFormConnections(gc->g, r, tensors_srcs,
					tensors_dests, tensors_ids, tensor_idx, true);

			// reset multi-edge batch count
			tensor_idx = 0;
		}

		// flush batch
		if(idx == BATCH_SIZE) {
			// flush batch
			Serializer_OptimizedFormConnections(gc->g, r, srcs, dests, ids, idx,
					false);

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
		Serializer_OptimizedFormConnections(gc->g, r, tensors_srcs,
				tensors_dests, tensors_ids, tensor_idx, true);
	}

	// flush batch
	if(idx > 0) {
		// flush batch
		Serializer_OptimizedFormConnections(gc->g, r, srcs, dests, ids, idx,
				false);
	}

	return decoded_edges;
}

// decode edges
static uint64_t _DecodeEdges
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	RelationID r       // relationship type
) {
	// Format:
	//  edge ID
	//  source node ID
	//  destination node ID
	//  edge properties

	Edge e;                        // current decoded edge
	uint64_t idx           = 0;    // batch index
	uint64_t decoded_edges = 0;    // number of decoded edges
	const int BATCH_SIZE   = 256;  // batch size

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
		e.id = SerializerIO_ReadUnsigned(rdb);

		// break on END MARKER
		if(e.id == INVALID_ENTITY_ID) {
			break;
		}

		// decode edge source node id
		e.src_id     = SerializerIO_ReadUnsigned(rdb);
		// decode edge destination node id
		e.dest_id    = SerializerIO_ReadUnsigned(rdb);
		// set edge relation id
		e.relationID = r;

		// load edge attributes
		Serializer_Graph_AllocEdgeAttributes(gc->g, e.id, &e);
		_RdbLoadEntity(rdb, gc, (GraphEntity *)&e);

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
			Serializer_OptimizedFormConnections(gc->g, r, srcs, dests, ids, idx,
					false);
			// reset batch index
			idx = 0;
		}

		// update number of decode edges
		decoded_edges++;
	}

	// flush batch
	if(idx > 0) {
		Serializer_OptimizedFormConnections(gc->g, r, srcs, dests, ids, idx,
				false);
	}

	return decoded_edges;
}

// decode edges
void RdbLoadEdges_v16
(
	SerializerIO rdb,  // RDB
	GraphContext *gc,  // graph context
	const uint64_t n   // virtual key capacity
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

// decode deleted edges
void RdbLoadDeletedEdges_v16
(
	SerializerIO rdb,                  // RDB
	GraphContext *gc,                  // graph context
	const uint64_t deleted_edge_count  // number of deleted edges
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
			Graph_DeletedEdgeCount(gc->g));
}

