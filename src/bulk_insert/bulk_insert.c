/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "bulk_insert.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "../schema/schema.h"
#include "../datatypes/array.h"
#include "../graph/graph_hub.h"
#include "../errors/error_msgs.h"
#include "../util/identifier_limits.h"

#include <string.h>
#include <stdlib.h>

// the first byte of each property in the binary stream
// is used to indicate the type of the subsequent SIValue
typedef enum {
	BI_NULL   = 0,
	BI_BOOL   = 1,
	BI_DOUBLE = 2,
	BI_STRING = 3,
	BI_LONG   = 4,
	BI_ARRAY  = 5,
} TYPE;

// binary header format:
// - entity name       : null-terminated C string
// - property count    : 4-byte unsigned integer
// [0..property_count] : null-terminated C string

// read the label strings from a header, update schemas
// and retrieve the label IDs
static int *_BulkInsert_ReadHeaderLabels
(
	RedisModuleCtx *ctx,
	GraphContext* gc,
	SchemaType t,
	const char* data,
	size_t* data_idx
) {
	ASSERT (ctx      != NULL) ;
	ASSERT (gc       != NULL) ;
	ASSERT (data     != NULL) ;
	ASSERT (data_idx != NULL) ;

	// first sequence is entity label(s)
	const char *labels = data + *data_idx ;
	size_t labels_len = strlen (labels) ;
	*data_idx += labels_len + 1 ;

	// array of all label IDs
	int *label_ids = arr_new(int, 1) ;
	// stack variable to contain a single label
	char label[labels_len + 1] ;

	while (true) {
		// look for a colon delimiting another label
		char* found = strchr (labels, ':') ;
		if (found) {
			ASSERT (t == SCHEMA_NODE) ; // only nodes can have multiple labels
			// this entity file describes multiple labels, copy the current one
			size_t len = found - labels ;
			memcpy (label, labels, len) ;
			label[len] = '\0' ;
			// update the labels pointer for the next seek
			labels += len + 1 ;
		} else {
			// reached the last (or only) label; copy it
			size_t len = strlen (labels) ;
			// also copy the terminating NULL character
			memcpy (label, labels, len + 1) ;
		}

		if (strnlen (label, MAX_IDENTIFIER_LEN + 1) > MAX_IDENTIFIER_LEN) {
			RedisModule_ReplyWithErrorFormat (ctx, EMSG_IDENTIFIER_TOO_LONG,
					"Label name", MAX_IDENTIFIER_LEN) ;
			arr_free (label_ids) ;
			return NULL ;
		}

		// create schema in case it doesn't exists
		Schema *s = GraphContext_FindOrAddSchema (gc, label, t, NULL) ;
		ASSERT (s != NULL) ;

		// store the label ID
		arr_append (label_ids, Schema_GetID (s)) ;

		// break if we've exhausted all labels
		if (!found) {
			break ;
		}
	}

	return label_ids ;
}

// read the property keys from a header
static AttributeID *_BulkInsert_ReadHeaderProperties
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	SchemaType t,
	const char *data,
	size_t *data_idx,
	uint16_t *prop_count
) {
	ASSERT (gc         != NULL) ;
	ASSERT (ctx        != NULL) ;
	ASSERT (data       != NULL) ;
	ASSERT (data_idx   != NULL) ;
	ASSERT (prop_count != NULL) ;

	// next 4 bytes are property count
	uint _prop_count = *(uint*)&data[*data_idx] ;
	assert (_prop_count < 65535) ;  // restrict number of attributes

	*prop_count = _prop_count ;
	*data_idx += sizeof (unsigned int) ;

	if (*prop_count == 0) {
		return NULL ;
	}

	AttributeID *prop_indices = rm_malloc (*prop_count * sizeof (AttributeID)) ;

	// the rest of the line is [char *prop_key] * prop_count
	for (uint j = 0; j < *prop_count; j++) {
		char* prop_key = (char*)data + *data_idx ;
		*data_idx += strlen(prop_key) + 1 ;

		if (strnlen (prop_key, MAX_IDENTIFIER_LEN + 1) > MAX_IDENTIFIER_LEN) {
			RedisModule_ReplyWithErrorFormat (ctx, EMSG_IDENTIFIER_TOO_LONG,
					"Property name", MAX_IDENTIFIER_LEN) ;
			rm_free (prop_indices) ;
			return NULL ;
		}

		// add properties to schemas
		prop_indices[j] = GraphContext_FindOrAddAttribute (gc, prop_key, NULL) ;
	}

	return prop_indices ;
}

// read an SIValue from the data stream and update the index appropriately
static SIValue _BulkInsert_ReadProperty
(
	const char *data,
	size_t *data_idx
) {
	// binary property format:
    // - 1 byte: TYPE enum
    // - NULL      : no payload
    // - BOOL      : 1 byte (0/1)
    // - DOUBLE    : 8 bytes
    // - LONG      : 8 bytes
    // - STRING    : null-terminated C string
    // - ARRAY     : 8-byte length + N serialized values

	TYPE t = data[*data_idx] ;
	(*data_idx)++ ;

	switch (t) {
		case BI_NULL:
			return SI_NullVal () ;

		case BI_BOOL: {
			bool b = data[*data_idx];
			(*data_idx)++ ;
			return SI_BoolVal (b) ;
		}

		case BI_DOUBLE: {
			double d = *(double*)&data[*data_idx] ;
			*data_idx += sizeof (double) ;
			return SI_DoubleVal (d) ;
		}

		case BI_LONG: {
			int64_t i = *(int64_t*)&data[*data_idx] ;
			*data_idx += sizeof (int64_t) ;
			return SI_LongVal (i) ;
		}

		case BI_STRING: {
			const char *s = data + *data_idx ;
			*data_idx += strlen (s) + 1 ;
			return SI_DuplicateStringVal ((char*)s) ;
		}

		case BI_ARRAY: {
			int64_t len = *(int64_t*)&data[*data_idx] ;
			*data_idx += sizeof (int64_t) ;
			SIValue arr = SIArray_New (len) ;
			for (uint i = 0; i < len; i++) {
				// convert every element and add to array.
				SIArray_Append (&arr, _BulkInsert_ReadProperty (data, data_idx)) ;
			}
			return arr ;
		}

		default:
			ASSERT (false && "unknown value type") ;
			return SI_NullVal () ;
	}
}

// validate the header identifiers of a single CSV file
// (label/rel-type names and property names) without modifying graph state
// returns BULK_OK if all identifiers are within length limits, BULK_FAIL otherwise
static int _BulkInsert_ValidateHeader
(
	RedisModuleCtx *ctx,
	SchemaType t,
	const char *data,
	size_t data_len
) {
	ASSERT (ctx  != NULL) ;
	ASSERT (data != NULL) ;

	size_t data_idx = 0 ;

	// read the entire label / rel-type segment
	const char *labels = data ;
	size_t labels_len = strlen (labels) ;
	data_idx += labels_len + 1 ;

	// validate each colon-delimited name
	const char *ptr = labels ;
	while (true) {
		char *found = strchr (ptr, ':') ;
		size_t len = found ? (size_t)(found - ptr) : strlen (ptr) ;

		if (len > MAX_IDENTIFIER_LEN) {
			RedisModule_ReplyWithErrorFormat (ctx, EMSG_IDENTIFIER_TOO_LONG,
					"Label name", MAX_IDENTIFIER_LEN) ;
			return BULK_FAIL ;
		}

		if (!found) break ;
		ptr = found + 1 ;
	}

	// read property count
	if (data_idx + sizeof (uint) > data_len) {
		return BULK_OK ;
	}

	uint prop_count = *(uint*)&data[data_idx] ;
	data_idx += sizeof (uint) ;

	// validate each property name
	for (uint j = 0; j < prop_count; j++) {
		if (data_idx >= data_len) break ;
		const char *prop_key = data + data_idx ;
		size_t n = strlen (prop_key) + 1 ;
		data_idx += n ;

		if (n > MAX_IDENTIFIER_LEN) {
			RedisModule_ReplyWithErrorFormat (ctx, EMSG_IDENTIFIER_TOO_LONG,
					"Property name", MAX_IDENTIFIER_LEN) ;
			return BULK_FAIL ;
		}
	}

	return BULK_OK ;
}

// validate headers of all CSV tokens of a given type without touching the graph
static int _BulkInsert_ValidateTokens
(
	RedisModuleCtx *ctx,
	int token_count,
	RedisModuleString **argv,
	SchemaType type
) {
	for (int i = 0; i < token_count; i++) {
		size_t len ;
		const char *data = RedisModule_StringPtrLen (argv[i], &len) ;
		if (_BulkInsert_ValidateHeader (ctx, type, data, len) != BULK_OK) {
			return BULK_FAIL ;
		}
	}

	return BULK_OK ;
}

// process a single node CSV file
static int _BulkInsert_ProcessNodeFile
(
	RedisModuleCtx *ctx,  // redis module context
	GraphContext *gc,     // graph context
	const char *data,     // raw data
	size_t data_len       // number of bytes in data
) {
	size_t   data_idx   = 0 ;
	uint16_t prop_count = 0 ;
	uint64_t iterations = 0 ;
	Graph *g = GraphContext_GetGraph (gc) ;

	//--------------------------------------------------------------------------
	// parse CSV headers
	//--------------------------------------------------------------------------

	int *label_ids = _BulkInsert_ReadHeaderLabels (ctx, gc, SCHEMA_NODE, data,
			&data_idx) ;
	ASSERT (label_ids != NULL) ;

	uint n_lbl = arr_len (label_ids) ;

	// read the CSV header properties and collect their indices
	AttributeID *prop_indices = _BulkInsert_ReadHeaderProperties (ctx, gc,
			SCHEMA_NODE, data, &data_idx, &prop_count) ;

	//--------------------------------------------------------------------------
	// load nodes
	//--------------------------------------------------------------------------

	uint32_t batch_size = 0 ;
	const uint32_t batch_cap = 4096 ;

	Node nodes[batch_cap] ;         // batched nodes
	Node *p_nodes[batch_cap] ;      // pointer to nodes
	AttributeSet sets[batch_cap] ;  // attribute sets

	while (data_idx < data_len) {
		Node *n = nodes + batch_size ;
		p_nodes[batch_size] = n ;
		*n = GE_NEW_NODE () ;

		// read properties
		SIValue props[prop_count] ;
		AttributeID prop_attr_ids[prop_count] ;

		uint idx = 0 ;
		// read node properties
		for (uint i = 0; i < prop_count; i++) {
			SIValue v = _BulkInsert_ReadProperty (data, &data_idx) ;

			// skip null values
			if (unlikely (SI_TYPE (v) == T_NULL)) {
				continue ;
			}

			// accumulate attributes
			props[idx] = v ;
			prop_attr_ids[idx] = prop_indices[i] ;
			idx++ ;
		}

		// assign properties
		AttributeSet set = NULL ;
		AttributeSet_Add (&set, prop_attr_ids, props, idx, false) ;
		sets[batch_size] = set ;

		batch_size++ ;
		// flush batch
		if (batch_size == 4096) {
			GraphHub_CreateNodes (gc, p_nodes, sets, batch_size, label_ids,
					n_lbl, false) ;
			batch_size = 0 ;
		}

		// yield every 500,000 iterations
		if (iterations++ == 500000) {
			RedisModule_Yield (ctx, REDISMODULE_YIELD_FLAG_CLIENTS, NULL) ;
			iterations = 0 ;
		}
	}

	// flush last batch
	if (batch_size > 0) {
		GraphHub_CreateNodes (gc, p_nodes, sets, batch_size, label_ids, n_lbl,
				false) ;
		batch_size = 0 ;
	}

	// clean up
	if (prop_indices) {
		rm_free (prop_indices) ;
	}
	arr_free (label_ids) ;

	return BULK_OK ;
}

// Attribute sets move into the edge DataBlock while records are decoded, so
// each relation partition retains only the topology and stable ID.
typedef struct {
	EdgeID id ;
	NodeID src ;
	NodeID dest ;
} BulkEdge ;

typedef struct {
	RelationID relation ;
	BulkEdge *edges ;
} BulkEdgeGroup ;

// The matrix build has a fixed setup cost, so retain the existing delta path
// for small requests and use it only once it can be amortized by the batch.
#define BULK_REBUILD_MIN_EDGES 131072U

// Retain the proven per-file path outside the bounded transaction. At larger
// command sizes retaining every relation partition at once increases peak
// memory and can make the public bulk-loader path slower.
#define BULK_TRANSACTION_MAX_EDGES 1048576U

// Locate (or create) the one command-wide record partition for a relation.
// Multiple descriptor files for the same relation deliberately share it.
static BulkEdgeGroup *_BulkInsert_GetEdgeGroup
(
	BulkEdgeGroup **groups,
	RelationID relation,
	uint initial_capacity
) {
	ASSERT (groups != NULL) ;

	for (uint i = 0; i < arr_len (*groups); i++) {
		if ((*groups)[i].relation == relation) {
			return *groups + i ;
		}
	}

	BulkEdgeGroup group = {
		.relation = relation,
		.edges = arr_new (BulkEdge, initial_capacity),
	} ;
	arr_append (*groups, group) ;
	return *groups + arr_len (*groups) - 1 ;
}

// compare bulk edges by endpoint pair
static int _BulkInsert_BulkEdgeEndpointCmp
(
	const void *a,
	const void *b
) {
	const BulkEdge *ea = a ;
	const BulkEdge *eb = b ;

	if (ea->src != eb->src) {
		return (ea->src > eb->src) - (ea->src < eb->src) ;
	}

	return (ea->dest > eb->dest) - (ea->dest < eb->dest) ;
}

// A matrix can be replaced only when both its forward and transpose state are
// empty. Existing matrices use the regular delta-matrix insertion path below.
static bool _BulkInsert_MatrixCanBeRebuilt
(
	Delta_Matrix matrix
) {
	ASSERT (matrix != NULL) ;

	GrB_Index nvals = 0 ;
	GrB_OK (Delta_Matrix_nvals (&nvals, matrix)) ;
	if (nvals != 0 || !Delta_Matrix_Synced (matrix)) {
		return false ;
	}

	Delta_Matrix transpose = Delta_Matrix_getTranspose (matrix) ;
	if (transpose == NULL) {
		return true ;
	}

	GrB_OK (Delta_Matrix_nvals (&nvals, transpose)) ;
	return nvals == 0 && Delta_Matrix_Synced (transpose) ;
}

// Merge the already-built relation topologies into an empty adjacency matrix,
// then derive its transpose once. This avoids another command-wide coordinate
// array while still avoiding per-edge forward and transpose delta mutations.
static void _BulkInsert_BuildAdjacency
(
	Graph *g,
	Delta_Matrix matrix,
	const RelationID *relations,
	uint relation_count,
	GrB_Index matrix_dim
) {
	ASSERT (g != NULL) ;
	ASSERT (matrix != NULL) ;
	ASSERT (relations != NULL) ;
	ASSERT (relation_count > 0) ;
	ASSERT (_BulkInsert_MatrixCanBeRebuilt (matrix)) ;

	GrB_Matrix adjacency = NULL ;
	GrB_Matrix transpose = NULL ;
	GrB_OK (GrB_Matrix_new (&adjacency, GrB_BOOL, matrix_dim, matrix_dim)) ;
	GrB_OK (GrB_set (adjacency, GxB_SPARSE | GxB_HYPERSPARSE,
			GxB_SPARSITY_CONTROL)) ;

	for (uint i = 0; i < relation_count; i++) {
		Tensor relation = Graph_GetRelationMatrix (g, relations[i], false) ;
		Delta_Matrix relation_transpose = Delta_Matrix_getTranspose (relation) ;

		// Relation transposes are Boolean topology matrices, so their transpose
		// can be ORed directly into the Boolean all-relation adjacency matrix.
		GrB_OK (GrB_transpose (adjacency, NULL, GrB_LOR,
				Delta_Matrix_M (relation_transpose), NULL)) ;
	}

	GrB_OK (GrB_Matrix_new (&transpose, GrB_BOOL, matrix_dim, matrix_dim)) ;
	GrB_OK (GrB_set (transpose, GxB_SPARSE | GxB_HYPERSPARSE,
			GxB_SPARSITY_CONTROL)) ;
	GrB_OK (GrB_transpose (transpose, NULL, NULL, adjacency, NULL)) ;

	GrB_OK (Delta_Matrix_setM (matrix, &adjacency)) ;
	GrB_OK (Delta_Matrix_setM (Delta_Matrix_getTranspose (matrix),
			&transpose)) ;
}

// Build one empty relation tensor from sorted records. A relation matrix keeps
// a scalar edge ID for a single endpoint pair and a GraphBLAS vector for a
// multiedge pair; its transpose stores only topology.
static void _BulkInsert_BuildRelation
(
	Tensor matrix,
	const BulkEdge *edges,
	uint edge_count,
	GrB_Index matrix_dim,
	GrB_Index *rows,
	GrB_Index *cols,
	uint64_t *ids
) {
	ASSERT (matrix != NULL) ;
	ASSERT (edges  != NULL) ;
	ASSERT (edge_count > 0) ;
	ASSERT (rows != NULL) ;
	ASSERT (cols != NULL) ;
	ASSERT (ids  != NULL) ;
	ASSERT (_BulkInsert_MatrixCanBeRebuilt (matrix)) ;

	uint tuple_count = 0 ;
	for (uint i = 0; i < edge_count;) {
		uint j = i + 1 ;
		while (j < edge_count && edges[j].src == edges[i].src &&
				edges[j].dest == edges[i].dest) {
			j++ ;
		}

		rows[tuple_count] = edges[i].src ;
		cols[tuple_count] = edges[i].dest ;

		if (j == i + 1) {
			ids[tuple_count] = edges[i].id ;
		} else {
			GrB_Vector vector = NULL ;
			GrB_OK (GrB_Vector_new (&vector, GrB_BOOL, GrB_INDEX_MAX)) ;

			for (uint k = i; k < j; k++) {
				GrB_OK (GrB_Vector_setElement_BOOL (vector, true, edges[k].id)) ;
			}

			GrB_OK (GrB_wait (vector, GrB_MATERIALIZE)) ;
			ids[tuple_count] = SET_MSB ((uint64_t)(uintptr_t)vector) ;
		}

		tuple_count++ ;
		i = j ;
	}

	GrB_Matrix relation = NULL ;
	GrB_Matrix transpose = NULL ;
	GrB_Descriptor transpose_input = NULL ;
	GrB_OK (GrB_Matrix_new (&relation, GrB_UINT64, matrix_dim, matrix_dim)) ;
	GrB_OK (GrB_set (relation, GxB_SPARSE | GxB_HYPERSPARSE,
			GxB_SPARSITY_CONTROL)) ;
	GrB_OK (GrB_Matrix_build_UINT64 (relation, rows, cols, ids, tuple_count,
			GrB_SECOND_UINT64)) ;

	// The relation's scalar may be edge ID zero or a vector pointer. Apply a
	// constant-one operator while transposing so topology remains true-valued.
	GrB_OK (GrB_Matrix_new (&transpose, GrB_BOOL, matrix_dim, matrix_dim)) ;
	GrB_OK (GrB_set (transpose, GxB_SPARSE | GxB_HYPERSPARSE,
			GxB_SPARSITY_CONTROL)) ;
	GrB_OK (GrB_Descriptor_new (&transpose_input)) ;
	GrB_OK (GrB_Descriptor_set (transpose_input, GrB_INP0, GrB_TRAN)) ;
	GrB_OK (GrB_apply (transpose, NULL, NULL, GxB_ONE_UINT64, relation,
			transpose_input)) ;
	GrB_OK (GrB_Descriptor_free (&transpose_input)) ;

	GrB_OK (Delta_Matrix_setM (matrix, &relation)) ;
	GrB_OK (Delta_Matrix_setM (Delta_Matrix_getTranspose (matrix),
			&transpose)) ;
}

// The delta path uses the same Tensor_SetEdges semantics as Graph_CreateEdges.
// Fixed local batches avoid reintroducing command-wide Edge pointer staging.
// Batches may split a duplicate group; Tensor_SetEdges extends the vector
// installed by the preceding batch.
static void _BulkInsert_SetRelationDelta
(
	Tensor matrix,
	const BulkEdge *bulk_edges,
	uint edge_count
) {
	const uint batch_cap = 4096 ;
	Edge edges[batch_cap] ;
	const Edge *edge_ptrs[batch_cap] ;

	for (uint offset = 0; offset < edge_count; offset += batch_cap) {
		uint batch_size = MIN (batch_cap, edge_count - offset) ;
		for (uint i = 0; i < batch_size; i++) {
			const BulkEdge *bulk_edge = bulk_edges + offset + i ;
			Edge *edge = edges + i ;
			*edge = (Edge) {
				.id = bulk_edge->id,
				.src_id = bulk_edge->src,
				.dest_id = bulk_edge->dest,
			} ;
			edge_ptrs[i] = edge ;
		}
		Tensor_SetEdges (matrix, edge_ptrs, batch_size) ;
	}
}

// Bulk insertion deliberately has no undo/effect log; preserve only the
// GraphHub side effect required for schemas with edge indexes.
static void _BulkInsert_IndexEdges
(
	GraphContext *gc,
	Graph *g,
	RelationID relation,
	const BulkEdge *edges,
	uint edge_count
) {
	ASSERT (gc != NULL) ;
	ASSERT (g != NULL) ;
	ASSERT (edges != NULL) ;

	Schema *schema = GraphContext_GetSchemaByID (gc, relation, SCHEMA_EDGE) ;
	ASSERT (schema != NULL) ;
	if (!Schema_HasIndices (schema)) {
		return ;
	}

	for (uint i = 0; i < edge_count; i++) {
		const BulkEdge *bulk_edge = edges + i ;
		Edge edge = GE_NEW_LABELED_EDGE (NULL, relation) ;
		edge.id = bulk_edge->id ;
		edge.src_id = bulk_edge->src ;
		edge.dest_id = bulk_edge->dest ;
		edge.attributes = DataBlock_GetItem (g->edges, edge.id) ;
		ASSERT (edge.attributes != NULL) ;
		Schema_AddEdgeToIndex (schema, &edge) ;
	}
}

// Install relation-partitioned records. IDs are assigned in wire order while
// decoding; each partition is endpoint-sorted only after that property is fixed.
static void _BulkInsert_CommitEdges
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	BulkEdgeGroup *groups,
	uint group_count,
	uint64_t existing_edge_count,
	uint total_edge_count
) {
	ASSERT (ctx != NULL) ;
	ASSERT (gc != NULL) ;
	ASSERT (groups != NULL) ;
	ASSERT (group_count > 0) ;

	Graph *g = GraphContext_GetGraph (gc) ;

#ifdef RG_DEBUG
	for (uint i = 0; i < group_count; i++) {
		for (uint j = 0; j < arr_len (groups[i].edges); j++) {
			Node node = GE_NEW_NODE () ;
			ASSERT (Graph_GetNode (g, groups[i].edges[j].src,  &node)) ;
			ASSERT (Graph_GetNode (g, groups[i].edges[j].dest, &node)) ;
		}
	}
#endif

	// Resize every matrix once while still using the bulk resize policy. Matrix
	// updates below do not need repeated synchronization or resize checks.
	ASSERT (Graph_GetMatrixPolicy (g) == SYNC_POLICY_RESIZE) ;
	Delta_Matrix adjacency = Graph_GetAdjacencyMatrix (g, false) ;
	RelationID *relations = arr_new (RelationID, group_count) ;
	uint max_relation_edge_count = 0 ;
	bool rebuild_adjacency = total_edge_count >= BULK_REBUILD_MIN_EDGES &&
		existing_edge_count == 0 && _BulkInsert_MatrixCanBeRebuilt (adjacency) ;

	for (uint i = 0; i < group_count; i++) {
		BulkEdgeGroup *group = groups + i ;
		uint edge_count = arr_len (group->edges) ;
		if (edge_count == 0) {
			continue ;
		}

		Tensor relation = Graph_GetRelationMatrix (g, group->relation, false) ;
		arr_append (relations, group->relation) ;
		max_relation_edge_count = MAX (max_relation_edge_count, edge_count) ;
		rebuild_adjacency &= _BulkInsert_MatrixCanBeRebuilt (relation) ;
	}

	ASSERT (max_relation_edge_count > 0) ;
	MATRIX_POLICY policy = Graph_SetMatrixPolicy (g, SYNC_POLICY_NOP) ;
	GrB_Index *rows = NULL ;
	GrB_Index *cols = NULL ;
	uint64_t *ids = NULL ;
	if (rebuild_adjacency) {
		rows = rm_malloc (sizeof (GrB_Index) * max_relation_edge_count) ;
		cols = rm_malloc (sizeof (GrB_Index) * max_relation_edge_count) ;
		ids = rm_malloc (sizeof (uint64_t) * max_relation_edge_count) ;
	}
	GrB_Index matrix_dim = Graph_RequiredMatrixDim (g) ;

	for (uint i = 0; i < group_count; i++) {
		BulkEdgeGroup *group = groups + i ;
		BulkEdge *bulk_edges = group->edges ;
		uint edge_count = arr_len (bulk_edges) ;
		if (edge_count == 0) {
			continue ;
		}

		qsort (bulk_edges, edge_count, sizeof (BulkEdge),
				_BulkInsert_BulkEdgeEndpointCmp) ;
		Tensor relation = Graph_GetRelationMatrix (g, group->relation, false) ;

		if (rebuild_adjacency) {
			_BulkInsert_BuildRelation (relation, bulk_edges, edge_count,
					matrix_dim, rows, cols, ids) ;
		} else {
			_BulkInsert_SetRelationDelta (relation, bulk_edges, edge_count) ;
		}

		GraphStatistics_IncEdgeCount (&g->stats, group->relation, edge_count) ;

		if (!rebuild_adjacency) {
			// Existing graph state uses the regular mutation path. Each distinct
			// endpoint is set once, preserving pending additions and deletions.
			for (uint j = 0; j < edge_count;) {
				GrB_OK (Delta_Matrix_setElement_BOOL (adjacency, bulk_edges[j].src,
					bulk_edges[j].dest)) ;

				uint k = j + 1 ;
				while (k < edge_count && bulk_edges[k].src == bulk_edges[j].src &&
						bulk_edges[k].dest == bulk_edges[j].dest) {
					k++ ;
				}
				j = k ;
			}
		}

		_BulkInsert_IndexEdges (gc, g, group->relation, bulk_edges, edge_count) ;

		// A partition has no remaining consumers after its relation, optional
		// adjacency effects, and index side effects are installed.
		arr_free (group->edges) ;
		group->edges = NULL ;
		RedisModule_Yield (ctx, REDISMODULE_YIELD_FLAG_CLIENTS, NULL) ;
	}

	if (rows != NULL) {
		rm_free (ids) ;
		rm_free (cols) ;
		rm_free (rows) ;
	}

	if (rebuild_adjacency) {
		// Relation matrices now retain every detail needed for adjacency. Their
		// compact Boolean transposes materialize the all-relation matrix once.
		_BulkInsert_BuildAdjacency (g, adjacency, relations, arr_len (relations),
				matrix_dim) ;
	}

	arr_free (relations) ;
	Graph_SetMatrixPolicy (g, policy) ;
}

// process a single edge CSV file
static int _BulkInsert_ProcessEdgeFile
(
	RedisModuleCtx *ctx,  // redis module context
	GraphContext *gc,     // graph context
	const char *data,     // raw data
	size_t data_len       // number of bytes in data
) {
	size_t   data_idx   = 0 ;
	uint16_t prop_count = 0 ;
	uint64_t iterations = 0 ;
	Graph *g = GraphContext_GetGraph (gc) ;

	//--------------------------------------------------------------------------
	// parse CSV headers
	//--------------------------------------------------------------------------

	RelationID *rels = _BulkInsert_ReadHeaderLabels (ctx, gc, SCHEMA_EDGE, data,
			&data_idx) ;
	ASSERT (rels != NULL) ;

	uint type_count = arr_len (rels) ;

	// // edges must have exactly one type
	ASSERT (type_count == 1) ;
	RelationID rel = rels[0] ;

	AttributeID *prop_indices = _BulkInsert_ReadHeaderProperties (ctx, gc,
			SCHEMA_EDGE, data, &data_idx, &prop_count) ;

	//--------------------------------------------------------------------------
	// prepare matrices
	//--------------------------------------------------------------------------

	ASSERT (Graph_GetMatrixPolicy(g) == SYNC_POLICY_RESIZE) ;

	// warm up matrices to avoid resizes
	Graph_GetRelationMatrix (g, rel, false) ;
	Graph_GetAdjacencyMatrix (g, false) ;

	// temporarily disable sync policy
	MATRIX_POLICY policy = Graph_SetMatrixPolicy (g, SYNC_POLICY_NOP) ;

	//--------------------------------------------------------------------------
	// load edges
	//--------------------------------------------------------------------------

	Edge *edges = arr_new (Edge, 1) ;
	AttributeSet *sets = arr_new (AttributeSet, 1) ;

	SIValue props[prop_count] ;
	AttributeID prop_attr_ids[prop_count] ;

	while (data_idx < data_len) {
		Edge e = GE_NEW_LABELED_EDGE (NULL, rel) ;

		// read source ID
		NodeID src = *(NodeID*)&data[data_idx] ;
		data_idx += sizeof (NodeID) ;

		// read destination ID
		NodeID dest = *(NodeID*)&data[data_idx] ;
		data_idx += sizeof (NodeID) ;

		// accumulate edges
		Edge_SetSrcNodeID  (&e, src) ;
		Edge_SetDestNodeID (&e, dest) ;
		arr_append (edges, e) ;

		uint n = 0 ;
		// read edge properties
		for (uint i = 0; i < prop_count; i++) {
			SIValue v = _BulkInsert_ReadProperty (data, &data_idx) ;

			// skip null values
			if (unlikely (SI_TYPE (v) == T_NULL)) {
				continue ;
			}

			// accumulate attributes
			props[n] = v ;
			prop_attr_ids[n] = prop_indices[i] ;
			n++ ;
		}

		// assign properties
		AttributeSet set = NULL ;
		AttributeSet_Add (&set, prop_attr_ids, props, n, false) ;
		arr_append (sets, set) ;

		// yield every 500000 iterations
		if (iterations++ == 500000) {
			RedisModule_Yield (ctx, REDISMODULE_YIELD_FLAG_CLIENTS, NULL) ;
			iterations = 0 ;
		}
	}

	//--------------------------------------------------------------------------
	// commit edges
	//--------------------------------------------------------------------------

	uint n = arr_len (edges) ;
	if (n > 0) {
		Edge **pedges = arr_newlen (Edge*, n) ;

		for (uint i = 0; i < n; i++) {
			pedges[i] = edges + i ;
		}

		// yield just before we're creating the edges
		RedisModule_Yield (ctx, REDISMODULE_YIELD_FLAG_CLIENTS, NULL) ;

		GraphHub_CreateEdges (gc, pedges, rel, sets, false) ;

		arr_free (pedges) ;
	}

	//--------------------------------------------------------------------------
	// cleanup
	//--------------------------------------------------------------------------

	arr_free (rels) ;
	arr_free (sets) ;
	arr_free (edges) ;

	if (prop_indices) {
		rm_free (prop_indices) ;
	}

	Graph_SetMatrixPolicy (g, policy) ;

	return BULK_OK ;
}

// decode a single edge CSV file into the command-wide transaction buffer
static int _BulkInsert_ProcessEdgeFileTransaction
(
	RedisModuleCtx *ctx,  // redis module context
	GraphContext *gc,     // graph context
	const char *data,     // raw data
	size_t data_len,           // number of bytes in data
	BulkEdgeGroup **groups,    // relation-partitioned transaction buffers
	uint partition_capacity    // initial record capacity per relation
) {
	size_t   data_idx   = 0 ;
	uint16_t prop_count = 0 ;
	uint64_t iterations = 0 ;
	Graph *g = GraphContext_GetGraph (gc) ;

	//--------------------------------------------------------------------------
	// parse CSV headers
	//--------------------------------------------------------------------------

	RelationID *rels = _BulkInsert_ReadHeaderLabels (ctx, gc, SCHEMA_EDGE, data,
			&data_idx) ;
	ASSERT (rels != NULL) ;

	uint type_count = arr_len (rels) ;

	// edges must have exactly one type
	ASSERT (type_count == 1) ;
	RelationID rel = rels[0] ;

	AttributeID *prop_indices = _BulkInsert_ReadHeaderProperties (ctx, gc,
			SCHEMA_EDGE, data, &data_idx, &prop_count) ;

	//--------------------------------------------------------------------------
	// decode edges
	//--------------------------------------------------------------------------

	SIValue props[prop_count] ;
	AttributeID prop_attr_ids[prop_count] ;
	BulkEdgeGroup *group = NULL ;

	while (data_idx < data_len) {
		BulkEdge edge = {
			.id = INVALID_ENTITY_ID,
		} ;

		// read source ID
		edge.src = *(NodeID*)&data[data_idx] ;
		data_idx += sizeof (NodeID) ;

		// read destination ID
		edge.dest = *(NodeID*)&data[data_idx] ;
		data_idx += sizeof (NodeID) ;

		uint n = 0 ;
		// read edge properties
		for (uint i = 0; i < prop_count; i++) {
			SIValue v = _BulkInsert_ReadProperty (data, &data_idx) ;

			// skip null values
			if (unlikely (SI_TYPE (v) == T_NULL)) {
				continue ;
			}

			// accumulate attributes
			props[n] = v ;
			prop_attr_ids[n] = prop_indices[i] ;
			n++ ;
		}

		// Allocate in decode order so deleted-slot reuse and edge IDs remain
		// exactly the same as Graph_CreateEdges without retaining attributes in
		// the command-wide topology buffer.
		AttributeSet set = NULL ;
		AttributeSet_Add (&set, prop_attr_ids, props, n, false) ;
		AttributeSet *slot = DataBlock_AllocateItem (g->edges, &edge.id) ;
		*slot = set ;

		if (group == NULL) {
			group = _BulkInsert_GetEdgeGroup (groups, rel, partition_capacity) ;
		}
		arr_append (group->edges, edge) ;

		// yield every 500000 iterations
		if (iterations++ == 500000) {
			RedisModule_Yield (ctx, REDISMODULE_YIELD_FLAG_CLIENTS, NULL) ;
			iterations = 0 ;
		}
	}

	arr_free (rels) ;
	if (prop_indices) {
		rm_free (prop_indices) ;
	}

	return BULK_OK ;
}

// Decode all relation files before committing their topology. This lets one
// transaction group coordinates across files while retaining parse-time yields.
static int _BulkInsert_ProcessEdgeTokens
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	int token_count,
	RedisModuleString **argv,
	uint edge_count
) {
	ASSERT (token_count > 0) ;

	// The transaction earns its setup and retained-record cost only in the
	// direct-build range. Outside that bounded range preserve the established
	// per-file path, including small and very large append batches.
	if (edge_count < BULK_REBUILD_MIN_EDGES ||
			edge_count > BULK_TRANSACTION_MAX_EDGES) {
		for (int i = 0; i < token_count; i++) {
			size_t len ;
			const char *data = RedisModule_StringPtrLen (argv[i], &len) ;
			if (_BulkInsert_ProcessEdgeFile (ctx, gc, data, len)
					!= BULK_OK) {
				return BULK_FAIL ;
			}
		}
		return BULK_OK ;
	}

	BulkEdgeGroup *groups = arr_new (BulkEdgeGroup, token_count) ;
	Graph *g = GraphContext_GetGraph (gc) ;
	uint64_t existing_edge_count = DataBlock_ItemCount (g->edges) ;
	uint partition_capacity = MAX ((uint)1, edge_count / (uint)token_count) ;
	int rc = BULK_OK ;

	for (int i = 0; i < token_count; i++) {
		size_t len ;
		const char *data = RedisModule_StringPtrLen (argv[i], &len) ;
		rc = _BulkInsert_ProcessEdgeFileTransaction (ctx, gc, data, len, &groups,
				partition_capacity) ;
		if (rc != BULK_OK) {
			break ;
		}

		// Match the old per-file commit yield while retaining the transaction.
		RedisModule_Yield (ctx, REDISMODULE_YIELD_FLAG_CLIENTS, NULL) ;
	}

	uint total_edge_count = 0 ;
	for (uint i = 0; i < arr_len (groups); i++) {
		total_edge_count += arr_len (groups[i].edges) ;
	}

	if (rc == BULK_OK && total_edge_count > 0) {
		RedisModule_Yield (ctx, REDISMODULE_YIELD_FLAG_CLIENTS, NULL) ;
		_BulkInsert_CommitEdges (ctx, gc, groups, arr_len (groups),
				existing_edge_count, total_edge_count) ;
	}

	for (uint i = 0; i < arr_len (groups); i++) {
		arr_free (groups[i].edges) ;
	}
	arr_free (groups) ;
	return rc ;
}

// Process node tokens. Edge tokens have a separate command-wide transaction
// path so their topology can be constructed after every relation file is read.
static int _BulkInsert_ProcessNodeTokens
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	int token_count,
	RedisModuleString **argv
) {
	for (int i = 0; i < token_count; i++) {
		size_t len ;
		// retrieve a pointer to the next binary stream and record its length
		const char *data = RedisModule_StringPtrLen (argv[i], &len) ;
		if (_BulkInsert_ProcessNodeFile (ctx, gc, data, len) != BULK_OK) {
			return BULK_FAIL ;
		}
	}

	return BULK_OK ;
}

// entry point for bulk insertion of nodes and edges
int BulkInsert
(
	RedisModuleCtx *ctx,       // redis context
	GraphContext *gc,          // graph context
	RedisModuleString **argv,  // arguments
	int argc,                  // number of arguments
	uint node_count,           // number of nodes
	uint edge_count            // number of edges
) {
	ASSERT (gc   != NULL) ;
	ASSERT (ctx  != NULL) ;
	ASSERT (argv != NULL) ;

	if (argc < 2) {
		RedisModule_ReplyWithError(ctx, "Bulk insert format error, \
				failed to parse bulk insert sections.");
		return BULK_FAIL;
	}

	//--------------------------------------------------------------------------
	// parse section token counts
	//--------------------------------------------------------------------------

	long long node_token_count;
	long long relation_token_count;

	if (RedisModule_StringToLongLong (*argv++, &node_token_count)
			!= REDISMODULE_OK) {
		RedisModule_ReplyWithError (ctx,
				"Error parsing number of node descriptor tokens.") ;
		return BULK_FAIL ;
	}

	if (RedisModule_StringToLongLong (*argv++, &relation_token_count)
			!= REDISMODULE_OK) {
		RedisModule_ReplyWithError(ctx,
				"Error parsing number of relation descriptor tokens.") ;
		return BULK_FAIL ;
	}

	argc -= 2 ;

	//--------------------------------------------------------------------------
	// validate all CSV headers before modifying any graph state
	// a bad identifier in CSV #N must not leave a half-constructed graph from
	// the already-processed CSV files #1 .. #N-1
	//--------------------------------------------------------------------------

	if (_BulkInsert_ValidateTokens (ctx, node_token_count, argv,
				SCHEMA_NODE) != BULK_OK) {
		return BULK_FAIL ;
	}

	if (_BulkInsert_ValidateTokens (ctx, relation_token_count,
				argv + node_token_count, SCHEMA_EDGE) != BULK_OK) {
		return BULK_FAIL ;
	}

	//--------------------------------------------------------------------------
	// prepare graph for bulk load
	//--------------------------------------------------------------------------

	// lock graph under write lock
	// allocate space for new nodes and edges
	// set graph sync policy to resize only
	GraphContext_AcquireWriteLock (gc) ;

	Graph *g = GraphContext_GetGraph (gc) ;
	Graph_AllocateNodes (g, node_count) ;
	Graph_AllocateEdges (g, edge_count) ;

	MATRIX_POLICY policy = Graph_SetMatrixPolicy (g, SYNC_POLICY_RESIZE) ;

	//--------------------------------------------------------------------------
	// process node tokens
	//--------------------------------------------------------------------------

	int res = BULK_OK ;
	if (node_token_count > 0) {
		ASSERT (argc >= node_token_count) ;

		if (_BulkInsert_ProcessNodeTokens (ctx, gc, node_token_count, argv)
				!= BULK_OK) {
			res = BULK_FAIL ;
			goto cleanup ;
		}

		argv += node_token_count ;
		argc -= node_token_count ;
	}

	//--------------------------------------------------------------------------
	// process edge tokens
	//--------------------------------------------------------------------------

	if (relation_token_count > 0) {
		ASSERT (argc >= relation_token_count) ;

		if (_BulkInsert_ProcessEdgeTokens (ctx, gc, relation_token_count, argv,
					edge_count) != BULK_OK) {
			res = BULK_FAIL ;
			goto cleanup ;
		}
		argv += relation_token_count ;
		argc -= relation_token_count ;
	}

	ASSERT (argc == 0) ;

cleanup:
	// reset graph sync policy
	Graph_SetMatrixPolicy (g, policy) ;
	GraphContext_ReleaseLock (gc) ;
	return res ;
}

