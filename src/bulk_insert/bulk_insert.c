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

#include <string.h>

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
	GraphContext* gc,
	SchemaType t,
	const char* data,
	size_t* data_idx
) {
	ASSERT (gc       != NULL) ;
	ASSERT (data     != NULL) ;
	ASSERT (data_idx != NULL) ;

	// first sequence is entity label(s)
	const char *labels = data + *data_idx ;
	size_t labels_len = strlen (labels) ;
	*data_idx += labels_len + 1 ;

	// array of all label IDs
	int *label_ids = array_new(int, 1) ;
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

		// try to retrieve the label's schema
		Schema *s = GraphContext_GetSchema (gc, label, t) ;
		// create the schema if it does not already exist
		if (s == NULL) {
			s = GraphContext_AddSchema (gc, label, t) ;
		}
		ASSERT (s != NULL) ;

		// store the label ID
		array_append (label_ids, Schema_GetID (s)) ;

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
	GraphContext *gc,
	SchemaType t,
	const char *data,
	size_t *data_idx,
	uint16_t *prop_count
) {
	ASSERT (gc         != NULL) ;
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

	int *label_ids = _BulkInsert_ReadHeaderLabels (gc, SCHEMA_NODE, data,
			&data_idx) ;
	ASSERT (label_ids != NULL) ;

	uint n_lbl = array_len (label_ids) ;

	// read the CSV header properties and collect their indices
	AttributeID *prop_indices = _BulkInsert_ReadHeaderProperties (gc,
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
	array_free (label_ids) ;

	return BULK_OK ;
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

	//--------------------------------------------------------------------------
	// parse CSV headers
	//--------------------------------------------------------------------------

	RelationID *rels = _BulkInsert_ReadHeaderLabels (gc, SCHEMA_EDGE, data,
			&data_idx) ;
	uint type_count = array_len (rels) ;

	// // edges must have exactly one type
	ASSERT (type_count == 1) ;
	RelationID rel = rels[0] ;

	AttributeID *prop_indices = _BulkInsert_ReadHeaderProperties (gc,
			SCHEMA_EDGE, data, &data_idx, &prop_count) ;

	//--------------------------------------------------------------------------
	// prepare matrices
	//--------------------------------------------------------------------------

	ASSERT (Graph_GetMatrixPolicy(gc->g) == SYNC_POLICY_RESIZE) ;

	// warm up matrices to avoid resizes
	Graph_GetRelationMatrix (gc->g, rel, false) ;
	Graph_GetAdjacencyMatrix (gc->g, false) ;

	// temporarily disable sync policy
	MATRIX_POLICY policy = Graph_SetMatrixPolicy (gc->g, SYNC_POLICY_NOP) ;

	//--------------------------------------------------------------------------
	// load edges
	//--------------------------------------------------------------------------

	Edge *edges = array_new (Edge, 1) ;
	AttributeSet *sets = array_new (AttributeSet, 1) ;

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
		array_append (edges, e) ;

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
		array_append (sets, set) ;

		// yield every 500000 iterations
		if (iterations++ == 500000) {
			RedisModule_Yield (ctx, REDISMODULE_YIELD_FLAG_CLIENTS, NULL) ;
			iterations = 0 ;
		}
	}

	//--------------------------------------------------------------------------
	// commit edges
	//--------------------------------------------------------------------------

	uint n = array_len (edges) ;
	if (n > 0) {
		Edge **pedges = array_newlen (Edge*, n) ;

		for (uint i = 0; i < n; i++) {
			pedges[i] = edges + i ;
		}

		// yield just before we're creating the edges
		RedisModule_Yield (ctx, REDISMODULE_YIELD_FLAG_CLIENTS, NULL) ;

		GraphHub_CreateEdges (gc, pedges, rel, sets, false) ;

		array_free (pedges) ;
	}

	//--------------------------------------------------------------------------
	// cleanup
	//--------------------------------------------------------------------------

	array_free (rels) ;
	array_free (sets) ;
	array_free (edges) ;

	if (prop_indices) {
		rm_free (prop_indices) ;
	}

	Graph_SetMatrixPolicy (gc->g, policy) ;

	return BULK_OK ;
}

static int _BulkInsert_ProcessTokens
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	int token_count,
	RedisModuleString **argv,
	SchemaType type
) {
	for (int i = 0; i < token_count; i++) {
		size_t len ;
		// retrieve a pointer to the next binary stream and record its length
		const char *data = RedisModule_StringPtrLen (argv[i], &len) ;
		int rc = (type == SCHEMA_NODE)
			? _BulkInsert_ProcessNodeFile (ctx, gc, data, len)
			: _BulkInsert_ProcessEdgeFile (ctx, gc, data, len) ;
		UNUSED (rc) ;
		ASSERT (rc == BULK_OK) ;
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
	// prepare graph for bulk load
	//--------------------------------------------------------------------------

	Graph *g = gc->g ;
	int res = BULK_OK ;

	// lock graph under write lock
	// allocate space for new nodes and edges
	// set graph sync policy to resize only
	Graph_AcquireWriteLock (g) ;

	Graph_AllocateNodes (g, node_count) ;
	Graph_AllocateEdges (g, edge_count) ;

	MATRIX_POLICY policy = Graph_SetMatrixPolicy (g, SYNC_POLICY_RESIZE) ;

	//--------------------------------------------------------------------------
	// process node tokens
	//--------------------------------------------------------------------------

	if (node_token_count > 0) {
		ASSERT (argc >= node_token_count) ;

		if (_BulkInsert_ProcessTokens (ctx, gc, node_token_count, argv,
					SCHEMA_NODE) != BULK_OK) {
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

		if (_BulkInsert_ProcessTokens (ctx, gc, relation_token_count, argv,
					SCHEMA_EDGE) != BULK_OK) {
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
	Graph_ReleaseLock (g) ;
	return res ;
}

