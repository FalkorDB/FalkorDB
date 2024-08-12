/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../value.h"
#include "../query_ctx.h"
#include "../index/indexer.h"
#include "../datatypes/map.h"
#include "../errors/errors.h"
#include "../graph/graphcontext.h"

// parse options
static bool _parseOptions
(
	const SIValue options,   // options
	uint32_t *dimension,     // vector length
	size_t *M,               // max outgoing edges
	size_t *efConstruction,  // construction parameter for HNSW
	size_t *efRuntime        // runtime parameter for HNSW
) {
	if(SI_TYPE(options) != T_MAP) {
		return false;
	}
	
	// expecting a map with the following fields:
	// {
	//     dimension:538,
	//     similarityFunction:'euclidean',
	//     M:16,
	//     efConstruction:200,
	//     efRuntime:10
	//  }

	if(Map_KeyCount(options) < 2) {
		return false;
	}

	// extract fields from the map
	SIValue val;

	//--------------------------------------------------------------------------
	// extract vector length
	//--------------------------------------------------------------------------

	if(!MAP_GET(options, "dimension", val) || SI_TYPE(val) != T_INT64) {
		return false;
	}
	*dimension = val.longval;

	//--------------------------------------------------------------------------
	// extract similarity function
	//--------------------------------------------------------------------------

	if(!MAP_GET(options, "similarityFunction", val) || SI_TYPE(val) != T_STRING) {
		return false;
	}

	// at the moment only euclidean distance is supported
	if(strcasecmp(val.stringval, "euclidean") != 0) {
		return false;
	}

	if(MAP_GET(options, "M", val)) {
		if(SI_TYPE(val) != T_INT64) {
			return false;
		}
		*M = val.longval;
	} else {
		*M = 16;
	}

	if(MAP_GET(options, "efConstruction", val)) {
		if(SI_TYPE(val) != T_INT64) {
			return false;
		}
		*efConstruction = val.longval;
	} else {
		*efConstruction = 200;
	}

	if(MAP_GET(options, "efRuntime", val)) {
		if(SI_TYPE(val) != T_INT64) {
			return false;
		}
		*efRuntime = val.longval;
	} else {
		*efRuntime = 10;
	}

	return true;
}

// create a vector index
//
// CREATE VECTOR INDEX FOR (n:Person) ON (n.embeddings) OPTIONS {
//     dimension:538,
//     similarityFunction:'euclidean'
// }
Index Index_VectorCreate
(
	const char *label,            // label/relationship type
	GraphEntityType entity_type,  // entity type (node/edge)
	const char *attr,             // attribute to index
	AttributeID attr_id,          // attribute id
	SIValue options               // index options
) {
	ASSERT(label != NULL);
	ASSERT(attr  != NULL);
	ASSERT(SI_TYPE(options) == T_MAP);
	ASSERT(entity_type == GETYPE_NODE || entity_type == GETYPE_EDGE);
	ASSERT(attr_id != ATTRIBUTE_ID_ALL && attr_id != ATTRIBUTE_ID_NONE);

	// arguments
	uint32_t dimension;     // vector length
	size_t M;               // max outgoing edges
	size_t efConstruction;  // construction parameter for HNSW
	size_t efRuntime;       // runtime parameter for HNSW

	// get schema
	SchemaType st = (entity_type == GETYPE_NODE) ?SCHEMA_NODE : SCHEMA_EDGE;
	GraphContext *gc = QueryCtx_GetGraphCtx();
	Schema *s = GraphContext_GetSchema(gc, label, st);
	ASSERT(s != NULL);

	//--------------------------------------------------------------------------
	// parse options
	//--------------------------------------------------------------------------

	if(!_parseOptions(options, &dimension, &M, &efConstruction, &efRuntime)) {
		ErrorCtx_SetError(EMSG_VECTOR_INDEX_INVALID_CONFIG);
		return NULL;
	}

	//--------------------------------------------------------------------------
	// create index
	//--------------------------------------------------------------------------

	// create index field
	IndexField field;
	IndexField_NewVectorField(&field, attr, attr_id, dimension, M, efConstruction, efRuntime);

	Index idx = NULL;
	Schema_AddIndex(&idx, s, &field);

	return idx;
}

