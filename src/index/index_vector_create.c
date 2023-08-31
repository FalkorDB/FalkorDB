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
	const SIValue options,  // options
	uint32_t *dimension     // vector length
) {
	if(SI_TYPE(options) != T_MAP) {
		return false;
	}
	
	// expecting a map with the following fields:
	// {
	//     length:538,
	//     similarityFunction:'euclidean'
	//  }

	if(Map_KeyCount(options) != 2) {
		return false;
	}

	// extract fields from the map
	SIValue val;

	//--------------------------------------------------------------------------
	// extract vector length
	//--------------------------------------------------------------------------

	if(!MAP_GET(val, "dim", val) || SI_TYPE(val) != T_INT64) {
		return false;
	}
	*dimension = val.longval;

	//--------------------------------------------------------------------------
	// extract similarity function
	//--------------------------------------------------------------------------

	if(!MAP_GET(val, "similarityFunction", val) || SI_TYPE(val) != T_STRING) {
		return false;
	}

	// at the moment only euclidean distance is supported
	if(strcasecmp(val.stringval, "euclidean") != 0) {
		return false;
	}

	return true;
}

// create a vector index
//
// CREATE VECTOR INDEX FOR (n:Person) ON (n.embeddings) OPTIONS {
//     dim:538,
//     similarityFunction:'euclidean'
// }
Index Index_VectorCreate
(
	const char *label,            // label/relationship type
	GraphEntityType entity_type,  // entity type (node/edge)
	const char *attribute,        // attribute to index
	SIValue options               // index options
) {
	ASSERT(ctx != NULL);

	// arguments
	uint32_t dimension;  // vector length
	SchemaType schema_type =
		(entity_type == GETYPE_NODE) ?SCHEMA_NODE : SCHEMA_EDGE;

	//--------------------------------------------------------------------------
	// parse options
	//--------------------------------------------------------------------------

	if(!_parseOptions(options, &dimension)) {
		ErrorCtx_SetError(EMSG_IDX_INVALID_OPTIONS);
		return NULL;
	}

	//--------------------------------------------------------------------------
	// create index
	//--------------------------------------------------------------------------

	Index idx = NULL;
	GraphContext *gc = QueryCtx_GetGraphCtx();

	if(GraphContext_AddVectorIndex(&idx, gc, schema_type, label, attribute,
				dimension) == false) {
		ErrorCtx_SetError(EMSG_VECTOR_IDX_CREATE_FAIL);
		return NULL;
	}

	//--------------------------------------------------------------------------
	// populate index asynchornously
	//--------------------------------------------------------------------------

	Schema *s = GraphContext_GetSchema(gc, label, schema_type);
	ASSERT(s != NULL);

	Indexer_PopulateIndex(gc, s, idx);

	return idx;
}

