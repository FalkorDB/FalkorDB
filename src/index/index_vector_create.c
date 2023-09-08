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
	//     dim:538,
	//     similarityFunction:'euclidean'
	//  }

	if(Map_KeyCount(options) < 2) {
		return false;
	}

	// extract fields from the map
	SIValue val;

	//--------------------------------------------------------------------------
	// extract vector length
	//--------------------------------------------------------------------------

	if(!MAP_GET(options, "dim", val) || SI_TYPE(val) != T_INT64) {
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
	const char *attr,             // attribute to index
	Attribute_ID attr_id,         // attribute id
	SIValue options               // index options
) {
	ASSERT(label != NULL);
	ASSERT(attr  != NULL);
	ASSERT(SI_TYPE(options) == T_MAP);
	ASSERT(entity_type == GETYPE_NODE || entity_type == GETYPE_EDGE);
	ASSERT(attr_id != ATTRIBUTE_ID_ALL && attr_id != ATTRIBUTE_ID_NONE);

	// arguments
	uint32_t dimension;  // vector length

	// get schema
	SchemaType st = (entity_type == GETYPE_NODE) ?SCHEMA_NODE : SCHEMA_EDGE;
	GraphContext *gc = QueryCtx_GetGraphCtx();
	Schema *s = GraphContext_GetSchema(gc, label, st);
	ASSERT(s != NULL);

	//--------------------------------------------------------------------------
	// parse options
	//--------------------------------------------------------------------------

	if(!_parseOptions(options, &dimension)) {
		ErrorCtx_SetError(EMSG_IDX_INVALID_CONFIG);
		return NULL;
	}

	//--------------------------------------------------------------------------
	// create index
	//--------------------------------------------------------------------------

	// create index field
	IndexField field;
	IndexField_NewVectorField(&field, attr, attr_id, dimension);

	Index idx = NULL;
	Schema_AddIndex(&idx, s, &field);

	return idx;
}

