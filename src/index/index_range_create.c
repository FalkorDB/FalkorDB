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

// create range index
//
// CREATE INDEX ON :Person(name)
// CREATE INDEX FOR (p:Person) ON (name)
// CREATE RANGE INDEX FOR (p:Person) ON (name)
Index Index_RangeCreate
(
	const char *label,            // label/relationship type
	GraphEntityType entity_type,  // entity type (node/edge)
	const char **fields,          // fields to index
	uint nfields	              // number of fields to index
) {
	ASSERT(label       != NULL);
	ASSERT(fields      != NULL);
	ASSERT(nfields     > 0);
	ASSERT(entity_type != GETYPE_UNKNOWN);

	Index        idx = NULL;
	GraphContext *gc = QueryCtx_GetGraphCtx();
	SchemaType   st  = (entity_type == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE;

	// make sure fields aren't already indexed
	for(uint i = 0; i < nfields; i++) {
		Attribute_ID attr_id = GraphContext_GetAttributeID(gc, fields[i]);
		if(attr_id == ATTRIBUTE_ID_NONE) continue;

		if(GraphContext_GetIndex(gc, label, &attr_id, 1, INDEX_FLD_RANGE, st)) {
			ErrorCtx_SetError(EMSG_INDEX_FIELD_ALREADY_EXISTS);
			return NULL;
		}
	}

	//--------------------------------------------------------------------------
	// create index
	//--------------------------------------------------------------------------

	bool res = GraphContext_AddRangeIndex(&idx, gc, st, label, fields, nfields);
	ASSERT(res == true);

	//--------------------------------------------------------------------------
	// populate index asynchornously
	//--------------------------------------------------------------------------

	Schema *s = GraphContext_GetSchema(gc, label, st);
	ASSERT(s != NULL);

	Indexer_PopulateIndex(gc, s, idx);

	return idx;
}

