/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "../util/rmalloc.h"
#include "../graph/graphcontext.h"
#include "../datatypes/datatypes.h"

// create range index
//
// CREATE INDEX ON :label(attr)
// CREATE INDEX FOR (n:label) ON (n.attr)
// CREATE RANGE INDEX FOR (n:label) ON (n.attr)
Index Index_RangeCreate
(
	const char *label,            // label/relationship type
	GraphEntityType entity_type,  // entity type (node/edge)
	const char *attr,             // attribute to index
	AttributeID attr_id           // attribute id
) {
	ASSERT(label       != NULL);
	ASSERT(attr        != NULL);
	ASSERT(entity_type != GETYPE_UNKNOWN);

	GraphContext *gc = QueryCtx_GetGraphCtx();

	//--------------------------------------------------------------------------
	// try to build index
	//--------------------------------------------------------------------------

	IndexField field;
	IndexField_NewRangeField(&field, attr, attr_id);

	// get schema
	SchemaType st = (entity_type == GETYPE_NODE) ?SCHEMA_NODE : SCHEMA_EDGE;
	Schema *s = GraphContext_GetSchema(gc, label, st);
	ASSERT(s != NULL);

	Index idx = NULL;
	Schema_AddIndex(&idx, s, &field);
	return idx;
}

