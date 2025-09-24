/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v10.h"

static Schema *_RdbLoadSchema
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	SchemaType type,
	bool already_loaded
) {
	/* Format:
	 * id
	 * name
	 * #indices
	 * (index type, indexed property) X M */

	int id = RedisModule_LoadUnsigned(rdb);
	char *name = RedisModule_LoadStringBuffer(rdb, NULL);
	Schema *s = already_loaded ? NULL : Schema_New(type, id, name);
	RedisModule_Free(name);

	Index idx = NULL;
	uint index_count = RedisModule_LoadUnsigned(rdb);
	for(uint i = 0; i < index_count; i++) {
		IndexType type = RedisModule_LoadUnsigned(rdb);
		char *field_name = RedisModule_LoadStringBuffer(rdb, NULL);

		// skip if we've already loaded this schema
		if(already_loaded) {
			// skip field name
			RedisModule_Free(field_name);
			continue;
		}

		IndexField field;
		AttributeID field_id = GraphContext_FindOrAddAttribute(gc, field_name,
				NULL);

		if(type == IDX_EXACT_MATCH) {
			IndexField_NewRangeField(&field, field_name, field_id);
		} else if(type == IDX_FULLTEXT) {
			IndexField_NewFullTextField(&field, field_name, field_id);
		} else {
			// error
			RedisModule_LogIOError(rdb, "warning", "unknown index type %d",
					type);
			assert(false);
		}

		Schema_AddIndex(&idx, s, &field);
		RedisModule_Free(field_name);
	}

	if(s) {
		// no entities are expected to be in the graph in this point in time
		if(PENDING_IDX(s)) {
			Index_Disable(PENDING_IDX(s));
		}
	}

	return s;
}

static void _RdbLoadAttributeKeys
(
	RedisModuleIO *rdb,
	GraphContext *gc
) {
	/* Format:
	 * #attribute keys
	 * attribute keys
	 */

	uint count = RedisModule_LoadUnsigned(rdb);
	for(uint i = 0; i < count; i ++) {
		char *attr = RedisModule_LoadStringBuffer(rdb, NULL);
		GraphContext_FindOrAddAttribute(gc, attr, NULL);
		RedisModule_Free(attr);
	}
}

void RdbLoadGraphSchema_v10
(
	RedisModuleIO *rdb,
	GraphContext *gc
) {
	/* Format:
	 * attributes
	 * #node schemas - N
	 * N * node schema
	 * #relation schemas - M
	 * M * relation schema
	 */

	// Attributes, Load the full attribute mapping.
	_RdbLoadAttributeKeys(rdb, gc);

	// #Node schemas
	uint schema_count = RedisModule_LoadUnsigned(rdb);

	bool already_loaded =
		GraphDecodeContext_GetProcessedKeyCount(gc->decoding_context) > 0;

	// load each node schema
	gc->node_schemas = array_ensure_cap(gc->node_schemas, schema_count);
	for(uint i = 0; i < schema_count; i ++) {
		Schema *s = _RdbLoadSchema(rdb, gc, SCHEMA_NODE, already_loaded);
		if(!already_loaded) array_append(gc->node_schemas, s);
	}

	// #edge schemas
	schema_count = RedisModule_LoadUnsigned(rdb);

	// load each edge schema
	gc->relation_schemas = array_ensure_cap(gc->relation_schemas, schema_count);
	for(uint i = 0; i < schema_count; i ++) {
		Schema *s = _RdbLoadSchema(rdb, gc, SCHEMA_EDGE, already_loaded);
		if(!already_loaded) array_append(gc->relation_schemas, s);
	}
}

