/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
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
		if(!already_loaded) {
			IndexField field;
			Attribute_ID field_id = GraphContext_FindOrAddAttribute(gc, field_name, NULL);
			IndexField_New(&field, field_id, field_name, INDEX_FIELD_DEFAULT_WEIGHT,
					INDEX_FIELD_DEFAULT_NOSTEM, INDEX_FIELD_DEFAULT_PHONETIC);
			Schema_AddIndex(&idx, s, &field, type);
		}
		RedisModule_Free(field_name);
	}

	if(s) {
		// no entities are expected to be in the graph in this point in time
		if(PENDING_EXACTMATCH_IDX(s)) {
			Index_Disable(PENDING_EXACTMATCH_IDX(s));
		}

		if(PENDING_FULLTEXT_IDX(s) != NULL) {
			Index_Disable(PENDING_FULLTEXT_IDX(s));
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

	// Load each node schema
	gc->node_schemas = array_ensure_cap(gc->node_schemas, schema_count);
	for(uint i = 0; i < schema_count; i ++) {
		Schema *s = _RdbLoadSchema(rdb, gc, SCHEMA_NODE, already_loaded);
		if(!already_loaded) array_append(gc->node_schemas, s);
	}

	// #Edge schemas
	schema_count = RedisModule_LoadUnsigned(rdb);

	// Load each edge schema
	gc->relation_schemas = array_ensure_cap(gc->relation_schemas, schema_count);
	for(uint i = 0; i < schema_count; i ++) {
		Schema *s = _RdbLoadSchema(rdb, gc, SCHEMA_EDGE, already_loaded);
		if(!already_loaded) array_append(gc->relation_schemas, s);
	}
}
