/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v10.h"

static void _RdbLoadSchema
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
	Schema *s = NULL ;

	if (!already_loaded) {
		s = GraphContext_AddSchema (gc, name, type) ;
		ASSERT (s != NULL) ;
		ASSERT (Schema_GetID (s) == id) ;
	}

	RedisModule_Free (name) ;

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
	_RdbLoadAttributeKeys (rdb, gc) ;

	// #Node schemas
	uint schema_count = RedisModule_LoadUnsigned (rdb) ;

	GraphDecodeContext *decoding_context = GraphContext_GetDecodingCtx (gc) ;
	bool already_loaded =
		GraphDecodeContext_GetProcessedKeyCount (decoding_context) > 0 ;

	// load each node schema
	for (uint i = 0; i < schema_count; i ++) {
		_RdbLoadSchema (rdb, gc, SCHEMA_NODE, already_loaded) ;
	}

	// #edge schemas
	schema_count = RedisModule_LoadUnsigned (rdb) ;

	// load each edge schema
	for (uint i = 0; i < schema_count; i ++) {
		_RdbLoadSchema (rdb, gc, SCHEMA_EDGE, already_loaded) ;
	}
}

