/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "decode_v13.h"
#include "../../../../schema/schema.h"

static void _RdbLoadFullTextIndex
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	Schema *s,
	bool already_loaded
) {
	/* Format:
	 * language
	 * #stopwords - N
	 * N * stopword
	 * #properties - M
	 * M * property: {name, weight, nostem, phonetic} */

	Index idx        = NULL;
	char *language   = RedisModule_LoadStringBuffer(rdb, NULL);
	char **stopwords = NULL;
	
	uint stopwords_count = RedisModule_LoadUnsigned(rdb);
	if(stopwords_count > 0) {
		stopwords = array_new(char *, stopwords_count);
		for (uint i = 0; i < stopwords_count; i++) {
			char *stopword = RedisModule_LoadStringBuffer(rdb, NULL);
			array_append(stopwords, stopword);
		}
	}

	uint fields_count = RedisModule_LoadUnsigned(rdb);
	for(uint i = 0; i < fields_count; i++) {
		char    *field_name  =  RedisModule_LoadStringBuffer(rdb, NULL);
		double  weight       =  RedisModule_LoadDouble(rdb);
		bool    nostem       =  RedisModule_LoadUnsigned(rdb);
		char    *phonetic    =  RedisModule_LoadStringBuffer(rdb, NULL);

		if(!already_loaded) {
			IndexField field;
			Attribute_ID field_id = GraphContext_FindOrAddAttribute(gc, field_name, NULL);
			IndexField_New(&field, field_id, field_name, weight, nostem, phonetic);
			Schema_AddIndex(&idx, s, &field, IDX_FULLTEXT);
		}

		RedisModule_Free(field_name);
		RedisModule_Free(phonetic);
	}

	if(!already_loaded) {
		ASSERT(idx != NULL);
		Index_SetLanguage(idx, language);
		Index_SetStopwords(idx, stopwords);
		// disable and create index structure
		// must be enabled once the graph is fully loaded
		Index_Disable(idx);
	}
	
	// free language
	RedisModule_Free(language);
}

static void _RdbLoadExactMatchIndex
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	Schema *s,
	bool already_loaded
) {
	/* Format:
	 * #properties - M
	 * M * property */

	Index idx = NULL;
	uint fields_count = RedisModule_LoadUnsigned(rdb);
	for(uint i = 0; i < fields_count; i++) {
		char *field_name = RedisModule_LoadStringBuffer(rdb, NULL);
		if(!already_loaded) {
			IndexField field;
			Attribute_ID field_id = GraphContext_GetAttributeID(gc, field_name);
			IndexField_New(&field, field_id, field_name, INDEX_FIELD_DEFAULT_WEIGHT,
				INDEX_FIELD_DEFAULT_NOSTEM, INDEX_FIELD_DEFAULT_PHONETIC);
			Schema_AddIndex(&idx, s, &field, IDX_EXACT_MATCH);
		}
		RedisModule_Free(field_name);
	}

	if(!already_loaded) {
		// disable index, internally creates the RediSearch index structure
		// must be enabled once the graph is fully loaded
		Index_Disable(idx);
	}
}

static void _RdbLoadConstaint
(
	RedisModuleIO *rdb,
	GraphContext *gc,    // graph context
	Schema *s,           // schema to populate
	bool already_loaded  // constraints already loaded
) {
	/* Format:
	 * constraint type
	 * fields count
	 * field IDs */

	Constraint c = NULL;

	//--------------------------------------------------------------------------
	// decode constraint type
	//--------------------------------------------------------------------------

	ConstraintType t = RedisModule_LoadUnsigned(rdb);

	//--------------------------------------------------------------------------
	// decode constraint fields count
	//--------------------------------------------------------------------------
	
	uint8_t n = RedisModule_LoadUnsigned(rdb);

	//--------------------------------------------------------------------------
	// decode constraint fields
	//--------------------------------------------------------------------------

	Attribute_ID attr_ids[n];
	const char *attr_strs[n];

	// read fields
	for(uint8_t i = 0; i < n; i++) {
		Attribute_ID attr = RedisModule_LoadUnsigned(rdb);
		attr_ids[i]  = attr;
		attr_strs[i] = GraphContext_GetAttributeString(gc, attr);
	}

	if(!already_loaded) {
		GraphEntityType et = (Schema_GetType(s) == SCHEMA_NODE) ?
			GETYPE_NODE : GETYPE_EDGE;

		c = Constraint_New((struct GraphContext*)gc, t, Schema_GetID(s),
				attr_ids, attr_strs, n, et, NULL);

		// set constraint status to active
		// only active constraints are encoded
		Constraint_SetStatus(c, CT_ACTIVE);

		// check if constraint already contained in schema
		ASSERT(!Schema_ContainsConstraint(s, t, attr_ids, n));

		// add constraint to schema
		Schema_AddConstraint(s, c);
	}
}

// load schema's constraints
static void _RdbLoadConstaints
(
	RedisModuleIO *rdb,
	GraphContext *gc,    // graph context
	Schema *s,           // schema to populate
	bool already_loaded  // constraints already loaded
) {
	// read number of constraints
	uint constraint_count = RedisModule_LoadUnsigned(rdb);

	for (uint i = 0; i < constraint_count; i++) {
		_RdbLoadConstaint(rdb, gc, s, already_loaded);
	}
}

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
	 * (index type, indexed property) X M 
	 * #constraints 
	 * (constraint type, constraint fields) X N
	 */

	Schema *s    = NULL;
	int     id   = RedisModule_LoadUnsigned(rdb);
	char   *name = RedisModule_LoadStringBuffer(rdb, NULL);

	if(!already_loaded) {
		s = Schema_New(type, id, name);
		if(type == SCHEMA_NODE) {
			ASSERT(array_len(gc->node_schemas) == id);
			array_append(gc->node_schemas, s);
		} else {
			ASSERT(array_len(gc->relation_schemas) == id);
			array_append(gc->relation_schemas, s);
		}
	}

	RedisModule_Free(name);

	//--------------------------------------------------------------------------
	// load indices
	//--------------------------------------------------------------------------

	uint index_count = RedisModule_LoadUnsigned(rdb);
	for(uint index = 0; index < index_count; index++) {
		IndexType index_type = RedisModule_LoadUnsigned(rdb);

		switch(index_type) {
			case IDX_FULLTEXT:
				_RdbLoadFullTextIndex(rdb, gc, s, already_loaded);
				break;
			case IDX_EXACT_MATCH:
				_RdbLoadExactMatchIndex(rdb, gc, s, already_loaded);
				break;
			default:
				ASSERT(false);
				break;
		}
	}

	//--------------------------------------------------------------------------
	// load constraints
	//--------------------------------------------------------------------------

	_RdbLoadConstaints(rdb, gc, s, already_loaded);
}

static void _RdbLoadAttributeKeys(RedisModuleIO *rdb, GraphContext *gc) {
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

void RdbLoadGraphSchema_v13
(
	RedisModuleIO *rdb,
	GraphContext *gc,
	bool already_loaded
) {
	/* Format:
	 * attribute keys (unified schema)
	 * #node schemas
	 * node schema X #node schemas
	 * #relation schemas
	 * unified relation schema
	 * relation schema X #relation schemas
	 */

	// Attributes, Load the full attribute mapping.
	_RdbLoadAttributeKeys(rdb, gc);

	// #Node schemas
	uint schema_count = RedisModule_LoadUnsigned(rdb);

	// Load each node schema
	gc->node_schemas = array_ensure_cap(gc->node_schemas, schema_count);
	for(uint i = 0; i < schema_count; i ++) {
		_RdbLoadSchema(rdb, gc, SCHEMA_NODE, already_loaded);
	}

	// #Edge schemas
	schema_count = RedisModule_LoadUnsigned(rdb);

	// Load each edge schema
	gc->relation_schemas = array_ensure_cap(gc->relation_schemas, schema_count);
	for(uint i = 0; i < schema_count; i ++) {
		_RdbLoadSchema(rdb, gc, SCHEMA_EDGE, already_loaded);
	}
}

