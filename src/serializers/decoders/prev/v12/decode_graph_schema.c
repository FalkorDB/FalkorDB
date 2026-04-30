/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v12.h"

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
		stopwords = arr_new(char *, stopwords_count);
		for (uint i = 0; i < stopwords_count; i++) {
			char *stopword = RedisModule_LoadStringBuffer(rdb, NULL);
			arr_append(stopwords, stopword);
		}
	}

	uint fields_count = RedisModule_LoadUnsigned(rdb);
	for(uint i = 0; i < fields_count; i++) {
		char   *field_name = RedisModule_LoadStringBuffer(rdb, NULL);
		double weight      = RedisModule_LoadDouble(rdb);
		bool   nostem      = RedisModule_LoadUnsigned(rdb);
		char   *phonetic   = RedisModule_LoadStringBuffer(rdb, NULL);

		if(!already_loaded) {
			IndexField field;
			AttributeID field_id = GraphContext_FindOrAddAttribute(gc,
					field_name, NULL);

			// create new index field
			IndexField_NewFullTextField(&field, field_name, field_id);

			// set field options
			IndexField_OptionsSetWeight(&field, weight);
			IndexField_OptionsSetStemming(&field, nostem);
			IndexField_OptionsSetPhonetic(&field, phonetic);

			Schema_AddIndex(&idx, s, &field);
		}

		RedisModule_Free(field_name);
		RedisModule_Free(phonetic);
	}

	if(!already_loaded) {
		ASSERT(idx != NULL);
		Index_SetLanguage(idx, language);
		if(stopwords != NULL) Index_SetStopwords(idx, &stopwords);
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
			AttributeID field_id = GraphContext_FindOrAddAttribute(gc, field_name, NULL);
			IndexField_NewRangeField(&field, field_name, field_id);
			Schema_AddIndex(&idx, s, &field);
		}
		RedisModule_Free(field_name);
	}

	// construct index structure
	if(!already_loaded) {
		Index_Disable(idx);
	}
}

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
	 * index type
	 * index data */

	Schema *s = NULL ;
	int id = RedisModule_LoadUnsigned (rdb) ;
	char *name = RedisModule_LoadStringBuffer (rdb, NULL) ;
	if (!already_loaded) {
		s = GraphContext_AddSchema (gc, name, type) ;	
		ASSERT (s != NULL) ;
		ASSERT (Schema_GetID (s) == id) ;
	}

	RedisModule_Free(name);

	uint index_count = RedisModule_LoadUnsigned (rdb) ;
	for (uint index = 0; index < index_count; index++) {
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

	return s ;
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

void RdbLoadGraphSchema_v12
(
	RedisModuleIO *rdb,
	GraphContext *gc
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

	bool already_loaded = GraphContext_SchemaCount (gc, SCHEMA_NODE) > 0 ;

	// load each node schema
	for (uint i = 0; i < schema_count; i ++) {
		_RdbLoadSchema (rdb, gc, SCHEMA_NODE, already_loaded) ;
	}

	// #Edge schemas
	schema_count = RedisModule_LoadUnsigned (rdb) ;

	// Load each edge schema
	for (uint i = 0; i < schema_count; i ++) {
		_RdbLoadSchema (rdb, gc, SCHEMA_EDGE, already_loaded) ;
	}
}

