/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v14.h"
#include "../../../schema/schema.h"

static void _RdbDecodeIndexField
(
	SerializerIO rdb,
	char **name,           // index field name
	IndexFieldType *type,  // index field type
	double *weight,        // index field option weight
	bool *nostem,          // index field option nostem
	char **phonetic,       // index field option phonetic
	uint32_t *dimension    // index field option dimension
) {
	// format:
	// name
	// type
	// options:
	//   weight
	//   nostem
	//   phonetic
	//   dimension

	// decode field name
	*name = SerializerIO_ReadBuffer(rdb, NULL);

	// docode field type
	*type = SerializerIO_ReadUnsigned(rdb);

	//--------------------------------------------------------------------------
	// decode field options
	//--------------------------------------------------------------------------

	// decode field weight
	*weight = SerializerIO_ReadDouble(rdb);

	// decode field nostem
	*nostem = SerializerIO_ReadUnsigned(rdb);

	// decode field phonetic
	*phonetic = SerializerIO_ReadBuffer(rdb, NULL);

	// decode field dimension
	*dimension = SerializerIO_ReadUnsigned(rdb);
}

static void _RdbLoadIndex
(
	SerializerIO rdb,
	GraphContext *gc,
	Schema *s,
	bool already_loaded
) {
	/* Format:
	 * language
	 * #stopwords - N
	 * N * stopword
	 * #properties - M
	 * M * property: {options} */

	Index idx        = NULL;
	char *language   = SerializerIO_ReadBuffer(rdb, NULL);
	char **stopwords = NULL;
	
	uint stopwords_count = SerializerIO_ReadUnsigned(rdb);
	if(stopwords_count > 0) {
		stopwords = array_new(char *, stopwords_count);
		for (uint i = 0; i < stopwords_count; i++) {
			char *stopword = SerializerIO_ReadBuffer(rdb, NULL);
			array_append(stopwords, stopword);
		}
	}

	uint fields_count = SerializerIO_ReadUnsigned(rdb);
	for(uint i = 0; i < fields_count; i++) {
		IndexFieldType type;
		double         weight;
		bool           nostem;
		char*          phonetic;
		char*          field_name;
		uint32_t       dimension;

		_RdbDecodeIndexField(rdb, &field_name, &type, &weight, &nostem,
				&phonetic, &dimension);

		if(!already_loaded) {
			IndexField field;
			AttributeID field_id = GraphContext_FindOrAddAttribute(gc,
					field_name, NULL);

			// create new index field
			IndexField_Init(&field, field_name, field_id, type);

			// set field options
			IndexField_SetOptions(&field, weight, nostem, phonetic, dimension);

			// add field to index
			Schema_AddIndex(&idx, s, &field);
		}

		RedisModule_Free(field_name);
		RedisModule_Free(phonetic);
	}

	if(!already_loaded) {
		ASSERT(idx != NULL);

		Index_SetLanguage(idx, language);
		if(stopwords != NULL) Index_SetStopwords(idx, &stopwords);

		// disable and create index structure
		// must be enabled once the graph is fully loaded
		Index_Disable(idx);
	}
	
	// free language
	RedisModule_Free(language);
}

static void _RdbLoadConstaint
(
	SerializerIO rdb,
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

	ConstraintType t = SerializerIO_ReadUnsigned(rdb);

	//--------------------------------------------------------------------------
	// decode constraint fields count
	//--------------------------------------------------------------------------
	
	uint8_t n = SerializerIO_ReadUnsigned(rdb);

	//--------------------------------------------------------------------------
	// decode constraint fields
	//--------------------------------------------------------------------------

	AttributeID attr_ids[n];
	const char *attr_strs[n];

	// read fields
	for(uint8_t i = 0; i < n; i++) {
		AttributeID attr = SerializerIO_ReadUnsigned(rdb);
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
	SerializerIO rdb,
	GraphContext *gc,    // graph context
	Schema *s,           // schema to populate
	bool already_loaded  // constraints already loaded
) {
	// read number of constraints
	uint constraint_count = SerializerIO_ReadUnsigned(rdb);

	for (uint i = 0; i < constraint_count; i++) {
		_RdbLoadConstaint(rdb, gc, s, already_loaded);
	}
}

static void _RdbLoadSchema
(
	SerializerIO rdb,
	GraphContext *gc,
	SchemaType type,
	bool already_loaded
) {
	/* Format:
	 * id
	 * name
	 * #indices
	 * (indexed property) X M 
	 * #constraints 
	 * (constraint type, constraint fields) X N
	 */

	Schema *s    = NULL;
	int     id   = SerializerIO_ReadUnsigned(rdb);
	char   *name = SerializerIO_ReadBuffer(rdb, NULL);

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

	uint index_count = SerializerIO_ReadUnsigned(rdb);
	for(uint index = 0; index < index_count; index++) {
		_RdbLoadIndex(rdb, gc, s, already_loaded);
	}

	//--------------------------------------------------------------------------
	// load constraints
	//--------------------------------------------------------------------------

	_RdbLoadConstaints(rdb, gc, s, already_loaded);
}

static void _RdbLoadAttributeKeys
(
	SerializerIO rdb,
	GraphContext *gc
) {
	/* Format:
	 * #attribute keys
	 * attribute keys
	 */

	uint count = SerializerIO_ReadUnsigned(rdb);
	for(uint i = 0; i < count; i ++) {
		char *attr = SerializerIO_ReadBuffer(rdb, NULL);
		GraphContext_FindOrAddAttribute(gc, attr, NULL);
		RedisModule_Free(attr);
	}
}

void RdbLoadGraphSchema_v14
(
	SerializerIO rdb,
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
	uint schema_count = SerializerIO_ReadUnsigned(rdb);

	// Load each node schema
	gc->node_schemas = array_ensure_cap(gc->node_schemas, schema_count);
	for(uint i = 0; i < schema_count; i ++) {
		_RdbLoadSchema(rdb, gc, SCHEMA_NODE, already_loaded);
	}

	// #Edge schemas
	schema_count = SerializerIO_ReadUnsigned(rdb);

	// Load each edge schema
	gc->relation_schemas = array_ensure_cap(gc->relation_schemas, schema_count);
	for(uint i = 0; i < schema_count; i ++) {
		_RdbLoadSchema(rdb, gc, SCHEMA_EDGE, already_loaded);
	}
}

