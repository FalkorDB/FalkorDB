/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v18.h"
#include "../../../../schema/schema.h"

static bool _RdbDecodeIndexField
(
	SerializerIO io,
	char **name,             // index field name
	IndexFieldType *type,    // index field type
	double *weight,          // index field option weight
	bool *nostem,            // index field option nostem
	char **phonetic,         // index field option phonetic
	uint32_t *dimension,     // index field option dimension
	size_t *M,               // index field option M
	size_t *efConstruction,  // index field option efConstruction
	size_t *efRuntime,       // index field option efRuntime
	VecSimMetric *simFunc    // index field option similarity function
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
	if (!SerializerIO_TryReadBuffer (io, (void**)name, NULL)) {
		return false ;
	}

	// docode field type
	uint64_t v ;
	TRY_READ (io, v) ;
	*type = v ;

	//--------------------------------------------------------------------------
	// decode field options
	//--------------------------------------------------------------------------

	// decode field weight
	double w ;
	TRY_READ (io, w) ;
	*weight = w ;

	// decode field nostem
	TRY_READ (io, v) ;
	*nostem = v ;

	// decode field phonetic
	if (!SerializerIO_TryReadBuffer (io, (void**)phonetic, NULL)) {
		return false ;
	}

	// decode field dimension
	if (*type & INDEX_FLD_VECTOR) {
		TRY_READ (io, v) ;
		*dimension = v ;

		TRY_READ (io, v) ;
		*M = v ;

		TRY_READ (io, v) ;
		*efConstruction = v ;

		TRY_READ (io, v);
		*efRuntime = v ;

		TRY_READ (io, v) ;
		*simFunc = v;
	}

	return true ;
}

static bool _RdbLoadIndex
(
	SerializerIO io,
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

	Index idx        = NULL ;
	char **stopwords = NULL ;

	char *language ;
	if (!SerializerIO_TryReadBuffer (io, (void**)&language, NULL)) {
		return false;
	}
	
	uint64_t stopwords_count ;
	TRY_READ (io, stopwords_count) ;
	if (stopwords_count > 0) {
		stopwords = array_new (char *, stopwords_count) ;
		for (uint i = 0; i < stopwords_count; i++) {
			char *stopword ;
			if (!SerializerIO_TryReadBuffer (io, (void**)&stopword, NULL)) {
				// TODO: free stopwords
				return false ;
			}
			array_append (stopwords, stopword) ;
		}
	}

	uint64_t fields_count ;
	TRY_READ (io, fields_count) ;

	for (uint i = 0; i < fields_count; i++) {
		IndexFieldType type ;
		double         weight ;
		bool           nostem ;
		char*          phonetic ;
		char*          field_name ;
		uint32_t       dimension ;
		size_t		   M ;
		size_t         efConstruction ;
		size_t         efRuntime ;
		VecSimMetric   simFunc ;

		if (!_RdbDecodeIndexField (io, &field_name, &type, &weight, &nostem,
					&phonetic, &dimension, &M, &efConstruction, &efRuntime,
					&simFunc)) {
			return false ;
		}

		if (!already_loaded) {
			IndexField field ;
			AttributeID field_id = GraphContext_FindOrAddAttribute (gc,
					field_name, NULL) ;

			// create new index field
			IndexField_Init (&field, field_name, field_id, type) ;

			// set field options
			IndexField_SetOptions (&field, weight, nostem, phonetic, dimension) ;
			IndexField_OptionsSetM (&field, M) ;
			IndexField_OptionsSetEfConstruction (&field, efConstruction) ;
			IndexField_OptionsSetEfRuntime (&field, efRuntime) ;
			IndexField_OptionsSetSimFunc (&field, simFunc) ;

			// add field to index
			Schema_AddIndex (&idx, s, &field) ;
		}

		RedisModule_Free (field_name) ;
		RedisModule_Free (phonetic) ;
	}

	if (!already_loaded) {
		ASSERT (idx != NULL) ;

		Index_SetLanguage (idx, language) ;
		if (stopwords != NULL) {
			Index_SetStopwords (idx, &stopwords) ;
		}

		// disable and create index structure
		// must be enabled once the graph is fully loaded
		Index_Disable (idx) ;
	}
	
	// free language
	RedisModule_Free (language) ;

	return true ;
}

static bool _RdbLoadConstaint
(
	SerializerIO io,
	GraphContext *gc,    // graph context
	Schema *s,           // schema to populate
	bool already_loaded  // constraints already loaded
) {
	/* Format:
	 * constraint type
	 * fields count
	 * field IDs */

	Constraint c = NULL ;

	//--------------------------------------------------------------------------
	// decode constraint type
	//--------------------------------------------------------------------------

	uint64_t v ;
	TRY_READ (io, v) ;
	ConstraintType t = v ;

	//--------------------------------------------------------------------------
	// decode constraint fields count
	//--------------------------------------------------------------------------
	
	TRY_READ (io, v) ;
	uint8_t n = v ;

	//--------------------------------------------------------------------------
	// decode constraint fields
	//--------------------------------------------------------------------------

	AttributeID attr_ids[n] ;
	const char *attr_strs[n] ;

	// read fields
	for (uint8_t i = 0; i < n; i++) {
		TRY_READ (io, v) ;
		AttributeID attr = v ;

		attr_ids[i]  = attr ;
		attr_strs[i] = GraphContext_GetAttributeString (gc, attr) ;
	}

	if (!already_loaded) {
		GraphEntityType et = (Schema_GetType(s) == SCHEMA_NODE) ?
			GETYPE_NODE : GETYPE_EDGE ;

		c = Constraint_New ((struct GraphContext*)gc, t, Schema_GetID (s),
				attr_ids, attr_strs, n, et, NULL) ;

		// set constraint status to active
		// only active constraints are encoded
		Constraint_SetStatus (c, CT_ACTIVE) ;

		// check if constraint already contained in schema
		ASSERT(!Schema_ContainsConstraint (s, t, attr_ids, n)) ;

		// add constraint to schema
		Schema_AddConstraint (s, c) ;
	}

	return true ;
}

// load schema's constraints
static bool _RdbLoadConstaints
(
	SerializerIO io,
	GraphContext *gc,    // graph context
	Schema *s,           // schema to populate
	bool already_loaded  // constraints already loaded
) {
	// read number of constraints
	uint64_t constraint_count ;
	TRY_READ (io, constraint_count) ;

	for (uint i = 0; i < constraint_count; i++) {
		if (!_RdbLoadConstaint (io, gc, s, already_loaded)) {
			return false ;
		}
	}

	return true ;
}

static bool _RdbLoadSchema
(
	SerializerIO io,
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

	Schema *s = NULL ;

	uint64_t id ;
	TRY_READ (io, id) ;

	char *name ;
	if (!SerializerIO_TryReadBuffer (io, (void**)&name, NULL)) {
		return false ;
	}

	if (!already_loaded) {
		s = Schema_New (type, id, name) ;
		if (type == SCHEMA_NODE) {
			ASSERT (array_len(gc->node_schemas) == id) ;
			array_append (gc->node_schemas, s) ;
		} else {
			ASSERT (array_len(gc->relation_schemas) == id) ;
			array_append (gc->relation_schemas, s) ;
		}
	}

	RedisModule_Free (name) ;

	//--------------------------------------------------------------------------
	// load indices
	//--------------------------------------------------------------------------

	uint64_t index_count ;
	TRY_READ (io, index_count);

	for(uint index = 0; index < index_count; index++) {
		if (!_RdbLoadIndex (io, gc, s, already_loaded)) {
			return false ;
		}
	}

	//--------------------------------------------------------------------------
	// load constraints
	//--------------------------------------------------------------------------

	return _RdbLoadConstaints (io, gc, s, already_loaded) ;
}

static bool _RdbLoadAttributeKeys
(
	SerializerIO io,
	GraphContext *gc
) {
	/* Format:
	 * #attribute keys
	 * attribute keys
	 */

	uint64_t count ;
	TRY_READ (io, count) ;

	for (uint i = 0; i < count; i ++) {
		char *attr ;
		if (!SerializerIO_TryReadBuffer (io, (void**)&attr, NULL)) {
			return false ;
		}

		GraphContext_FindOrAddAttribute (gc, attr, NULL) ;
		RedisModule_Free (attr) ;
	}

	return true ;
}

bool RdbLoadGraphSchema_v18
(
	SerializerIO io,
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
	if (!_RdbLoadAttributeKeys(io, gc)) {
		return false;
	}

	// #Node schemas
	uint64_t schema_count ;
	TRY_READ (io, schema_count) ;

	// Load each node schema
	gc->node_schemas = array_ensure_cap (gc->node_schemas, schema_count) ;
	for (uint i = 0; i < schema_count; i ++) {
		if (!_RdbLoadSchema (io, gc, SCHEMA_NODE, already_loaded)) {
			return false;
		}
	}

	// #Edge schemas
	TRY_READ (io, schema_count) ;

	// Load each edge schema
	gc->relation_schemas =
		array_ensure_cap (gc->relation_schemas, schema_count) ;

	for (uint i = 0; i < schema_count; i ++) {
		if (!_RdbLoadSchema (io, gc, SCHEMA_EDGE, already_loaded)) {
			return false ;
		}
	}

	return true ;
}


