/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "decode_v19.h"
#include "../../../../errors/errors.h"
#include "../../../../schema/schema.h"

static void _RdbDecodeIndexField
(
	SerializerIO rdb,
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
	*name = SerializerIO_ReadCString(rdb);

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
	*phonetic = SerializerIO_ReadCString(rdb);

	// decode field dimension
	if(*type & INDEX_FLD_VECTOR) {
		*dimension = SerializerIO_ReadUnsigned(rdb);

		*M = SerializerIO_ReadUnsigned(rdb);

		*efConstruction = SerializerIO_ReadUnsigned(rdb);

		*efRuntime = SerializerIO_ReadUnsigned(rdb);

		*simFunc = SerializerIO_ReadUnsigned(rdb);
	}
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

	Index idx        = NULL ;
	char *language   = SerializerIO_ReadCString (rdb) ;
	char **stopwords = NULL ;
	
	uint stopwords_count = SerializerIO_ReadUnsigned (rdb) ;
	if (stopwords_count > 0 && !SerializerIO_Error (rdb)) {
		stopwords = arr_new (char *, 0) ;
		for (uint i = 0; i < stopwords_count; i++) {
			char *stopword = SerializerIO_ReadCString (rdb) ;
			if (SerializerIO_Error (rdb)) {
				rm_free (stopword) ;
				break ;
			}
			arr_append (stopwords, stopword) ;
		}
	}

	// the language selects RediSearch's stemmer; an unknown one crashes it
	if (!SerializerIO_Error (rdb) && RediSearch_ValidateLanguage (language)) {
		SerializerIO_SetError (rdb, "unsupported index language") ;
	}

	uint fields_count = SerializerIO_ReadUnsigned(rdb);
	for(uint i = 0; i < fields_count; i++) {
		if (SerializerIO_Error (rdb)) {
			break ;
		}

		IndexFieldType type;
		double         weight;
		bool           nostem;
		char*          phonetic;
		char*          field_name;

		// vector related options
		uint32_t       dimension = 0 ;
		size_t         M = INDEX_FIELD_DEFAULT_M ;
		size_t         efConstruction = INDEX_FIELD_DEFAULT_EF_CONSTRUCTION ;
		size_t         efRuntime = INDEX_FIELD_DEFAULT_EF_RUNTIME ;
		VecSimMetric   simFunc;

		_RdbDecodeIndexField (rdb, &field_name, &type, &weight, &nostem,
				&phonetic, &dimension, &M, &efConstruction, &efRuntime,
				&simFunc) ;

		// short read: the field is zeroed, don't add it to the index
		if (SerializerIO_Error (rdb)) {
			RedisModule_Free (phonetic) ;
			RedisModule_Free (field_name) ;
			break ;
		}

		// field type and similarity function are handed to RediSearch
		// unchecked; an unknown metric crashes vector indexing
		if (type == INDEX_FLD_UNKNOWN || (type & ~INDEX_FLD_ANY) ||
			((type & INDEX_FLD_VECTOR) &&
			 (unsigned) simFunc > VecSimMetric_Cosine)) {
			RedisModule_Free (phonetic) ;
			RedisModule_Free (field_name) ;
			SerializerIO_SetError (rdb, "invalid index field") ;
			break ;
		}

		if (!already_loaded) {
			IndexField field ;
			AttributeID field_id =
				GraphContext_FindOrAddAttribute (gc, field_name, NULL) ;

			// create new index field
			IndexField_Init (&field, field_name, field_id, type) ;

			// set field options
			IndexField_SetOptions (&field, weight, nostem, phonetic, dimension) ;

			if (type == INDEX_FLD_VECTOR) {
				IndexField_OptionsSetM (&field, M) ;
				IndexField_OptionsSetEfConstruction (&field, efConstruction) ;
				IndexField_OptionsSetEfRuntime (&field, efRuntime) ;
				IndexField_OptionsSetSimFunc (&field, simFunc) ;
			}

			// add field to index
			Schema_AddIndex (&idx, s, &field) ;
		}

		RedisModule_Free (phonetic) ;
		RedisModule_Free (field_name) ;
	}

	// idx is NULL when no fields were decoded; a partial index (short read)
	// is left unfinalized, the whole graph is torn down
	if (!already_loaded && idx != NULL && !SerializerIO_Error (rdb)) {
		Index_SetLanguage (idx, language) ;
		if (stopwords != NULL) {
			if (!Index_SetStopwords (idx, &stopwords)) {
				// not a query: don't leave the error on this thread's context
				ErrorCtx_Clear () ;
				SerializerIO_SetError (rdb, "index stopwords set twice") ;
			}
		}

		// disable and create index structure
		// must be enabled once the graph is fully loaded
		Index_Disable (idx) ;
	}
	
	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	if (stopwords != NULL) {
		for (uint i = 0; i < arr_len (stopwords); i++) {
			rm_free (stopwords [i]) ;
		}
		arr_free (stopwords) ;
	}

	RedisModule_Free (language) ;
}

static void _RdbLoadConstraint
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
	
	uint64_t n_fields = SerializerIO_ReadUnsigned(rdb);
	if (SerializerIO_Error (rdb)) {
		return ;
	}
	if (n_fields == 0 || n_fields > UINT8_MAX) {
		SerializerIO_SetError (rdb, "constraint with %llu attributes",
				(unsigned long long) n_fields) ;
		return ;
	}
	uint8_t n = n_fields;

	//--------------------------------------------------------------------------
	// decode constraint fields
	//--------------------------------------------------------------------------

	AttributeID attr_ids[n];
	const char *attr_strs[n];

	// read fields
	uint attr_count = GraphContext_AttributeCount (gc) ;
	for (uint8_t i = 0; i < n; i++) {
		uint64_t attr = SerializerIO_ReadUnsigned (rdb) ;

		// abort on a short read before building a constraint from partial
		// fields, and on an attribute the graph doesn't have
		if (SerializerIO_Error (rdb)) {
			return ;
		}
		if (attr >= attr_count) {
			SerializerIO_SetError (rdb, "constraint on an unknown attribute") ;
			return ;
		}

		attr_ids  [i] = attr ;
		attr_strs [i] = GraphContext_GetAttributeName (gc, attr) ;
	}

	if (t != CT_UNIQUE && t != CT_MANDATORY) {
		SerializerIO_SetError (rdb, "unknown constraint type") ;
		return ;
	}

	if(!already_loaded) {
		// a schema holds each constraint once
		if(Schema_ContainsConstraint(s, t, attr_ids, n)) {
			SerializerIO_SetError(rdb, "duplicate constraint");
			return;
		}

		GraphEntityType et = (Schema_GetType(s) == SCHEMA_NODE) ?
			GETYPE_NODE : GETYPE_EDGE;

		const char *err = NULL;
		c = Constraint_New((struct GraphContext*)gc, t, Schema_GetID(s),
				attr_ids, attr_strs, n, et, &err);
		if(c == NULL) {
			SerializerIO_SetError(rdb, "constraint can't be created: %s",
					err != NULL ? err : "unknown reason");
			return;
		}

		// set constraint status to active
		// only active constraints are encoded
		Constraint_SetStatus(c, CT_ACTIVE);

		// add constraint to schema
		Schema_AddConstraint(s, c);
	}
}

// load schema's constraints
static void _RdbLoadConstraints
(
	SerializerIO rdb,
	GraphContext *gc,    // graph context
	Schema *s,           // schema to populate
	bool already_loaded  // constraints already loaded
) {
	// read number of constraints
	uint constraint_count = SerializerIO_ReadUnsigned(rdb);

	for (uint i = 0; i < constraint_count; i++) {
		if(SerializerIO_Error(rdb)) {
			return;
		}
		_RdbLoadConstraint(rdb, gc, s, already_loaded);
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
	int     id   = SerializerIO_ReadUnsigned (rdb) ;
	char   *name = SerializerIO_ReadCString (rdb) ;

	// abort on a short read before building schema objects from empty data
	if (SerializerIO_Error (rdb)) {
		RedisModule_Free (name) ;
		return ;
	}

	if (!already_loaded) {
		bool created = false ;
		s = GraphContext_FindOrAddSchema (gc, name, type, &created) ;

		// schemas are encoded once each, in id order
		if (!created || Schema_GetID (s) != id) {
			RedisModule_Free (name) ;
			SerializerIO_SetError (rdb, "duplicate or out of order schema") ;
			return ;
		}
	}

	RedisModule_Free (name) ;

	//--------------------------------------------------------------------------
	// load indices
	//--------------------------------------------------------------------------

	uint index_count = SerializerIO_ReadUnsigned (rdb) ;
	for (uint index = 0; index < index_count; index++) {
		if (SerializerIO_Error (rdb)) {
			return ;
		}
		_RdbLoadIndex (rdb, gc, s, already_loaded) ;
	}

	//--------------------------------------------------------------------------
	// load constraints
	//--------------------------------------------------------------------------

	_RdbLoadConstraints (rdb, gc, s, already_loaded) ;
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
		// stop on a short read
		if(SerializerIO_Error(rdb)) {
			return;
		}
		char *attr = SerializerIO_ReadCString(rdb);
		AttributeID id = GraphContext_FindOrAddAttribute(gc, attr, NULL);
		RedisModule_Free(attr);

		// attribute keys are encoded once each, in id order; a repeated name
		// would map two encoded ids onto one attribute
		if(!SerializerIO_Error(rdb) && id != i) {
			SerializerIO_SetError(rdb, "duplicate attribute name");
			return;
		}
	}
}

void RdbLoadGraphSchema_v19
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
	_RdbLoadAttributeKeys (rdb, gc) ;

	// #Node schemas
	uint schema_count = SerializerIO_ReadUnsigned (rdb) ;

	// Load each node schema
	for (uint i = 0 ; i < schema_count ; i++) {
		if(SerializerIO_Error(rdb)) {
			return;
		}
		_RdbLoadSchema (rdb, gc, SCHEMA_NODE, already_loaded) ;
	}

	// #Edge schemas
	schema_count = SerializerIO_ReadUnsigned (rdb) ;

	// Load each edge schema
	for (uint i = 0 ; i < schema_count ; i++) {
		if(SerializerIO_Error(rdb)) {
			return;
		}
		_RdbLoadSchema (rdb, gc, SCHEMA_EDGE, already_loaded) ;
	}
}

