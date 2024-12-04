/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "encode_v16.h"
#include "../../../util/arr.h"

static void _RdbSaveAttributeKeys
(
	SerializerIO rdb,
	GraphContext *gc
) {
	// Format:
	// #attribute keys
	// attribute keys

	uint count = GraphContext_AttributeCount(gc);
	SerializerIO_WriteUnsigned(rdb, count);
	for(uint i = 0; i < count; i ++) {
		char *key = gc->string_mapping[i];
		SerializerIO_WriteBuffer(rdb, key, strlen(key) + 1);
	}
}

// encode index field
static void _RdbSaveIndexField
(
	SerializerIO rdb,    // io
	const IndexField *f  // index field to encode
) {
	// format:
	// name
	// type
	// options:
	//   weight
	//   nostem
	//   phonetic
	//   dimension

	ASSERT(f != NULL);

	// encode field name
	SerializerIO_WriteBuffer(rdb, f->name, strlen(f->name) + 1);

	// encode field type
	SerializerIO_WriteUnsigned(rdb, f->type);

	//--------------------------------------------------------------------------
	// encode field options
	//--------------------------------------------------------------------------

	// encode field weight
	SerializerIO_WriteDouble(rdb, f->options.weight);

	// encode field nostem
	SerializerIO_WriteUnsigned(rdb, f->options.nostem);

	// encode field phonetic
	SerializerIO_WriteBuffer(rdb, f->options.phonetic,
			strlen(f->options.phonetic) + 1);

	// encode field dimension
	if(IndexField_GetType(f) & INDEX_FLD_VECTOR) {
		SerializerIO_WriteUnsigned(rdb, IndexField_OptionsGetDimension(f));
		SerializerIO_WriteUnsigned(rdb, IndexField_OptionsGetM(f));
		SerializerIO_WriteUnsigned(rdb, IndexField_OptionsGetEfConstruction(f));
		SerializerIO_WriteUnsigned(rdb, IndexField_OptionsGetEfRuntime(f));
		SerializerIO_WriteUnsigned(rdb, IndexField_OptionsGetSimFunc(f));
	}
}

static inline void _RdbSaveIndexData
(
	SerializerIO rdb,
	SchemaType type,
	Index idx
) {
	if(idx == NULL) return;

	// Format:
	// language
	// #stopwords - N
	// N * stopword
	// #properties - M
	// M * property {options}

	// encode language
	const char *language = Index_GetLanguage(idx);
	SerializerIO_WriteBuffer(rdb, language, strlen(language) + 1);

	size_t stopwords_count;
	char **stopwords = Index_GetStopwords(idx, &stopwords_count);
	// encode stopwords count
	SerializerIO_WriteUnsigned(rdb, stopwords_count);
	for (size_t i = 0; i < stopwords_count; i++) {
		char *stopword = stopwords[i];
		SerializerIO_WriteBuffer(rdb, stopword, strlen(stopword) + 1);
		rm_free(stopword);
	}
	rm_free(stopwords);

	// encode field count
	uint fields_count = Index_FieldsCount(idx);
	SerializerIO_WriteUnsigned(rdb, fields_count);

	// encode fields
	const IndexField *fields = Index_GetFields(idx);
	for(uint i = 0; i < fields_count; i++) {
		_RdbSaveIndexField(rdb, fields + i);
	}
}

static void _RdbSaveConstraint
(
	SerializerIO rdb,
	const Constraint c
) {
	// Format:
	// constraint type
	// fields count
	// field IDs

	// only encode active constraint
	ASSERT(Constraint_GetStatus(c) == CT_ACTIVE);

	//--------------------------------------------------------------------------
	// encode constraint type
	//--------------------------------------------------------------------------

	ConstraintType t = Constraint_GetType(c);
	SerializerIO_WriteUnsigned(rdb, t);

	//--------------------------------------------------------------------------
	// encode constraint fields count
	//--------------------------------------------------------------------------

	const AttributeID *attrs;
	uint8_t n = Constraint_GetAttributes(c, &attrs, NULL);
	SerializerIO_WriteUnsigned(rdb, n);

	//--------------------------------------------------------------------------
	// encode constraint fields
	//--------------------------------------------------------------------------

	for(uint8_t i = 0; i < n; i++) {
		AttributeID attr = attrs[i];
		SerializerIO_WriteUnsigned(rdb, attr);
	}
}

static void _RdbSaveConstraintsData
(
	SerializerIO rdb,
	Constraint *constraints
) {
	uint n_constraints = array_len(constraints);
	Constraint *active_constraints = array_new(Constraint, n_constraints);

	// collect active constraints
	for (uint i = 0; i < n_constraints; i++) {
		Constraint c = constraints[i];
		if (Constraint_GetStatus(c) == CT_ACTIVE) {
			array_append(active_constraints, c);
		}
	}

	// encode number of active constraints
	uint n_active_constraints = array_len(active_constraints);
	SerializerIO_WriteUnsigned(rdb, n_active_constraints);

	// encode constraints
	for (uint i = 0; i < n_active_constraints; i++) {
		Constraint c = active_constraints[i];
		_RdbSaveConstraint(rdb, c);
	}

	// clean up
	array_free(active_constraints);
}

static void _RdbSaveSchema
(
	SerializerIO rdb,
	Schema *s
) {
	// Format:
	// id
	// name
	// #indices
	// (indexed property) X M 
	// #constraints 
	// (constraint type, constraint fields) X N

	// Schema ID.
	SerializerIO_WriteUnsigned(rdb, s->id);

	// Schema name.
	SerializerIO_WriteBuffer(rdb, s->name, strlen(s->name) + 1);

	// Number of indices.
	SerializerIO_WriteUnsigned(rdb, Schema_HasIndices(s));

	// index, prefer pending over active
	Index idx = PENDING_IDX(s)
		? PENDING_IDX(s)
		: ACTIVE_IDX(s);

	_RdbSaveIndexData(rdb, s->type, idx);

	// constraints
	_RdbSaveConstraintsData(rdb, s->constraints);
}

void RdbSaveGraphSchema_v16
(
	SerializerIO rdb,
	GraphContext *gc
) {
	// Format:
	// attribute keys (unified schema)
	// #node schemas
	// node schema X #node schemas
	// #relation schemas
	// relation schema X #relation schemas

	// Serialize all attribute keys
	_RdbSaveAttributeKeys(rdb, gc);

	// #Node schemas.
	unsigned short schema_count = GraphContext_SchemaCount(gc, SCHEMA_NODE);
	SerializerIO_WriteUnsigned(rdb, schema_count);

	// Name of label X #node schemas.
	for(int i = 0; i < schema_count; i++) {
		Schema *s = gc->node_schemas[i];
		_RdbSaveSchema(rdb, s);
	}

	// #Relation schemas.
	unsigned short relation_count = GraphContext_SchemaCount(gc, SCHEMA_EDGE);
	SerializerIO_WriteUnsigned(rdb, relation_count);

	// Name of label X #relation schemas.
	for(unsigned short i = 0; i < relation_count; i++) {
		Schema *s = gc->relation_schemas[i];
		_RdbSaveSchema(rdb, s);
	}
}

