/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "index_field.h"
#include "../util/rmalloc.h"

//------------------------------------------------------------------------------
// index field creation
//------------------------------------------------------------------------------

static void _ResetFulltextOptions
(
	IndexField *f
) {
	ASSERT(f != NULL);

	rm_free(f->options.phonetic);

	f->options.weight   = INDEX_FIELD_DEFAULT_WEIGHT;
	f->options.nostem   = INDEX_FIELD_DEFAULT_NOSTEM;
	f->options.phonetic = rm_strdup(INDEX_FIELD_DEFAULT_PHONETIC);
}

static void _ResetVectorOptions
(
	IndexField *f
) {
	ASSERT(f != NULL);

	f->hnsw_options.dimension = 0;
}

// initialize index field
void IndexField_Init
(
	IndexField *field,   // field to initialize
	const char *name,    // field name
	AttributeID id,      // attribute ID
	IndexFieldType type  // field type
) {
	ASSERT(name     != NULL);
	ASSERT(field    != NULL);

	// clear field
	memset(field, 0, sizeof(IndexField));

	field->id   = id;
	field->name = rm_strdup(name);
	field->type = type;

	// set default options
	field->options.weight              = INDEX_FIELD_DEFAULT_WEIGHT;
	field->options.nostem              = INDEX_FIELD_DEFAULT_NOSTEM;
	field->options.phonetic            = rm_strdup(INDEX_FIELD_DEFAULT_PHONETIC);
	field->hnsw_options.dimension      = 0;
	field->hnsw_options.M              = INDEX_FIELD_DEFAULT_M;
	field->hnsw_options.efConstruction = INDEX_FIELD_DEFAULT_EF_CONSTRUCTION;
	field->hnsw_options.efRuntime      = INDEX_FIELD_DEFAULT_EF_RUNTIME;

	if(type & INDEX_FLD_FULLTEXT) {
		field->fulltext_name = field->name;
	}

	if(type & INDEX_FLD_RANGE) {
		// create all 3 fields associated with a range field

		// exact match field, e.g. n.v = 4
		// 'range:' + field name
		field->range_name = rm_malloc(strlen(name) + 7);
		sprintf(field->range_name, "range:%s", name);

		// strign multi-value field, e.g. 'x' in n.v
		// 'range:' + field name + ':string:arr'
		field->range_string_arr_name = rm_malloc(strlen(field->range_name) + 12);
		sprintf(field->range_string_arr_name, "%s:string:arr", field->range_name);

		// numeric multi-value field, e.g. 5 in n.v
		// 'range:' + field name + ':numeric:arr'
		field->range_numeric_arr_name = rm_malloc(strlen(field->range_name) + 13);
		sprintf(field->range_numeric_arr_name, "%s:numeric:arr", field->range_name);
	}

	if(type & INDEX_FLD_VECTOR) {
		field->vector_name = rm_malloc(strlen(name) + 8);
		sprintf(field->vector_name, "vector:%s", name);
	}
}

// set index field options
// note not all options are applicable to all field types
void IndexField_SetOptions
(
	IndexField *field,  // field to update
	double weight,      // field's weight
	bool nostem,        // field's stemming
	char *phonetic,     // field's phonetic
	uint32_t dimension  // field's vector dimension
) {
	ASSERT(field != NULL);
	ASSERT(phonetic != NULL);

	// default options
	ASSERT(field->options.weight    == INDEX_FIELD_DEFAULT_WEIGHT);
	ASSERT(field->options.nostem    == INDEX_FIELD_DEFAULT_NOSTEM);
	ASSERT(field->hnsw_options.dimension == 0);
	ASSERT(strcmp(field->options.phonetic, INDEX_FIELD_DEFAULT_PHONETIC) == 0);

	// set options
	field->options.weight         = weight;
	field->options.nostem         = nostem;
	field->hnsw_options.dimension = dimension;

	if(phonetic != NULL) {
		rm_free(field->options.phonetic);
		field->options.phonetic	= rm_malloc(strlen(phonetic)+1);
		strcpy(field->options.phonetic, phonetic);
	}
}

// create a new range index field
void IndexField_NewRangeField
(
	IndexField *field,   // field to initialize
	const char *name,    // field name
	AttributeID id       // field id
) {
	IndexFieldType t = INDEX_FLD_RANGE;
	IndexField_Init(field, name, id, t);
}

// create a new full text index field
void IndexField_NewFullTextField
(
	IndexField *field,   // field to initialize
	const char *name,    // field name
	AttributeID id       // field id
) {
	IndexFieldType t = INDEX_FLD_FULLTEXT;
	IndexField_Init(field, name, id, t);
}

// create a new vector index field
void IndexField_NewVectorField
(
	IndexField *field,      // field to initialize
	const char *name,       // field name
	AttributeID id,         // field id
	uint32_t dimension,     // vector dimension
	size_t M,               // max outgoing edges
	size_t efConstruction,  // construction error factor
	size_t efRuntime,       // runtime error factor
	VecSimMetric simFunc      // similarity function
) {
	IndexField_Init(field, name, id, INDEX_FLD_VECTOR);
	IndexField_OptionsSetDimension(field, dimension);
	IndexField_OptionsSetM(field, M);
	IndexField_OptionsSetEfConstruction(field, efConstruction);
	IndexField_OptionsSetEfRuntime(field, efRuntime);
	IndexField_OptionsSetSimFunc(field, simFunc);
}

// clone index field
void IndexField_Clone
(
	const IndexField *src,  // field to clone
	IndexField *dest        // cloned field
) {
	ASSERT(src  != NULL);
	ASSERT(dest != NULL);

	memcpy(dest, src, sizeof(IndexField));

	dest->name = rm_strdup(src->name);

	if(src->options.phonetic != NULL) {
		dest->options.phonetic = rm_strdup(src->options.phonetic);
	}

	//--------------------------------------------------------------------------
	// clone type specific field names
	//--------------------------------------------------------------------------

	if(src->type & INDEX_FLD_FULLTEXT) {
		dest->fulltext_name = dest->name;
	}

	if(src->type & INDEX_FLD_RANGE) {
		ASSERT(src->range_name              != NULL);
		ASSERT(src->range_string_arr_name   != NULL);
		ASSERT(src->range_numeric_arr_name  != NULL);

		dest->range_name             = rm_strdup(src->range_name);
		dest->range_string_arr_name  = rm_strdup(src->range_string_arr_name);
		dest->range_numeric_arr_name = rm_strdup(src->range_numeric_arr_name);
	}

	if(src->type & INDEX_FLD_VECTOR) {
		dest->vector_name = rm_strdup(src->vector_name);
	}
}

// return number of types in field
int IndexField_TypeCount
(
	const IndexField *f  // field
) {
	ASSERT(f != NULL);
	int count = 0;

	if(f->type & INDEX_FLD_RANGE)    count++;
	if(f->type & INDEX_FLD_VECTOR)   count++;
	if(f->type & INDEX_FLD_FULLTEXT) count++;

	return count;
}

inline IndexFieldType IndexField_GetType
(
	const IndexField *f  // field to get type
) {
	ASSERT(f != NULL);

	return f->type;
}

const char *IndexField_GetName
(
	const IndexField *f  // field to get name
) {
	ASSERT(f != NULL);	
	
	return f->name;
}

// remove type from field
void IndexField_RemoveType
(
	IndexField *f,    // field to update
	IndexFieldType t  // type to remove
) {
	ASSERT(f != NULL);
	ASSERT(t & (INDEX_FLD_RANGE | INDEX_FLD_FULLTEXT | INDEX_FLD_VECTOR));
	ASSERT(f->type & t);

	// remove RANGE type
	if(t & INDEX_FLD_RANGE) {
		// free all 3 range fields

		rm_free(f->range_name);
		f->range_name = NULL;

		rm_free(f->range_string_arr_name);
		f->range_string_arr_name = NULL;

		rm_free(f->range_numeric_arr_name);
		f->range_numeric_arr_name = NULL;
	}

	// remove FULLTEXT type
	if(t & INDEX_FLD_FULLTEXT) {
		f->fulltext_name = NULL;
		_ResetFulltextOptions(f);
	}

	// remove VECTOR type
	if(t & INDEX_FLD_VECTOR) {
		rm_free(f->vector_name);
		f->vector_name = NULL;
		_ResetVectorOptions(f);
	}

	f->type &= ~t;
}

//------------------------------------------------------------------------------
// index field options
//------------------------------------------------------------------------------

// set index field weight
void IndexField_OptionsSetWeight
(
	IndexField *field,  // field to update
	double weight       // new weight
) {
	ASSERT(field != NULL);
	field->options.weight = weight;
}

// set index field stemming
void IndexField_OptionsSetStemming
(
	IndexField *field,  // field to update
	bool nostem         // enable/disable stemming
) {
	ASSERT(field != NULL);
	field->options.nostem = nostem;
}

// set index field phonetic
void IndexField_OptionsSetPhonetic
(
	IndexField *field,    // field to update
	const char *phonetic  // phonetic
) {
	ASSERT(field    != NULL);
	ASSERT(phonetic != NULL);

	if(field->options.phonetic) rm_free(field->options.phonetic);
	field->options.phonetic = rm_strdup(phonetic);
}

// set index field vector dimension
void IndexField_OptionsSetDimension
(
	IndexField *field,  // field to update
	uint32_t dimension  // vector dimension
) {
	ASSERT(field != NULL);
	field->hnsw_options.dimension = dimension;
}

// set index field vector max outgoing edges
void IndexField_OptionsSetM
(
	IndexField *field,  // field to update
	size_t M            // max outgoing edges
) {
	ASSERT(field != NULL);
	field->hnsw_options.M = M;
}

// get index field vector max outgoing edges
size_t IndexField_OptionsGetM
(
	const IndexField *field   // field to update
) {
	ASSERT(field != NULL);
	return field->hnsw_options.M;
}

void IndexField_OptionsSetEfConstruction
(
	IndexField *field,     // field to update
	size_t efConstruction  // construction error factor
) {
	ASSERT(field != NULL);
	field->hnsw_options.efConstruction = efConstruction;
}

size_t IndexField_OptionsGetEfConstruction
(
	const IndexField *field     // field to update
) {
	ASSERT(field != NULL);
	return field->hnsw_options.efConstruction;
}

void IndexField_OptionsSetEfRuntime
(
	IndexField *field,  // field to update
	size_t efRuntime    // runtime error factor
) {
	ASSERT(field != NULL);
	field->hnsw_options.efRuntime = efRuntime;
}

size_t IndexField_OptionsGetEfRuntime
(
	const IndexField *field  // field to update
) {
	ASSERT(field != NULL);
	return field->hnsw_options.efRuntime;
}

void IndexField_OptionsSetSimFunc
(
	IndexField *field,  // field to update
	VecSimMetric func   // similarity function
) {
	ASSERT(field != NULL);
	field->hnsw_options.simFunc = func;
}

VecSimMetric IndexField_OptionsGetSimFunc
(
	const IndexField *field  // field to update
) {
	ASSERT(field != NULL);
	return field->hnsw_options.simFunc;
}

uint32_t IndexField_OptionsGetDimension
(
	const IndexField *field  // field to get dimension
) {
	ASSERT(field != NULL);
	ASSERT(field->type & INDEX_FLD_VECTOR);

	return field->hnsw_options.dimension;
}

// free index field
void IndexField_Free
(
	IndexField *f
) {
	ASSERT(f != NULL);

	rm_free(f->name);
	rm_free(f->options.phonetic);

	// free type specific field names
	if(f->range_name             != NULL) rm_free(f->range_name);
	if(f->vector_name            != NULL) rm_free(f->vector_name);
	if(f->range_string_arr_name  != NULL) rm_free(f->range_string_arr_name);
	if(f->range_numeric_arr_name != NULL) rm_free(f->range_numeric_arr_name);
}

