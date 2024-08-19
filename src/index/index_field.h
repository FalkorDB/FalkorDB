/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../graph/entities/attribute_set.h"

#define INDEX_FIELD_NONE_INDEXED "NONE_INDEXABLE_FIELDS"
#define INDEX_FIELD_DEFAULT_WEIGHT 1.0
#define INDEX_FIELD_DEFAULT_NOSTEM false
#define INDEX_FIELD_DEFAULT_PHONETIC "no"
#define INDEX_FIELD_DEFAULT_M 16
#define INDEX_FIELD_DEFAULT_EF_CONSTRUCTION 200
#define INDEX_FIELD_DEFAULT_EF_RUNTIME 10

// type of index field
// multiple types can be combined via bitwise OR
typedef enum {
	INDEX_FLD_UNKNOWN  = 0x00,  // unknown field type
	INDEX_FLD_FULLTEXT = 0x01,  // full text field
	INDEX_FLD_NUMERIC  = 0x02,  // numeric field
	INDEX_FLD_GEO      = 0x04,  // geo field
	INDEX_FLD_STR      = 0x08,  // string field
	INDEX_FLD_VECTOR   = 0x10,  // vector field
} IndexFieldType;

#define INDEX_FLD_RANGE (INDEX_FLD_NUMERIC | INDEX_FLD_GEO | INDEX_FLD_STR)
#define INDEX_FLD_ANY (INDEX_FLD_FULLTEXT | INDEX_FLD_RANGE | INDEX_FLD_VECTOR)

typedef struct {
	char *name;              // field name
	AttributeID id;          // field id
	IndexFieldType type;     // field type(s)
	struct {
		double weight;          // the importance of text
		bool nostem;            // disable stemming of the text
		char *phonetic;         // phonetic search of text
	} options;
	struct {
		uint32_t dimension;     // vector dimension
		size_t M;               // max outgoing edges
		size_t efConstruction;  // construction parameter for HNSW
		size_t efRuntime;       // runtime parameter for HNSW
	} hnsw_options;
	char *range_name;        // 'range:'  + field name
	char *vector_name;       // 'vector:' + field name
	char *fulltext_name;     // field name
} IndexField;

//------------------------------------------------------------------------------
// index field creation
//------------------------------------------------------------------------------

// initialize index field
void IndexField_Init
(
	IndexField *field,   // field to initialize
	const char *name,    // field name
	AttributeID id,      // attribute ID
	IndexFieldType type  // field type
);

// clone index field
void IndexField_Clone
(
	const IndexField *src,  // field to clone
	IndexField *dest        // cloned field
);

// create a new range index field
void IndexField_NewRangeField
(
	IndexField *field,   // field to initialize
	const char *name,    // field name
	AttributeID id       // field id
);

// create a new full text index field
void IndexField_NewFullTextField
(
	IndexField *field,   // field to initialize
	const char *name,    // field name
	AttributeID id       // field id
);

// create a new vector index field
void IndexField_NewVectorField
(
	IndexField *field,      // field to initialize
	const char *name,       // field name
	AttributeID id,         // field id
	uint32_t dimension,     // vector dimension
	size_t M,               // max outgoing edges
	size_t efConstruction,  // construction error factor
	size_t efRuntime        // runtime error factor
);

// return number of types in field
int IndexField_TypeCount
(
	const IndexField *f  // field
);

IndexFieldType IndexField_GetType
(
	const IndexField *f  // field to get type
);

const char *IndexField_GetName
(
	const IndexField *f  // field to get name
);

// remove type from field
void IndexField_RemoveType
(
	IndexField *f,    // field to update
	IndexFieldType t  // type to remove
);

//------------------------------------------------------------------------------
// index field options
//------------------------------------------------------------------------------

// set index field options
// note not all options are applicable to all field types
void IndexField_SetOptions
(
	IndexField *field,  // field to update
	double weight,      // field's weight
	bool nostem,        // field's stemming
	char *phonetic,     // field's phonetic
	uint32_t dimension  // field's vector dimension
);

// set index field weight
void IndexField_OptionsSetWeight
(
	IndexField *field,  // field to update
	double weight       // new weight
);

// set index field stemming
void IndexField_OptionsSetStemming
(
	IndexField *field,  // field to update
	bool nostem         // enable/disable stemming
);

// set index field phonetic
void IndexField_OptionsSetPhonetic
(
	IndexField *field,    // field to update
	const char *phonetic  // phonetic
);

// set index field vector dimension
void IndexField_OptionsSetDimension
(
	IndexField *field,  // field to update
	uint32_t dimension  // vector dimension
);

// get vector index field dimension
// field must be a vector field
uint32_t IndexField_OptionsGetDimension
(
	const IndexField *field  // field to get dimension
);

// set index field vector max outgoing edges
void IndexField_OptionsSetM
(
	IndexField *field,  // field to update
	size_t M            // max outgoing edges
);

// get index field vector max outgoing edges
size_t IndexField_OptionsGetM
(
	const IndexField *field   // field to update
);

void IndexField_OptionsSetEfConstruction
(
	IndexField *field,    // field to update
	size_t efConstruction // construction error factor
);

size_t IndexField_OptionsGetEfConstruction
(
	const IndexField *field     // field to update
);

void IndexField_OptionsSetEfRuntime
(
	IndexField *field,  // field to update
	size_t efRuntime    // runtime error factor
);

size_t IndexField_OptionsGetEfRuntime
(
	const IndexField *field  // field to update
);

// free index field
void IndexField_Free
(
	IndexField *field  // index field to be freed
);

