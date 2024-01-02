/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "../errors/errors.h"
#include "../graph/graphcontext.h"
#include "../datatypes/datatypes.h"

// validate index configuration options map
// returns true if valid, false otherwise
//
// {
//     weight: 2.0,
//     phonetic: 'dm:en',
//     nostem: true,
//     language: 'english',
//     stopwords: ['the', 'a', 'an']
// }
static bool _validateOptions
(
	SIValue options
) {
	uint n = Map_KeyCount(options);
	if(n > 5) {
		ErrorCtx_SetError("invlaid index configuration");
		return false;
	}

	SIValue tmp;
	uint matched = n;
	for(uint i = 0; i < n; i++) {
		//----------------------------------------------------------------------
		// validate weight
		//----------------------------------------------------------------------

		if(MAP_GET(options, "weight", tmp)) {
			matched--;
			if(!(SI_TYPE(tmp) & SI_NUMERIC)) {
				return false;
			}
			continue;
		}

		//----------------------------------------------------------------------
		// validate phonetic
		//----------------------------------------------------------------------

		if(MAP_GET(options, "phonetic", tmp)) {
			matched--;
			if(!(SI_TYPE(tmp) != T_STRING)) {
				return false;
			}
			continue;
		}

		//----------------------------------------------------------------------
		// validate nostem
		//----------------------------------------------------------------------

		if(MAP_GET(options, "nostem", tmp)) {
			matched--;
			if(!(SI_TYPE(tmp) & T_BOOL)) {
				return false;
			}
			continue;
		}

		//----------------------------------------------------------------------
		// validate language
		//----------------------------------------------------------------------

		if(MAP_GET(options, "language", tmp)) {
			matched--;
			if(!(SI_TYPE(tmp) & T_STRING)) {
				return false;
			}
			continue;
		}

		//----------------------------------------------------------------------
		// validate stopwords
		//----------------------------------------------------------------------

		if(MAP_GET(options, "stopwords", tmp)) {
			matched--;
			if(!(SI_TYPE(tmp) & T_ARRAY)) {
				return false;
			}
			continue;
		}
	}

	// make sure all specified options are valid
	return (matched == 0);
}

// create fulltext index
//
// CALL db.idx.fulltext.createNodeIndex('book', 'title', 'authors')
// CALL db.idx.fulltext.createNodeIndex({label:'L', stopwords:['The']}, 'v')
// CALL db.idx.fulltext.createNodeIndex('L', {field:'v', weight:2.1})
// CREATE FULLTEXT INDEX FOR (n:Person) ON (n.name) OPTIONS {
//     weight: 2.0,
//     phonetic: 'dm:en',
//     nostem: true,
//     language: 'english',
//     stopwords: ['the', 'a', 'an']
// }
Index Index_FulltextCreate
(
	const char *label,            // label/relationship type
	GraphEntityType entity_type,  // entity type (node/edge)
	const char *attr,             // attribute to index
	AttributeID attr_id,          // attribute id
	const SIValue options         // index options
) {
	ASSERT(label       != NULL);
	ASSERT(attr        != NULL);
	ASSERT(entity_type != GETYPE_UNKNOWN);

	if(SI_TYPE(options) != T_MAP || !_validateOptions(options)) {
		ErrorCtx_SetError("invlaid index configuration");
		return NULL;
	}

	// validation passed, create full-text index
	double      weight   = INDEX_FIELD_DEFAULT_WEIGHT;
	bool        nostem   = INDEX_FIELD_DEFAULT_NOSTEM;
	const char *phonetic = INDEX_FIELD_DEFAULT_PHONETIC;

	//--------------------------------------------------------------------------
	// read configuration options
	//--------------------------------------------------------------------------

	SIValue tmp;
	if(MAP_GET(options, "weight",   tmp)) weight   = SI_GET_NUMERIC(tmp);
	if(MAP_GET(options, "nostem",   tmp)) nostem   = tmp.longval;
	if(MAP_GET(options, "phonetic", tmp)) phonetic = tmp.stringval;

	//--------------------------------------------------------------------------
	// try to build index
	//--------------------------------------------------------------------------

	IndexField field;

	IndexField_NewFullTextField(&field, attr, attr_id);
	IndexField_OptionsSetWeight(&field,   weight);
	IndexField_OptionsSetStemming(&field, nostem);
	IndexField_OptionsSetPhonetic(&field, phonetic);

	// get schema
	GraphContext *gc = QueryCtx_GetGraphCtx();
	SchemaType st = (entity_type == GETYPE_NODE) ?SCHEMA_NODE : SCHEMA_EDGE;
	Schema *s = GraphContext_GetSchema(gc, label, st);
	ASSERT(s != NULL);

	Index idx = NULL;
	Schema_AddIndex(&idx, s, &field);

	return idx;
}

