/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "../errors/errors.h"
#include "../index/indexer.h"
#include "../util/rmalloc.h"
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
	const char *attribute,        // attribute to index
	const SIValue options         // index options
) {
	ASSERT(label       != NULL);
	ASSERT(attribute   != NULL);
	ASSERT(entity_type != GETYPE_UNKNOWN);

	if(SI_TYPE(options) != T_MAP || !_validateOptions(options)) {
		return NULL;
	}

	// validation passed, create full-text index
	GraphContext *gc = QueryCtx_GetGraphCtx();

	double      weight   = INDEX_FIELD_DEFAULT_WEIGHT;
	bool        nostem   = INDEX_FIELD_DEFAULT_NOSTEM;
	const char *phonetic = INDEX_FIELD_DEFAULT_PHONETIC;

	//--------------------------------------------------------------------------
	// read configuration options
	//--------------------------------------------------------------------------

	SIValue tmp;
	if(MAP_GET(options, "weight", tmp))   weight   = SI_GET_NUMERIC(tmp);
	if(MAP_GET(options, "nostem", tmp))   nostem   = tmp.longval;
	if(MAP_GET(options, "phonetic", tmp)) phonetic = tmp.stringval;

	//--------------------------------------------------------------------------
	// index stopwords
	//--------------------------------------------------------------------------

	SIValue sw;
	char **stopwords = NULL;

	if(MAP_GET(options, "stopwords", sw)) {
		uint stopwords_count = SIArray_Length(sw);
		stopwords = array_new(char*, stopwords_count); // freed by the index
		for (uint i = 0; i < stopwords_count; i++) {
			SIValue stopword = SIArray_Get(sw, i);
			array_append(stopwords, rm_strdup(stopword.stringval));
		}
	}

	//--------------------------------------------------------------------------
	// index language
	//--------------------------------------------------------------------------

	SIValue lang;  
	const char *language = NULL;

	if(MAP_GET(options, "language", lang)) {
		language = lang.stringval;
	}

	// make sure index configuration isn't overridden
	Index idx = NULL;
	Attribute_ID attr = GraphContext_GetAttributeID(gc, label);
	if(attr != ATTRIBUTE_ID_NONE) {
		// make sure field isn't already indexed
		idx = GraphContext_GetIndex(gc, label, &attr, 1, INDEX_FLD_FULLTEXT,
				SCHEMA_NODE);
		if(idx != NULL) {
			ErrorCtx_SetError(EMSG_INDEX_ALREADY_EXISTS);
			goto cleanup;
		}

		// make sure index level configuration isn't already set
		SchemaType st = (entity_type == GETYPE_NODE)? SCHEMA_NODE : SCHEMA_EDGE;
		idx = GraphContext_GetIndex(gc, label, &attr, 1, INDEX_FLD_ANY, st);

		if( idx != NULL                                        &&
		   (stopwords != NULL && Index_ContainsStopwords(idx)) || 
		   (language  != NULL && Index_GetLanguage(idx))) {
			ErrorCtx_SetError(EMSG_INDEX_ALREADY_EXISTS);
			goto cleanup;
		}
	}

	//--------------------------------------------------------------------------
	// try to build index
	//--------------------------------------------------------------------------

	bool res = GraphContext_AddFullTextIndex(&idx, gc, label, &attribute, 1,
			&weight, &nostem, &phonetic);
	ASSERT(res == true);

	// set stopwords
	if(stopwords != NULL) Index_SetStopwords(idx, &stopwords);

	// set language
	if(language != NULL) Index_SetLanguage(idx, language);

	return idx;

cleanup:
	if(stopwords != NULL) {
		array_free_cb(stopwords, rm_free);
	}

	return NULL;
}

