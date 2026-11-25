/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include <string.h>

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

	// THE MAP'S OWN PAIRS, each walked once.
	//
	// This used to run the whole MAP_GET ladder n times over, which validates
	// whichever key sits highest in the ladder n times and never reaches the
	// rest. With a weight present, "weight" answered on every iteration and
	// continued, so phonetic, nostem, language and stopwords went unchecked
	// while 'matched' still counted down to zero:
	// OPTIONS {weight: 1, phonetic: 123} was accepted, and Index_FulltextCreate
	// below then read that integer's union as 'stringval'.
	//
	// It was invisible while the fulltext procedure put a weight into the map
	// for EVERY field whether the statement named one or not - the ladder's
	// first rung always answered, so no later rung was reachable to be wrong.
	// A map carrying only what the statement stated is what reaches here now.
	uint known = 0;
	for(uint i = 0; i < n; i++) {
		SIValue key;
		SIValue val;
		Map_GetIdx(options, i, &key, &val);

		const char *k = key.stringval;
		SIType      t = SI_TYPE(val);

		if(strcmp(k, "weight") == 0) {
			if(!(t & SI_NUMERIC)) return false;
		} else if(strcmp(k, "phonetic") == 0) {
			if(!(t & T_STRING)) return false;
		} else if(strcmp(k, "nostem") == 0) {
			if(!(t & T_BOOL)) return false;
		} else if(strcmp(k, "language") == 0) {
			if(!(t & T_STRING)) return false;
		} else if(strcmp(k, "stopwords") == 0) {
			if(!(t & T_ARRAY)) return false;
		} else {
			continue;    // not an option this index knows
		}

		known++;
	}

	// ACCEPTANCE IS LEFT EXACTLY WHERE IT WAS, which is not where it would be
	// put from scratch: a map is valid if it names at least one option this
	// index knows, and an unrecognised key alongside it is ignored rather than
	// refused. {weight: 1, typo: 2} creates the index; {typo: 2} alone does
	// not.
	//
	// Refusing the stray key is the better rule and was tried here. It cannot
	// land as part of a type-checking fix, because the v2 effect carries the
	// options map WHOLE (writers/v2/write_ddl.c:49) and the replica hands the
	// decoded map straight back to GraphHub_AddIndex
	// (create_index_effect.c:126). A primary running an older build accepts
	// the stray key, creates the index and replicates that map; a replica that
	// refused it would log a warning and skip the index, and the two would
	// diverge with nothing raised to anyone. An AOF written by an older build
	// replays the same way.
	//
	// The type checks above have no such window: an ill-typed option did not
	// work before this, it crashed the server on the primary, so no index was
	// created and no effect was ever written for a replica to refuse.
	return (n == 0 || known > 0);
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

