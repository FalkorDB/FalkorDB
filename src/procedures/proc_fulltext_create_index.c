/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */


#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "../errors/errors.h"
#include "../index/indexer.h"
#include "../util/rmalloc.h"
#include "../graph/graph_hub.h"
#include "../graph/graphcontext.h"
#include "../datatypes/datatypes.h"
#include "../util/identifier_limits.h"
#include "proc_fulltext_create_index.h"

//------------------------------------------------------------------------------
// fulltext createNodeIndex
//------------------------------------------------------------------------------

// validate index configuration map
// [required] label <string>
// [optional] stopwords <string[]>
// [optional] language <string>
// configuration can't change if index exists 
static ProcedureResult _validateIndexConfigMap
(
	SIValue config
) {
	SIValue sw;
	SIValue lang;
	SIValue label;

	bool multi_config    = Map_KeyCount(config) > 1;
	bool label_exists    = MAP_GET(config, "label",     label);
	bool lang_exists     = MAP_GET(config, "language",  lang);
	bool stopword_exists = MAP_GET(config, "stopwords", sw);

	if(!label_exists) {
		ErrorCtx_SetError(EMSG_IS_MISSING, "Label");
		return PROCEDURE_ERR;
	}

	//--------------------------------------------------------------------------
	// validate stopwords
	//--------------------------------------------------------------------------

	if(stopword_exists) {
		if(SI_TYPE(sw) == T_ARRAY) {
			if(!SIArray_AllOfType(sw, T_STRING)) {
				ErrorCtx_SetError(EMSG_MUST_BE, "Stopword", "string");
				return PROCEDURE_ERR;
			}
		} else {
			ErrorCtx_SetError(EMSG_MUST_BE, "Stopwords", "array");
			return PROCEDURE_ERR;
		}
	}

	//--------------------------------------------------------------------------
	// validate language
	//--------------------------------------------------------------------------

	if(lang_exists) {
		if(!(SI_TYPE(lang) & T_STRING)) {
			ErrorCtx_SetError(EMSG_MUST_BE, "Language", "string");
			return PROCEDURE_ERR;
		}
		if(RediSearch_ValidateLanguage(lang.stringval)) {
			ErrorCtx_SetError(EMSG_NOT_SUPPORTED, "Language");
			return PROCEDURE_ERR;
		}
	}

	return PROCEDURE_OK;
}

// validate field configuration map
// [required] field <string>
// [optional] weight <number>
// [optional] phonetic <string>
// [optional] nostem <bool>
// configuration can't change if index exists 
static ProcedureResult _validateFieldConfigMap
(
	SIValue config
) {
	SIValue field;
	SIValue weight;
	SIValue nostem;
	SIValue phonetic;

	bool  multi_config    = Map_KeyCount(config) > 1;
	bool  field_exists    = MAP_GET(config, "field",    field);
	bool  weight_exists   = MAP_GET(config, "weight",   weight);
	bool  nostem_exists   = MAP_GET(config, "nostem",   nostem);
	bool  phonetic_exists = MAP_GET(config, "phonetic", phonetic);

	// field name is mandatory
	if(!field_exists) {
		ErrorCtx_SetError(EMSG_IS_MISSING, "Field");
		return PROCEDURE_ERR;
	}

	if(!(SI_TYPE(field) & T_STRING)) {
		ErrorCtx_SetError(EMSG_MUST_BE, "Field", "string");
		return PROCEDURE_ERR;
	}

	if(weight_exists) {
		if((SI_TYPE(weight) & SI_NUMERIC) == 0) {
			ErrorCtx_SetError(EMSG_MUST_BE, "Weight", "numeric");
			return PROCEDURE_ERR;
		}
	}

	if(nostem_exists) {
		if(SI_TYPE(nostem) != T_BOOL) {
			ErrorCtx_SetError(EMSG_MUST_BE, "Nostem", "bool");
			return PROCEDURE_ERR;
		}
	}

	if(phonetic_exists) {
		if(!(SI_TYPE(phonetic) & T_STRING)) {
			ErrorCtx_SetError(EMSG_MUST_BE, "Phonetic", "string");
			return PROCEDURE_ERR;
		}
	}

	return PROCEDURE_OK;
}

// extract index level configuration from options map
static void extract_index_level_config
(
	char ***stopwords,  // index stopwods
	char **language,    // index language
	SIValue options     // options map
) {
	ASSERT(language  != NULL);
	ASSERT(stopwords != NULL);

	// set default values
	*language  = NULL;
	*stopwords = NULL;

	// quick return if options is not a map
	if(SI_TYPE(options) != T_MAP) return;

	//--------------------------------------------------------------------------
	// extract language
	//--------------------------------------------------------------------------

	SIValue language_val;
	if(MAP_GET(options, "language", language_val)) {
		ASSERT(SI_TYPE(language_val) == T_STRING);
		*language = language_val.stringval;
	}

	//--------------------------------------------------------------------------
	// extract stopwords
	//--------------------------------------------------------------------------

	SIValue stopwords_val;
	if(MAP_GET(options, "stopwords", stopwords_val)) {
		// validate stopwords is an array of strings
		ASSERT(SI_TYPE(stopwords_val) == T_ARRAY &&
			   SIArray_AllOfType(stopwords_val, T_STRING));

		uint nstopwords = SIArray_Length(stopwords_val);
		*stopwords = arr_new(char*, nstopwords);
		for(uint i = 0; i < nstopwords; i++) {
			SIValue stopword = SIArray_Get(stopwords_val, i);
			arr_append((*stopwords), rm_strdup(stopword.stringval));
		}
	}
}

// CALL db.idx.fulltext.createNodeIndex(label, fields...)
// CALL db.idx.fulltext.createNodeIndex('book', 'title', 'authors')
// CALL db.idx.fulltext.createNodeIndex({label:'L', stopwords:['The']}, 'v')
// CALL db.idx.fulltext.createNodeIndex('L', {field:'v', weight:2.1})
ProcedureResult Proc_FulltextCreateNodeIdxInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	bool res = true;
	uint arg_count = arr_len((SIValue *)args);
	if(arg_count < 2) {
		ErrorCtx_SetError(EMSG_FULLTEXT_MIN_ARGS);
		return PROCEDURE_ERR;
	}

	// label argument should be of type string or map
	if(!(SI_TYPE(args[0]) & (T_STRING | T_MAP))) {
		ErrorCtx_SetError(EMSG_FULLTEXT_LABEL_TYPE);
		return PROCEDURE_ERR;
	}

	if(SI_TYPE(args[0]) == T_MAP &&
			_validateIndexConfigMap(args[0]) == PROCEDURE_ERR) {
		return PROCEDURE_ERR;
	}

	char *label = NULL;
	SIValue label_config = args[0];

	// validation, fields arguments should be of type string or map
	for(uint i = 1; i < arg_count; i++) {
		if(!(SI_TYPE(args[i]) & (T_STRING | T_MAP))) {
			ErrorCtx_SetError(EMSG_FULLTEXT_FIELD_TYPE);
			return PROCEDURE_ERR;
		}
		if(SI_TYPE(args[i]) == T_MAP &&
			_validateFieldConfigMap(args[i]) == PROCEDURE_ERR) {
			return PROCEDURE_ERR;
		}
	}

	// extract index label
	if(SI_TYPE(label_config) & T_STRING) {
		label = label_config.stringval;
	} else if(SI_TYPE(label_config) == T_MAP) {
		SIValue label_value;
		MAP_GET(label_config, "label", label_value);
		label = label_value.stringval;
	}

	// label is mandatory
	ASSERT(label != NULL);

	if (strnlen (label, MAX_IDENTIFIER_LEN + 1) > MAX_IDENTIFIER_LEN) {
		ErrorCtx_SetError (EMSG_IDENTIFIER_TOO_LONG, "Label name",
				MAX_IDENTIFIER_LEN) ;
		return PROCEDURE_ERR ;
	}

	// validation passed, create full-text index
	Index idx             = NULL;
	char *language        = NULL;
	char **stopwords      = NULL;
	GraphContext *gc      = QueryCtx_GetGraphCtx();
	uint fields_count     = arg_count - 1; // skip label
	const SIValue *fields = args + 1;      // skip index name

	const char* _fields[fields_count];
	bool        nostems[fields_count];
	double      weights[fields_count];
	const char* phonetics[fields_count];

	// WHICH of the three the statement actually STATED, as opposed to which
	// ones ended up with a value - every field ends up with all three.
	//
	// The distinction is invisible locally, because an option left out gets
	// the same default here that Index_FulltextCreate would apply anyway. It
	// is not invisible on the wire: an effect carries a presence flag per
	// option meaning "the statement said this", and an effect MUTATES an index
	// that may already exist. Announcing a default as though it had been
	// stated has already diverged a live replica once with language, and
	// phonetic is worse - C's default is the literal string "no" while Rust
	// reads any non-empty phonetic as ENABLED, so an unstated phonetic sent as
	// its default turns itself ON when it crosses engines.
	bool weight_stated  [fields_count];
	bool nostem_stated  [fields_count];
	bool phonetic_stated[fields_count];

	// collect fields and configuration
	for(uint i = 0; i < fields_count; i++) {
		weights  [i] = INDEX_FIELD_DEFAULT_WEIGHT ;
		nostems  [i] = INDEX_FIELD_DEFAULT_NOSTEM ;
		phonetics[i] = INDEX_FIELD_DEFAULT_PHONETIC ;

		weight_stated  [i] = false ;
		nostem_stated  [i] = false ;
		phonetic_stated[i] = false ;

		if(SI_TYPE(fields[i]) & T_STRING) {
			_fields[i] = fields[i].stringval;
		} else {
			SIValue tmp;
			MAP_GET(fields[i], "field", tmp);
			_fields[i] = tmp.stringval;

			if(MAP_GET(fields[i], "weight", tmp)) {
				weights[i]       = SI_GET_NUMERIC(tmp);
				weight_stated[i] = true;
			}
			if(MAP_GET(fields[i], "nostem", tmp)) {
				nostems[i]       = tmp.longval;
				nostem_stated[i] = true;
			}
			if(MAP_GET(fields[i], "phonetic", tmp)) {
				phonetics[i]       = tmp.stringval;
				phonetic_stated[i] = true;
			}
		}

		if (strnlen (_fields [i], MAX_IDENTIFIER_LEN + 1) > MAX_IDENTIFIER_LEN) {
			ErrorCtx_SetError (EMSG_IDENTIFIER_TOO_LONG, "Property name",
					MAX_IDENTIFIER_LEN) ;
			return PROCEDURE_ERR ;
		}
	}

	//--------------------------------------------------------------------------
	// create index one field at a time
	//--------------------------------------------------------------------------

	ResultSet *result_set = QueryCtx_GetResultSet();
	ASSERT(result_set != NULL);

	// extract index-level configuration (language / stopwords) up front and
	// fold it into the per-field options map below, so it's embedded in
	// every field's create-index effect - mirroring the CREATE FULLTEXT
	// INDEX ... OPTIONS {} syntax (index_operations.c). a replica applies
	// this via the effect alone (ApplyCreateIndex), it never re-executes
	// this procedure, so language/stopwords living only in 'label_config'
	// (as opposed to 'options') would never reach it.
	extract_index_level_config(&stopwords, &language, label_config);

	SIValue options = SI_Map(3);
	if(language != NULL) {
		Map_Add(&options, SI_ConstStringVal("language"),
				SI_ConstStringVal(language));
	}
	if(stopwords != NULL) {
		SIValue sw = SIArray_New(arr_len(stopwords));
		for(uint i = 0; i < arr_len(stopwords); i++) {
			SIArray_Append(&sw, SI_ConstStringVal(stopwords[i]));
		}
		Map_Add(&options, SI_ConstStringVal("stopwords"), sw);
		SIArray_Free(sw);
	}

	for(uint i = 0; i < fields_count; i++) {
		// construct options map
		//
		// ONLY WHAT THE STATEMENT STATED. The map is reused across fields, so
		// an option this field did not state has to be REMOVED rather than
		// merely not added - otherwise the previous field's value is still in
		// the map and this field silently inherits it.
		//
		// Index_FulltextCreate seeds all three from INDEX_FIELD_DEFAULT_* and
		// overrides only on a hit, so leaving a key out builds exactly the
		// same field it built before. What changes is the EFFECT: an option
		// the user never mentioned no longer travels as though they had.
		if(weight_stated[i]) {
			Map_Add(&options, SI_ConstStringVal("weight"),
					SI_DoubleVal(weights[i]));
		} else {
			Map_Remove(options, SI_ConstStringVal("weight"));
		}

		if(phonetic_stated[i]) {
			Map_Add(&options, SI_ConstStringVal("phonetic"),
					SI_ConstStringVal(phonetics[i]));
		} else {
			Map_Remove(options, SI_ConstStringVal("phonetic"));
		}

		if(nostem_stated[i]) {
			Map_Add(&options, SI_ConstStringVal("nostem"),
					SI_BoolVal(nostems[i]));
		} else {
			Map_Remove(options, SI_ConstStringVal("nostem"));
		}

		idx = GraphHub_AddIndex(gc, label, _fields[i], GETYPE_NODE,
				INDEX_FLD_FULLTEXT, options, true);
		if(idx != NULL) {
			ResultSet_IndexCreated(result_set, INDEX_OK);
		} else {
			// operation failed
			res = false;
			goto cleanup;
		}
	}

	// index created, populate
	if(idx != NULL) {
		//----------------------------------------------------------------------
		// set index level configuration
		//----------------------------------------------------------------------

		if(language != NULL && !Index_SetLanguage(idx, language)) {
			res = false;
			goto cleanup;
		}

		if(stopwords != NULL && !Index_SetStopwords(idx, &stopwords)) {
			res = false;
			goto cleanup;
		}

		Index_Disable(idx);

		// populate index
		Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
		ASSERT(s != NULL);
		Indexer_PopulateIndex(gc, s, idx);
	}

cleanup:
	if(stopwords != NULL) arr_free_cb(stopwords, rm_free);
	Map_Free(options);

	return (res) ? PROCEDURE_OK : PROCEDURE_ERR;
}

SIValue *Proc_FulltextCreateNodeIdxStep
(
	ProcedureCtx *ctx
) {
	return NULL;
}

ProcedureCtx *Proc_FulltextCreateNodeIdxGen() {
	ProcedureOutput *output = arr_new(ProcedureOutput, 0);
	return ProcCtxNew("db.idx.fulltext.createNodeIndex",
			PROCEDURE_VARIABLE_ARG_COUNT, output,
			Proc_FulltextCreateNodeIdxStep, Proc_FulltextCreateNodeIdxInvoke,
			NULL, NULL, false);
}

