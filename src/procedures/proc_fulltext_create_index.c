/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "proc_fulltext_create_index.h"
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
		if(SI_TYPE(lang) != T_STRING) {
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

	if((SI_TYPE(field) & T_STRING) == 0) {
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
		if(SI_TYPE(phonetic) != T_STRING) {
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
		*stopwords = array_new(char*, nstopwords);
		for(uint i = 0; i < nstopwords; i++) {
			SIValue stopword = SIArray_Get(stopwords_val, i);
			array_append((*stopwords), rm_strdup(stopword.stringval));
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
	uint arg_count = array_len((SIValue *)args);
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
	if(SI_TYPE(label_config) == T_STRING) {
		label = label_config.stringval;
	} else if(SI_TYPE(label_config) == T_MAP) {
		SIValue label_value;
		MAP_GET(label_config, "label", label_value);
		label = label_value.stringval;
	}

	// label is mandatory
	ASSERT(label != NULL);

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

	// collect fields and configuration
	for(uint i = 0; i < fields_count; i++) {
		weights[i]   = INDEX_FIELD_DEFAULT_WEIGHT;
		nostems[i]   = INDEX_FIELD_DEFAULT_NOSTEM;
		phonetics[i] = INDEX_FIELD_DEFAULT_PHONETIC;

		if(SI_TYPE(fields[i]) == T_STRING) {
			_fields[i] = fields[i].stringval;
		} else {
			SIValue tmp;
			MAP_GET(fields[i], "field", tmp);
			_fields[i] = tmp.stringval;

			if(MAP_GET(fields[i], "weight", tmp)) {
				weights[i] = SI_GET_NUMERIC(tmp);
			}
			if(MAP_GET(fields[i], "nostem", tmp)) {
				nostems[i] = tmp.longval;
			}
			if(MAP_GET(fields[i], "phonetic", tmp)) {
				phonetics[i] = tmp.stringval;
			}
		}
	}

	//--------------------------------------------------------------------------
	// create index one field at a time
	//--------------------------------------------------------------------------

	ResultSet *result_set = QueryCtx_GetResultSet();
	ASSERT(result_set != NULL);

	SIValue options = SI_Map(3);
	for(uint i = 0; i < fields_count; i++) {
		// construct options map
		Map_Add(&options, SI_ConstStringVal("weight"),
				SI_DoubleVal(weights[i]));
		Map_Add(&options, SI_ConstStringVal("phonetic"),
				SI_ConstStringVal(phonetics[i]));
		Map_Add(&options, SI_ConstStringVal("nostem"),
				SI_BoolVal(nostems[i]));

		idx = AddIndex(label, _fields[i], GETYPE_NODE, INDEX_FLD_FULLTEXT,
				options, true);
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

		extract_index_level_config(&stopwords, &language, label_config);

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
	if(stopwords != NULL) array_free_cb(stopwords, rm_free);
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
	ProcedureOutput *output = array_new(ProcedureOutput, 0);
	return ProcCtxNew("db.idx.fulltext.createNodeIndex",
			PROCEDURE_VARIABLE_ARG_COUNT, output,
			Proc_FulltextCreateNodeIdxStep, Proc_FulltextCreateNodeIdxInvoke,
			NULL, NULL, false, "Builds a full-text searchable index on a label and the 1 or more specified properties.");
}

