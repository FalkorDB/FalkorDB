/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "../index/indexer.h"
#include "index_operations.h"
#include "../graph/graph_hub.h"
#include "../datatypes/datatypes.h"
#include "../arithmetic/arithmetic_expression_construct.h"

// delete index
// DROP INDEX ON :N(name)
// DROP INDEX FOR (n:N) ON (n.name)
// DROP INDEX FOR ()-[e:R]-() ON (e.name)
// DROP FULLTEXT INDEX FOR (n:N) ON (n.name)
// DROP VECTOR INDEX FOR ()-[e:R]-() ON (e.name)
static bool index_delete
(
	GraphContext *gc,  // graph context
	AST *ast           // AST
) {
	Schema *s = NULL;
	const cypher_astnode_t *index_op = ast->root;

	cypher_astnode_type_t t = cypher_astnode_type(index_op);

	// extract label and attribute from AST
	const char *label;
	const char *attr;
	enum cypher_ast_index_type index_type;
	bool is_node;
	bool is_relation;
	if(t == CYPHER_AST_DROP_PROPS_INDEX) {
		label = cypher_ast_label_get_name(
			cypher_ast_drop_props_index_get_label(index_op));
	 	attr = cypher_ast_prop_name_get_value(
			cypher_ast_drop_props_index_get_prop_name(index_op, 0));
		index_type = CYPHER_INDEX_TYPE_RANGE;
		is_node = true;
		is_relation = true;
	} else {
		label = cypher_ast_label_get_name(
			cypher_ast_drop_pattern_props_index_get_label(index_op));
	 	attr = cypher_ast_prop_name_get_value(
			cypher_ast_property_operator_get_prop_name(
				cypher_ast_drop_pattern_props_index_get_property_operator(
				index_op, 0)));
		index_type = cypher_ast_drop_pattern_props_index_get_index_type(index_op);
		is_relation = cypher_ast_drop_pattern_props_index_pattern_is_relation(index_op);
		is_node = !is_relation;
	}

	// resolve attribute ID
	Attribute_ID attr_id = GraphContext_GetAttributeID(gc, attr);

	// lock
	QueryCtx_LockForCommit();

	// try deleting node RANGE index
	s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
	if(s != NULL) {
		if(Schema_GetIndex(s, &attr_id, 1, INDEX_FLD_RANGE, true) != NULL) {
			// try deleting a node range index
			// operation may fail if this index supports a constraint
			return GraphContext_DeleteIndex(gc, SCHEMA_NODE, label, attr,
					INDEX_FLD_RANGE);
		}
	}

	// try deleting edge RANGE index
	s = GraphContext_GetSchema(gc, label, SCHEMA_EDGE);
	if(s != NULL) {
		if(Schema_GetIndex(s, &attr_id, 1, INDEX_FLD_RANGE, true) != NULL) {
			// try deleting an edge range index
			// operation may fail if this index supports a constraint
			return GraphContext_DeleteIndex(gc, SCHEMA_EDGE, label, attr,
					INDEX_FLD_RANGE);
		}
	}

	// no matching index
	ErrorCtx_SetError(EMSG_UNABLE_TO_DROP_INDEX, label, attr);

	return false;
}

// extract index information from AST provided in the new format
// CREATE [RANGE|FULLTEXT|VECTOR] INDEX FOR (n:N) ON n.name
static void parse_new_format
(
	const cypher_astnode_t *index_op,  // AST index create node
	char **label,                      // label to index
	char ***fields,                    // fields to index
	uint *nfields,                     // number of fields to index
	GraphEntityType *et,               // entity type to index
    IndexFieldType *idx_type,          // index type
	SIValue *options                   // index options
) {
	ASSERT(et       != NULL);
	ASSERT(label    != NULL);
	ASSERT(fields   != NULL);
	ASSERT(nfields  != NULL);
	ASSERT(options  != NULL);
	ASSERT(idx_type != NULL);
	ASSERT(index_op != NULL);

	//--------------------------------------------------------------------------
	// extract label
	//--------------------------------------------------------------------------

	*label = (char*)cypher_ast_label_get_name(
			cypher_ast_create_pattern_props_index_get_label(index_op));

	//--------------------------------------------------------------------------
	// extract fields
	//--------------------------------------------------------------------------

	*nfields = cypher_ast_create_pattern_props_index_nprops(index_op);
	*fields = rm_malloc(sizeof(char*) * (*nfields));

	for(uint i = 0; i < *nfields; i++) {
		const cypher_astnode_t *field_name =
			cypher_ast_property_operator_get_prop_name(
					cypher_ast_create_pattern_props_index_get_property_operator(
						index_op, i));

		(*fields)[i] = (char*)cypher_ast_prop_name_get_value(field_name);
	}

	//--------------------------------------------------------------------------
	// extract entity type
	//--------------------------------------------------------------------------

	if(cypher_ast_create_pattern_props_index_pattern_is_relation(index_op)) {
		*et = GETYPE_EDGE;
	} else {
		*et = GETYPE_NODE;
	}

	//--------------------------------------------------------------------------
	// extract index type
	//--------------------------------------------------------------------------

	switch(cypher_ast_create_pattern_props_index_get_index_type(index_op)) {
		case CYPHER_INDEX_TYPE_RANGE:
			*idx_type = INDEX_FLD_RANGE;
			break;
		case CYPHER_INDEX_TYPE_FULLTEXT:
			*idx_type = INDEX_FLD_FULLTEXT;
			break;
		case CYPHER_INDEX_TYPE_VECTOR:
			*idx_type = INDEX_FLD_VECTOR;
			break;
	}

	//--------------------------------------------------------------------------
	// extract options
	//--------------------------------------------------------------------------

	const cypher_astnode_t *options_ast =
		cypher_ast_create_pattern_props_index_get_options(index_op);
	if(options_ast != NULL) {
		AR_ExpNode *exp = AR_EXP_FromASTNode(options_ast);
		*options = AR_EXP_Evaluate(exp, NULL);
		SIValue_Persist(options);
		AR_EXP_Free(exp);
	} else {
		*options = SI_Map(0);
	}
}

// extract index information from AST provided in the old format
// CREATE INDEX :N(n)
static void parse_old_format
(
	const cypher_astnode_t *index_op,  // AST index create node
	char **label,                      // label to index
	char ***fields,                    // fields to index
	uint *nfields,                     // number of fields to index
	GraphEntityType *et,               // entity type to index
    IndexFieldType *idx_type,          // index type
	SIValue *options                   // index options
) {
	ASSERT(et       != NULL);
	ASSERT(label    != NULL);
	ASSERT(fields   != NULL);
	ASSERT(nfields  != NULL);
	ASSERT(options  != NULL);
	ASSERT(idx_type != NULL);
	ASSERT(index_op != NULL);

	//--------------------------------------------------------------------------
	// extract label
	//--------------------------------------------------------------------------

	*label = (char*)cypher_ast_label_get_name(
			cypher_ast_create_node_props_index_get_label(index_op));

	//--------------------------------------------------------------------------
	// extract fields
	//--------------------------------------------------------------------------

	*nfields = cypher_ast_create_node_props_index_nprops(index_op);
	*fields = rm_malloc(sizeof(char*) * (*nfields));
	for(uint i = 0; i < *nfields ; i++) {
		const cypher_astnode_t *prop_name =
			cypher_ast_create_node_props_index_get_prop_name(index_op, i);
		(*fields)[i] = (char*)cypher_ast_prop_name_get_value(prop_name);
	}

	//--------------------------------------------------------------------------
	// set entity type
	//--------------------------------------------------------------------------

	*et = GETYPE_NODE;

	//--------------------------------------------------------------------------
	// set index type
	//--------------------------------------------------------------------------

	*idx_type = INDEX_FLD_RANGE;

	//--------------------------------------------------------------------------
	// set options
	//--------------------------------------------------------------------------

	*options = SI_Map(0);
}

// extract index level configuration from options map
static bool extract_index_level_config
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

	if(SI_TYPE(options) != T_MAP) return false;

	//--------------------------------------------------------------------------
	// extract language
	//--------------------------------------------------------------------------

	SIValue language_val;
	bool language_specified = MAP_GET(options, "language", language_val);

	if(language_specified) {
		if(SI_TYPE(language_val) != T_STRING) {
			ErrorCtx_SetError("Index configuration error");
			return false;
		} else {
			*language = language_val.stringval;
		}
	}

	//--------------------------------------------------------------------------
	// extract stopwords
	//--------------------------------------------------------------------------

	SIValue stopwords_val;
	bool stopwords_specified = MAP_GET(options, "stopwords", stopwords_val);

	if(stopwords_specified) {
		// validate stopwords is an array of strings
		if(SI_TYPE(stopwords_val) != T_ARRAY) {
			ErrorCtx_SetError("Index configuration error");
			return false;
		}

		if(!SIArray_AllOfType(stopwords_val, T_STRING)) {
			ErrorCtx_SetError("Index configuration error");
			return false;
		}

		uint nstopwords = SIArray_Length(stopwords_val);
		*stopwords = array_new(char*, nstopwords);
		for(uint i = 0; i < nstopwords; i++) {
			SIValue stopword = SIArray_Get(stopwords_val, i);
			array_append((*stopwords), rm_strdup(stopword.stringval));
		}
	}

	return true;
}

// create index
// CREATE INDEX ON :N(name)
// CREATE INDEX FOR (n:N) ON (n.name)
// CREATE INDEX FOR ()-[e:R]-() ON (e.name)
// CREATE FULLTEXT INDEX FOR (n:N) ON (n.name)
// CREATE VECTOR INDEX FOR ()-[e:R]-() ON (e.name)
static void index_create
(
	GraphContext *gc,  // graph context
	AST *ast           // AST
) {
	ASSERT(gc  != NULL);
	ASSERT(ast != NULL);

	const cypher_astnode_t *index_op = ast->root;

	//--------------------------------------------------------------------------
	// retrieve index label and attributes from AST
	//--------------------------------------------------------------------------

	uint            nfields  = 0;             // number of fields
	char            *label   = NULL;          // label to index
	char            **fields = NULL;          // fields to index
	GraphEntityType et       = GETYPE_NODE;   // type of entity to index
	SIValue         options  = SI_NullVal();  // index options
	IndexFieldType  idx_type;                 // index type

	// extract info from AST
	cypher_astnode_type_t t = cypher_astnode_type(index_op);
	if(t == CYPHER_AST_CREATE_NODE_PROPS_INDEX) {
		parse_old_format(index_op, &label, &fields, &nfields, &et, &idx_type,
				&options);
	} else {
		parse_new_format(index_op, &label, &fields, &nfields, &et, &idx_type,
				&options);
	}

	//--------------------------------------------------------------------------
	// index level configuration
	//--------------------------------------------------------------------------

	char *language   = NULL;
	char **stopwords = NULL;
	if(!extract_index_level_config(&stopwords, &language, options)) {
		// failed to extract index level configuration
		goto cleanup;
	}

	// validate all arguments are valid
	ASSERT(nfields > 0);
	ASSERT(label   != NULL);
	ASSERT(fields  != NULL);
	ASSERT(SI_TYPE(options) == T_MAP);
	ASSERT(et == GETYPE_NODE || et == GETYPE_EDGE);
	ASSERT(idx_type == INDEX_FLD_RANGE    ||
		   idx_type == INDEX_FLD_FULLTEXT ||
		   idx_type == INDEX_FLD_VECTOR);

	// lock
	QueryCtx_LockForCommit();

	Index idx = NULL;
	ResultSet *result_set = QueryCtx_GetResultSet();
	ASSERT(result_set != NULL);

	for(uint i = 0; i < nfields; i++) {
		idx = AddIndex(label, fields[i], et, idx_type, options, true);
		if(idx != NULL) {
			ResultSet_IndexCreated(result_set, INDEX_OK);
		} else {
			// operation failed
			goto cleanup;
		}
	}

	// index created, populate
	if(idx != NULL) {
		//----------------------------------------------------------------------
		// set index level configuration
		//----------------------------------------------------------------------

		if(language != NULL && !Index_SetLanguage(idx, language)) {
			goto cleanup;
		}

		if(stopwords != NULL && !Index_SetStopwords(idx, &stopwords)) {
			goto cleanup;
		}

		Index_Disable(idx);

		// populate index
		SchemaType st = (et == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE;
		Schema *s = GraphContext_GetSchema(gc, label, st);
		ASSERT(s != NULL);
		Indexer_PopulateIndex(gc, s, idx);
	}

cleanup:
	if(stopwords != NULL) array_free_cb(stopwords, rm_free);
	SIValue_Free(options);

	// make sure no effects were generated
	// as index creation isn't replicated via effects
	ASSERT(EffectsBuffer_Length(QueryCtx_GetEffectsBuffer()) == 0);
}

// handle index creation/deletion
void IndexOperation_Run
(
	GraphContext *gc,  // graph context
	AST *ast,          // AST
	ExecutionType op   // operation type
) {
	switch(op) {
		case EXECUTION_TYPE_INDEX_CREATE:
			index_create(gc, ast);
			break;
		case EXECUTION_TYPE_INDEX_DROP:
			index_delete(gc, ast);
			break;
		default:
			ErrorCtx_SetError(EMSG_UNKNOWN_EXECUTION_TYPE);
	}
}
