/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "../index/indexer.h"
#include "../index/cch_index.h"
#include "index_operations.h"
#include "../graph/graph_hub.h"
#include "../effects/effects.h"
#include "../util/arr.h"
#include "../datatypes/datatypes.h"
#include "../arithmetic/arithmetic_expression_construct.h"

#include <strings.h>   // strcasecmp

// forward declarations (definitions further down)
static const char *_create_index_type_name(const cypher_astnode_t *op);
static const char *_drop_index_type_name(const cypher_astnode_t *op);
static bool _is_cch(const char *type_name);
static void cch_index_create(GraphContext *gc, const cypher_astnode_t *index_op);
static bool cch_index_drop(GraphContext *gc, const cypher_astnode_t *op);

// parse drop index old format
// DROP INDEX ON :N(name)
static void _index_delete_parse_old_format
(
	bool *is_node,              // node index
	bool *is_relation,          // relation index
	const char **attr,          // attribute to index
	const char **label,         // label to index
	IndexFieldType *idx_type,   // index type
	const cypher_astnode_t *op  // AST drop index node
) {
	ASSERT(op          != NULL);
	ASSERT(attr        != NULL);
	ASSERT(label       != NULL);
	ASSERT(is_node     != NULL);
	ASSERT(is_relation != NULL);

	// extract label
	*label = cypher_ast_label_get_name(
			cypher_ast_drop_props_index_get_label(op));

	// extract attribute
	*attr = cypher_ast_prop_name_get_value(
			cypher_ast_drop_props_index_get_prop_name(op, 0));

	// we don't know if this is a node or relation index
	*is_node     = true;
	*is_relation = true;
	*idx_type     = INDEX_FLD_RANGE;
}

// parse drop index new format
// DROP VECTOR INDEX FOR (n:N) ON (n.name)
static void _index_delete_parse_new_format
(
	bool *is_node,              // node index
	bool *is_relation,          // relation index
	const char **attr,          // attribute to index
	const char **label,         // label to index
	IndexFieldType *idx_type,   // index type
	const cypher_astnode_t *op  // AST drop index node
) {
	ASSERT(op          != NULL);
	ASSERT(attr        != NULL);
	ASSERT(label       != NULL);
	ASSERT(is_node     != NULL);
	ASSERT(is_relation != NULL);

	// extract label
	*label = cypher_ast_label_get_name(
			cypher_ast_drop_pattern_props_index_get_label(op));

	// extract attribute
	*attr = cypher_ast_prop_name_get_value(
			cypher_ast_property_operator_get_prop_name(
				cypher_ast_drop_pattern_props_index_get_property_operator(op,
					0)));

	// determine if this is a node or relation index
	*is_relation = cypher_ast_drop_pattern_props_index_pattern_is_relation(op);
	*is_node = !*is_relation;

	// determine index type (NULL keyword => plain range); CCH drops take their
	// own path and never reach here
	const char *type_name = _drop_index_type_name(op);
	if(type_name == NULL) {
		*idx_type = INDEX_FLD_RANGE;
	} else if(strcasecmp(type_name, "fulltext") == 0) {
		*idx_type = INDEX_FLD_FULLTEXT;
	} else if(strcasecmp(type_name, "vector") == 0) {
		*idx_type = INDEX_FLD_VECTOR;
	} else {
		*idx_type = INDEX_FLD_RANGE;
	}
}

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
	const cypher_astnode_t *op      = ast->root;
	cypher_astnode_type_t  t        = cypher_astnode_type(op);
	IndexFieldType         idx_type = INDEX_FLD_RANGE;

	// extract label and attribute from AST
	Schema     *s           = NULL;   // schema
	bool       is_node      = false;  // node index
	bool       is_relation  = false;  // relation index
	const char *lbl         = NULL;   // removed label
	const char *attr        = NULL;   // removed attribute

	// a CCH path index drop takes its own path (own object + effect)
	if(t == CYPHER_AST_DROP_PATTERN_PROPS_INDEX &&
			_is_cch(_drop_index_type_name(op))) {
		return cch_index_drop(gc, op);
	}

	if(t == CYPHER_AST_DROP_PROPS_INDEX) {
		_index_delete_parse_old_format(&is_node, &is_relation, &attr, &lbl,
				&idx_type, op);
	} else {
		_index_delete_parse_new_format(&is_node, &is_relation, &attr, &lbl,
				&idx_type, op);
	}

	//--------------------------------------------------------------------------
	// resolve attribute ID
	//--------------------------------------------------------------------------

	// quickly return if attribute doesn't exist
	AttributeID attr_id = GraphContext_GetAttributeID(gc, attr);
	if(attr_id == ATTRIBUTE_ID_NONE) {
		ErrorCtx_SetError(EMSG_UNABLE_TO_DROP_INDEX, lbl, attr);
		return false;
	}

	//--------------------------------------------------------------------------
	// resolve schema
	//--------------------------------------------------------------------------

	// lock
	QueryCtx_AcquireWriteLock () ;

	ResultSet *result_set = QueryCtx_GetResultSet () ;

	if (is_node) {
		// try deleting node index
		s = GraphContext_GetSchema (gc, lbl, SCHEMA_NODE) ;
		if (s != NULL) {
			if (Schema_GetIndex (s, &attr_id, 1, idx_type, true) != NULL) {
				// try deleting a node index
				// operation may fail if this index supports a constraint
				int res = GraphHub_DropIndex (gc, SCHEMA_NODE, lbl, attr,
						idx_type, true) ;
				if (res == INDEX_OK) {
					ResultSet_IndexDeleted (result_set, res) ;
				}
				return res == INDEX_OK ;
			}
		}
	}

	if (is_relation) {
		// try deleting edge index
		s = GraphContext_GetSchema (gc, lbl, SCHEMA_EDGE) ;
		if (s != NULL) {
			if (Schema_GetIndex (s, &attr_id, 1, idx_type, true) != NULL) {
				// try deleting an edge index
				// operation may fail if this index supports a constraint
				int res = GraphHub_DropIndex (gc, SCHEMA_EDGE, lbl, attr,
						idx_type, true) ;
				if (res == INDEX_OK) {
					ResultSet_IndexDeleted (result_set, res) ;
				}
				return res == INDEX_OK ;
			}
		}
	}

	// no matching index
	ErrorCtx_SetError (EMSG_UNABLE_TO_DROP_INDEX, lbl, attr) ;

	return false ;
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

	// index type keyword (NULL => plain range); CCH is handled on its own path
	const char *type_name = _create_index_type_name(index_op);
	if(type_name == NULL) {
		*idx_type = INDEX_FLD_RANGE;
	} else if(strcasecmp(type_name, "fulltext") == 0) {
		*idx_type = INDEX_FLD_FULLTEXT;
	} else if(strcasecmp(type_name, "vector") == 0) {
		*idx_type = INDEX_FLD_VECTOR;
	} else {
		// unknown types are rejected during validation; default keeps this safe
		*idx_type = INDEX_FLD_RANGE;
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
bool IndexOperation_ExtractLevelConfig
(
	char ***stopwords,  // index stopwods
	char **language,    // index language
	SIValue options     // options map
) {
	ASSERT (language  != NULL) ;
	ASSERT (stopwords != NULL) ;

	// set default values
	*language  = NULL ;
	*stopwords = NULL ;

	if (SI_TYPE (options) != T_MAP) {
		return false ;
	}

	//--------------------------------------------------------------------------
	// extract language
	//--------------------------------------------------------------------------

	SIValue language_val ;
	bool language_specified = MAP_GET (options, "language", language_val) ;

	if (language_specified) {
		if(SI_TYPE (language_val) != T_STRING) {
			ErrorCtx_SetError ("Index configuration error") ;
			return false ;
		} else {
			*language = language_val.stringval ;
		}
	}

	//--------------------------------------------------------------------------
	// extract stopwords
	//--------------------------------------------------------------------------

	SIValue stopwords_val ;
	bool stopwords_specified = MAP_GET (options, "stopwords", stopwords_val) ;

	if (stopwords_specified) {
		// validate stopwords is an array of strings
		if(SI_TYPE (stopwords_val) != T_ARRAY) {
			ErrorCtx_SetError ("Index configuration error") ;
			return false ;
		}

		if(!SIArray_AllOfType (stopwords_val, T_STRING)) {
			ErrorCtx_SetError ("Index configuration error") ;
			return false ;
		}

		uint nstopwords = SIArray_Length (stopwords_val) ;
		*stopwords = arr_new (char*, nstopwords) ;
		for (uint i = 0; i < nstopwords; i++) {
			SIValue stopword = SIArray_Get (stopwords_val, i) ;
			arr_append ((*stopwords), rm_strdup (stopword.stringval)) ;
		}
	}

	return true ;
}

// the index-type keyword string of a create/drop pattern-props-index node, or
// NULL for a plain range index
static const char *_create_index_type_name(const cypher_astnode_t *op) {
	const cypher_astnode_t *t =
		cypher_ast_create_pattern_props_index_get_index_type(op);
	return (t == NULL) ? NULL : cypher_ast_string_get_value(t);
}

static const char *_drop_index_type_name(const cypher_astnode_t *op) {
	const cypher_astnode_t *t =
		cypher_ast_drop_pattern_props_index_get_index_type(op);
	return (t == NULL) ? NULL : cypher_ast_string_get_value(t);
}

static bool _is_cch(const char *type_name) {
	return type_name != NULL && strcasecmp(type_name, "cch") == 0;
}

// collect the relationship-type names of a pattern-props-index node into a fresh
// arr_ of (non-owned, parser-owned) name strings. a relationship pattern index
// may span several types: ()-[e:A|B]->()
static const char **_index_reltype_names(const cypher_astnode_t *op) {
	const char **names = arr_new(const char *, 1);
	uint n = cypher_astnode_nchildren(op);
	for(uint i = 0; i < n; i++) {
		const cypher_astnode_t *c = cypher_astnode_get_child(op, i);
		if(cypher_astnode_type(c) == CYPHER_AST_LABEL) {
			arr_append(names, cypher_ast_label_get_name(c));
		}
	}
	return names;
}

// CREATE CCH INDEX FOR ()-[e:A|B]->() ON (e.weight)
// builds a Customizable Contraction Hierarchy path index over the relationship
// types for the weight property. unknown relationship types / weight attribute
// are created (as any CREATE INDEX may introduce new schema). writes nothing to
// the graph; the hierarchy lives inside the index.
static void cch_index_create
(
	GraphContext *gc,                  // graph context
	const cypher_astnode_t *index_op   // AST create pattern-props-index node
) {
	// CCH is a relationship pathfinding index
	if(!cypher_ast_create_pattern_props_index_pattern_is_relation(index_op)) {
		ErrorCtx_SetError("CCH index is only supported on relationships");
		return;
	}

	// exactly one property: the edge weight
	uint nprops = cypher_ast_create_pattern_props_index_nprops(index_op);
	if(nprops != 1) {
		ErrorCtx_SetError("CCH index requires exactly one property, the edge weight");
		return;
	}
	const char *weight = cypher_ast_prop_name_get_value(
			cypher_ast_property_operator_get_prop_name(
				cypher_ast_create_pattern_props_index_get_property_operator(
					index_op, 0)));

	const char **rel_names = _index_reltype_names(index_op);   // >= 1 (grammar)

	QueryCtx_AcquireWriteLock();

	// resolve-or-create the relationship schemas + weight attribute; these emit
	// their own schema/attribute effects, ordered before the CCH effect below
	uint rc = arr_len(rel_names);
	RelationID *rel_ids = arr_new(RelationID, rc);
	for(uint i = 0; i < rc; i++) {
		Schema *s = GraphHub_AddSchema(gc, rel_names[i], SCHEMA_EDGE, true);
		arr_append(rel_ids, Schema_GetID(s));
	}
	AttributeID weight_attr = GraphHub_FindOrAddAttribute(gc, weight, true);

	// one CCH index per (relTypes, weightProp)
	if(GraphContext_GetCCHIndex(gc, rel_ids, arr_len(rel_ids), weight_attr)
			!= NULL) {
		ErrorCtx_SetError("a CCH index already exists over these relationship "
				"types and weight attribute");
		arr_free(rel_ids);
		arr_free(rel_names);
		return;
	}

	CCHIndex *idx = CCHIndex_New(rel_ids, arr_len(rel_ids), weight_attr);
	CCHIndex_Build(idx, GraphContext_GetGraph(gc));
	GraphContext_AddCCHIndex(gc, idx);

	ResultSet_IndexCreated(QueryCtx_GetResultSet(), INDEX_OK);

	// definition-only create-CCH effect (replication + AOF); each receiver
	// rebuilds its own hierarchy
	EffectsBuffer_AddCreateCCHEffect(QueryCtx_GetEffectsBuffer(), rel_ids,
			rel_names, rc, weight_attr, weight);

	arr_free(rel_ids);
	arr_free(rel_names);
}

// DROP CCH INDEX FOR ()-[e:A|B]->() ON (e.weight)
static bool cch_index_drop
(
	GraphContext *gc,            // graph context
	const cypher_astnode_t *op   // AST drop pattern-props-index node
) {
	if(!cypher_ast_drop_pattern_props_index_pattern_is_relation(op)) {
		ErrorCtx_SetError("CCH index is only supported on relationships");
		return false;
	}
	uint nprops = cypher_ast_drop_pattern_props_index_nprops(op);
	if(nprops != 1) {
		ErrorCtx_SetError("CCH index requires exactly one property, the edge weight");
		return false;
	}
	const char *weight = cypher_ast_prop_name_get_value(
			cypher_ast_property_operator_get_prop_name(
				cypher_ast_drop_pattern_props_index_get_property_operator(op, 0)));

	const char **rel_names = _index_reltype_names(op);

	// resolve ids -- all must already exist to match a live index
	uint rc = arr_len(rel_names);
	RelationID *rel_ids = arr_new(RelationID, rc);
	AttributeID weight_attr = GraphContext_GetAttributeID(gc, weight);
	bool resolvable = (weight_attr != ATTRIBUTE_ID_NONE);
	for(uint i = 0; i < rc && resolvable; i++) {
		Schema *s = GraphContext_GetSchema(gc, rel_names[i], SCHEMA_EDGE);
		if(s == NULL) { resolvable = false; break; }
		arr_append(rel_ids, Schema_GetID(s));
	}

	QueryCtx_AcquireWriteLock();

	bool removed = resolvable && GraphContext_RemoveCCHIndex(gc, rel_ids,
			arr_len(rel_ids), weight_attr);
	if(!removed) {
		ErrorCtx_SetError("no CCH index over these relationship types and weight "
				"attribute");
		arr_free(rel_ids);
		arr_free(rel_names);
		return false;
	}

	ResultSet_IndexDeleted(QueryCtx_GetResultSet(), INDEX_OK);
	EffectsBuffer_AddDropCCHEffect(QueryCtx_GetEffectsBuffer(), rel_ids,
			rel_names, arr_len(rel_ids), weight_attr, weight);

	arr_free(rel_ids);
	arr_free(rel_names);
	return true;
}

// create index
// CREATE INDEX ON :N(name)
// CREATE INDEX FOR (n:N) ON (n.name)
// CREATE INDEX FOR ()-[e:R]-() ON (e.name)
// CREATE FULLTEXT INDEX FOR (n:N) ON (n.name)
// CREATE VECTOR INDEX FOR ()-[e:R]-() ON (e.name)
// CREATE CCH INDEX FOR ()-[e:A|B]->() ON (e.name)
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

	// a CCH path index is a graph-level structure, not a Schema/RediSearch index;
	// it takes a dedicated path (own object + effect), bypassing the generic flow
	if(t == CYPHER_AST_CREATE_PATTERN_PROPS_INDEX &&
			_is_cch(_create_index_type_name(index_op))) {
		cch_index_create(gc, index_op);
		return;
	}

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
	if(!IndexOperation_ExtractLevelConfig(&stopwords, &language, options)) {
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
	QueryCtx_AcquireWriteLock () ;

	Index idx = NULL;
	ResultSet *result_set = QueryCtx_GetResultSet();
	ASSERT(result_set != NULL);

	for(uint i = 0; i < nfields; i++) {
		idx = GraphHub_AddIndex(gc, label, fields[i], et, idx_type, options, true);
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
	if(fields    != NULL) rm_free(fields);
	if(stopwords != NULL) arr_free_cb(stopwords, rm_free);
	SIValue_Free(options);
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

