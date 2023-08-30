/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../query_ctx.h"
#include "../index/indexer.h"
#include "index_operations.h"
#include "../datatypes/datatypes.h"
#include "../arithmetic/arithmetic_expression_construct.h"

bool _index_operation_delete
(
	GraphContext *gc,
	AST *ast
) {
	Schema *s = NULL;
	SchemaType schema_type = SCHEMA_NODE;
	const cypher_astnode_t *index_op = ast->root;

	// retrieve strings from AST node
	const char *label = cypher_ast_label_get_name(
			cypher_ast_drop_props_index_get_label(index_op));
	const char *attr = cypher_ast_prop_name_get_value(
			cypher_ast_drop_props_index_get_prop_name(index_op, 0));

	Attribute_ID attr_id = GraphContext_GetAttributeID(gc, attr);

	// try deleting a NODE EXACT-MATCH index

	// lock
	QueryCtx_LockForCommit();

	s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
	if(s != NULL) {
		if(Schema_GetIndex(s, &attr_id, 1, INDEX_FLD_RANGE, true) != NULL) {
			// try deleting an exact match node index
			return GraphContext_DeleteIndex(gc, SCHEMA_NODE, label, attr,
					INDEX_FLD_RANGE);
		}
	}

	// try removing from an edge schema
	s = GraphContext_GetSchema(gc, label, SCHEMA_EDGE);
	if(s != NULL) {
		if(Schema_GetIndex(s, &attr_id, 1, INDEX_FLD_RANGE, true) != NULL) {
			// try deleting an exact match edge index
			return GraphContext_DeleteIndex(gc, SCHEMA_EDGE, label, attr,
					INDEX_FLD_RANGE);
		}
	}

	// no matching index
	ErrorCtx_SetError(EMSG_UNABLE_TO_DROP_INDEX, label, attr);

	return false;
}

// create index structure
static void _index_operation_create
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	AST *ast
) {
	ASSERT(gc  != NULL);
	ASSERT(ctx != NULL);
	ASSERT(ast != NULL);

	uint nprops            = 0;            // number of fields indexed
	const char *label      = NULL;         // label being indexed
	SchemaType schema_type = SCHEMA_NODE;  // type of entities being indexed

    enum cypher_ast_index_type type = range_index; // index type
	const cypher_astnode_t *index_op = ast->root;
	cypher_astnode_type_t t = cypher_astnode_type(index_op);
    SIValue options = SI_NullVal();

	//--------------------------------------------------------------------------
	// retrieve label and attributes from AST
	//--------------------------------------------------------------------------

	if(t == CYPHER_AST_CREATE_NODE_PROPS_INDEX) {
		// old format
		// CREATE INDEX ON :N(name)
		nprops = cypher_ast_create_node_props_index_nprops(index_op);
		label  = cypher_ast_label_get_name(
				cypher_ast_create_node_props_index_get_label(index_op));
	} else {
		// new format
		// CREATE [FULLTEXT][VECTOR] INDEX FOR (n:N) ON n.name
		nprops = cypher_ast_create_pattern_props_index_nprops(index_op);
		label  = cypher_ast_label_get_name(
				cypher_ast_create_pattern_props_index_get_label(index_op));

		// determine if index is created over node label or edge relationship
		// default to node
		if(cypher_ast_create_pattern_props_index_pattern_is_relation(index_op)) {
			schema_type = SCHEMA_EDGE;
		}

        type = cypher_ast_create_pattern_props_index_get_index_type(index_op);
        const cypher_astnode_t *options_ast = cypher_ast_create_pattern_props_index_get_options(index_op);
        if(options_ast) {
            AR_ExpNode *exp = AR_EXP_FromASTNode(options_ast);
            options = AR_EXP_Evaluate(exp, NULL);
        }
	}

	ASSERT(nprops > 0);
	ASSERT(label != NULL);

	const char *fields[nprops];
	for(uint i = 0; i < nprops; i++) {
		const cypher_astnode_t *prop_name =
			(t == CYPHER_AST_CREATE_NODE_PROPS_INDEX) ?
			cypher_ast_create_node_props_index_get_prop_name(index_op, i) :
			cypher_ast_property_operator_get_prop_name
			(cypher_ast_create_pattern_props_index_get_property_operator(index_op, i));

		fields[i] = cypher_ast_prop_name_get_value(prop_name);
	}

	// lock
	QueryCtx_LockForCommit();

	Index idx;
    bool created = false;
	// add fields to index
    switch(type) {
        case range_index:
            created = GraphContext_AddRangeIndex(&idx, gc, schema_type, label,
                fields, nprops);
            break;
        case fulltext_index: {
            char **stopwords     = NULL;
            const char *language = NULL;

            bool        nostems[nprops];
            double      weights[nprops];
            const char* phonetics[nprops];

            // collect fields
            for(uint i = 0; i < nprops; i++) {
                weights[i]   = INDEX_FIELD_DEFAULT_WEIGHT;
                nostems[i]   = INDEX_FIELD_DEFAULT_NOSTEM;
                phonetics[i] = INDEX_FIELD_DEFAULT_PHONETIC;

                SIValue tmp;
                if(!SIValue_IsNull(options)) {
                    if(Map_Get(options, SI_ConstStringVal("weight"), &tmp)) {
                        weights[i] = SI_GET_NUMERIC(tmp);
                    }
                    if(Map_Get(options, SI_ConstStringVal("nostem"), &tmp)) {
                        nostems[i] = tmp.longval;
                    }
                    if(Map_Get(options, SI_ConstStringVal("phonetic"), &tmp)) {
                        phonetics[i] = tmp.stringval;
                    }
                }
            }
            created = GraphContext_AddFullTextIndex(&idx, gc, label, fields,
                nprops, weights, nostems, phonetics, stopwords, language);
            break;
        }
        case vector_index:
            SIValue dim = SI_NullVal();
            if(SIValue_IsNull(options) || 
               !Map_Get(options, SI_ConstStringVal("dim"), &dim)) {
                ErrorCtx_SetError(EMSG_VECTOR_IDX_CREATE_FAIL);
                return;
            }
            created = GraphContext_AddVectorIndex(&idx, gc, schema_type, label,
                fields[0], dim.longval);
            break;
        default:
            ASSERT(false);
            break;
    }
	if(created) {
		Schema *s = GraphContext_GetSchema(gc, label, schema_type);
		Indexer_PopulateIndex(gc, s, idx);
	}
}

// handle index/constraint operation
// either index/constraint creation or index/constraint deletion
void IndexOperation_Run
(
	RedisModuleCtx *ctx,
	GraphContext *gc,
	AST *ast,
	ExecutionType exec_type
) {
	switch(exec_type) {
		case EXECUTION_TYPE_INDEX_CREATE:
			_index_operation_create(ctx, gc, ast);
			break;
		case EXECUTION_TYPE_INDEX_DROP:
			_index_operation_delete(gc, ast);
			break;
		default:
			ErrorCtx_SetError(EMSG_UNKNOWN_EXECUTION_TYPE);
	}
}
