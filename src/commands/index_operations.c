/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "../query_ctx.h"
#include "../index/index.h"
#include "../index/indexer.h"
#include "index_operations.h"
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

	// extract label and attribute from AST
	const char *label = cypher_ast_label_get_name(
			cypher_ast_drop_props_index_get_label(index_op));
	const char *attr = cypher_ast_prop_name_get_value(
			cypher_ast_drop_props_index_get_prop_name(index_op, 0));

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

// create index
// CREATE INDEX ON :N(name)
// CREATE INDEX FOR (n:N) ON (n.name)
// CREATE INDEX FOR ()-[e:R]-() ON (e.name)
// CREATE FULLTEXT INDEX FOR (n:N) ON (n.name)
// CREATE VECTOR INDEX FOR ()-[e:R]-() ON (e.name)
static void index_create
(
	RedisModuleCtx *ctx,  // Redis context
	GraphContext *gc,     // graph context
	AST *ast              // AST
) {
	ASSERT(gc  != NULL);
	ASSERT(ctx != NULL);
	ASSERT(ast != NULL);

	uint nfields       = 0;            // number of fields to index
	const char *label  = NULL;         // label to index
	GraphEntityType et = GETYPE_NODE;  // type of entity to index

	const cypher_astnode_t *index_op = ast->root;
    enum cypher_ast_index_type idx_type = CYPHER_INDEX_TYPE_RANGE;  // idx type
	cypher_astnode_type_t t = cypher_astnode_type(index_op);
    SIValue options = SI_NullVal();

	//--------------------------------------------------------------------------
	// retrieve index label and attributes from AST
	//--------------------------------------------------------------------------

	if(t == CYPHER_AST_CREATE_NODE_PROPS_INDEX) {
		// old format
		// CREATE INDEX ON :N(name)
		nfields = cypher_ast_create_node_props_index_nprops(index_op);
		label   = cypher_ast_label_get_name(
				cypher_ast_create_node_props_index_get_label(index_op));
	} else {
		// new format
		// CREATE [RANGE|FULLTEXT|VECTOR] INDEX FOR (n:N) ON n.name
		nfields = cypher_ast_create_pattern_props_index_nprops(index_op);
		label   = cypher_ast_label_get_name(
				cypher_ast_create_pattern_props_index_get_label(index_op));

		// determine indexed entity type
		if(cypher_ast_create_pattern_props_index_pattern_is_relation(index_op)){
			et = GETYPE_EDGE;
		}

		// determine index type
        idx_type =
			cypher_ast_create_pattern_props_index_get_index_type(index_op);

		// get index options
        const cypher_astnode_t *options_ast =
			cypher_ast_create_pattern_props_index_get_options(index_op);
        if(options_ast != NULL) {
            AR_ExpNode *exp = AR_EXP_FromASTNode(options_ast);
            options = AR_EXP_Evaluate(exp, NULL);
			AR_EXP_Free(exp);
        }
	}

	ASSERT(nfields > 0);
	ASSERT(label != NULL);

	const char *fields[nfields];
	for(uint i = 0; i < nfields ; i++) {
		const cypher_astnode_t *prop_name =
			(t == CYPHER_AST_CREATE_NODE_PROPS_INDEX) ?
			cypher_ast_create_node_props_index_get_prop_name(index_op, i) :
			cypher_ast_property_operator_get_prop_name(
					cypher_ast_create_pattern_props_index_get_property_operator(
						index_op, i));

		fields[i] = cypher_ast_prop_name_get_value(prop_name);
	}

	// lock
	QueryCtx_LockForCommit();

	Index idx = NULL;
	// add fields to index
    switch(idx_type) {
        case CYPHER_INDEX_TYPE_RANGE:
            idx = Index_RangeCreate(label, et, (const char**)fields, nfields);
            break;

        case CYPHER_INDEX_TYPE_FULLTEXT:
			idx = Index_FulltextCreate(label, et, fields[0], options);
			break;

		case CYPHER_INDEX_TYPE_VECTOR:
			idx = Index_VectorCreate(label, et, fields[0], options);
            break;

        default:
            ASSERT(false);
            break;
    }

	if(idx != NULL) {
		// build index
		SchemaType st = (et == GETYPE_NODE) ? SCHEMA_NODE : SCHEMA_EDGE;
		Schema *s = GraphContext_GetSchema(gc, label, st);
		Indexer_PopulateIndex(gc, s, idx);
	}
}

// handle index creation/deletion
void IndexOperation_Run
(
	RedisModuleCtx *ctx,  // Redis context
	GraphContext *gc,     // graph context
	AST *ast,             // AST
	ExecutionType op      // operation type
) {
	switch(op) {
		case EXECUTION_TYPE_INDEX_CREATE:
			index_create(ctx, gc, ast);
			break;
		case EXECUTION_TYPE_INDEX_DROP:
			index_delete(gc, ast);
			break;
		default:
			ErrorCtx_SetError(EMSG_UNKNOWN_EXECUTION_TYPE);
	}
}

