/* Fix for Issue #622: Crash
 * When a property access expression (e.g. x.n1) is used within a node pattern
 * inside a CALL subquery referencing an outer variable, the expression resolution
 * fails to properly handle the outer variable reference, resulting in a null
 * pointer dereference in _convertPropertyMap.
 */

#include "RG.h"
#include "ast_build_filter_tree.h"
#include "ast_shared.h"
#include "../util/arr.h"
#include "../errors/errors.h"
#include "../arithmetic/arithmetic_expression_construct.h"

// Forward declaration
FT_FilterNode *_FilterNode_FromAST(const cypher_astnode_t *expr);

FT_FilterNode *_CreatePredicateFilterNode(AST_Operator op, const cypher_astnode_t *lhs,
  const cypher_astnode_t *rhs) {
	return FilterTree_CreatePredicateFilter(op, AR_EXP_FromASTNode(lhs), AR_EXP_FromASTNode(rhs));
}

void _FT_Append(FT_FilterNode **root_ptr, FT_FilterNode *child) {
	ASSERT(child);

	FT_FilterNode *root = *root_ptr;
	// If the tree is uninitialized, its root is the child
	if(root == NULL) {
		*root_ptr = child;
		return;
	}
}

static FT_FilterNode *_convertPropertyMap(
	const cypher_astnode_t *entity,
	const cypher_astnode_t *prop_map,
	GraphContext *gc
) {
	uint nelems = cypher_ast_map_nentries(prop_map);
	if(nelems == 0) return NULL;

	FT_FilterNode *root = NULL;

	for(uint i = 0; i < nelems; i++) {
		const cypher_astnode_t *key = cypher_ast_map_get_key(prop_map, i);
		const cypher_astnode_t *val = cypher_ast_map_get_value(prop_map, i);

		/* Issue #622: when an outer-scope property reference (e.g. x.n1) cannot
		 * be resolved during AST compilation, val is NULL. Silently skipping it
		 * with 'continue' would drop the predicate and widen MATCH semantics,
		 * potentially returning rows that should not match. Instead, propagate
		 * NULL upward so the caller can surface a proper planning error.
		 */
		if(val == NULL) {
			ErrorCtx_SetError(EMSG_UNKNOWN,
				"Property map value could not be resolved; "
				"outer-scope variable reference in CALL subquery pattern "
				"is not supported at plan time");
			FilterTree_Free(root);
			return NULL;
		}

		const char *prop_name = cypher_ast_prop_name_get_value(key);

		/* Build equality predicate: entity.prop_name = val
		 * Each allocation is checked individually; on failure we free any
		 * partial state, set an error, and return NULL to match the error
		 * handling pattern above (mirrors Cppcheck findings: nullPointerOutOfMemory,
		 * memleak on lines 63-69).
		 */
		AR_ExpNode *inner = AR_EXP_NewVariableOperandNode(
			cypher_ast_identifier_get_name(entity));
		if(inner == NULL) {
			ErrorCtx_SetError(EMSG_UNKNOWN, "Failed to allocate variable operand node");
			FilterTree_Free(root);
			return NULL;
		}

		AR_ExpNode *lhs = AR_EXP_NewAttributeAccessNode(inner, prop_name);
		if(lhs == NULL) {
			AR_EXP_Free(inner);
			ErrorCtx_SetError(EMSG_UNKNOWN, "Failed to allocate attribute access node");
			FilterTree_Free(root);
			return NULL;
		}

		AR_ExpNode *rhs = AR_EXP_FromASTNode(val);
		if(rhs == NULL) {
			AR_EXP_Free(lhs);
			ErrorCtx_SetError(EMSG_UNKNOWN, "Failed to build RHS expression from AST node");
			FilterTree_Free(root);
			return NULL;
		}

		FT_FilterNode *pred = FilterTree_CreatePredicateFilter(OP_EQUAL, lhs, rhs);
		if(pred == NULL) {
			AR_EXP_Free(lhs);
			AR_EXP_Free(rhs);
			ErrorCtx_SetError(EMSG_UNKNOWN, "Failed to create predicate filter node");
			FilterTree_Free(root);
			return NULL;
		}

		if(root == NULL) {
			root = pred;
		} else {
			FT_FilterNode *and_node = FilterTree_CreateConditionFilter(OP_AND);
			if(and_node == NULL) {
				FilterTree_Free(pred);
				ErrorCtx_SetError(EMSG_UNKNOWN, "Failed to create AND condition filter node");
				FilterTree_Free(root);
				return NULL;
			}
			FilterTree_AppendLeftChild(and_node, root);
			FilterTree_AppendRightChild(and_node, pred);
			root = and_node;
		}
	}

	return root;
}
