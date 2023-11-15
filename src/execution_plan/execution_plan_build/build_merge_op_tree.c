/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "../ops/ops.h"
#include "../../query_ctx.h"
#include "../execution_plan.h"
#include "execution_plan_util.h"
#include "execution_plan_modify.h"
#include "execution_plan_construct.h"
#include "../../util/rax_extensions.h"
#include "../../ast/ast_build_op_contexts.h"

void buildMergeOp
(
	ExecutionPlan *plan,
	AST *ast,
	const cypher_astnode_t *clause,
	GraphContext *gc
) {
	// a MERGE clause provides a single path that must exist or be created
	// if we have built ops already
	// they will form the bound branch into the Merge op
	// a clone of the Record produced by this stream will be passed into the
	// other Merge streams
	// so that they properly work with bound variables
	//
	// as with paths in a MATCH query, build the appropriate traversal ops
	// and add them as another stream into Merge
	//
	// finally, we'll add a last stream that creates the pattern
	// if it did not get matched
	//
	// simple case (2 streams, no bound variables):
	// MERGE (:A {val: 5})
	//                           Merge
	//                          /     \
	//                     Filter    Create
	//                      /
	//                Label Scan
	//
	// Complex case:
	// MATCH (a:A) MERGE (a)-[:E]->(:B)
	//                                  Merge
	//                           /        |        \
	//                    LabelScan CondTraverse  Create
	//                                    |          \
	//                                Argument     Argument
	//

	// collect the variables that are bound at this point
	// as MERGE shouldn't construct them
	rax *bound_vars = NULL;
	const char **arguments = NULL;
	if(plan->root) {
		bound_vars = raxNew();
		// rather than cloning the record map
		// collect the bound variables along with their parser-generated
		// constant strings
		ExecutionPlan_BoundVariables(plan->root, bound_vars, plan);

		// collect the variable names from bound_vars to populate the
		// Argument ops we will build
		arguments = (const char **)raxValues(bound_vars);
	}

	// convert all AST data required to populate our operations tree
	AST_MergeContext merge_ctx =
		AST_PrepareMergeOp(clause, gc, plan->query_graph, bound_vars);

	// build the Match stream as a Merge child
	const cypher_astnode_t *path = cypher_ast_merge_get_pattern_path(clause);
	OpBase *match_stream = ExecutionPlan_BuildOpsFromPath(plan, arguments, path);

	// create a Merge operation
	// it will store no information at this time except for any graph updates
	// it should make due to ON MATCH and ON CREATE SET directives in the query
	OpBase *merge_op = NewMergeOp(plan, merge_ctx.on_match, merge_ctx.on_create);

	// in case we're dealing with a merge without a bound brench
	// e.g. MERGE (n:N {v:$v})
	// then introduce an empty record operation to act as the bound branch
	if(plan->root == NULL) {
		ExecutionPlan_UpdateRoot(plan, NewEmptyRecordOp(plan));
	}

	// set Merge op as new root and add previously-built ops, if any,
	// as Merge's first stream

	// build the Create stream as a Merge child
	// if we have bound variables, we must ensure that
	// all of our created entities are unique
	// consider:
	// UNWIND [1, 1] AS x MERGE ({val: x})
	// exactly one node should be created in the UNWIND...MERGE query
	OpBase *create_stream = NewMergeCreateOp(plan, merge_ctx.nodes_to_merge,
			merge_ctx.edges_to_merge);

	//--------------------------------------------------------------------------
	// introduce child ops to MERGE
	//--------------------------------------------------------------------------

	ExecutionPlan_UpdateRoot(plan, merge_op);
	ExecutionPlan_AddOp(merge_op, match_stream);
	ExecutionPlan_AddOp(merge_op, create_stream);

	if(bound_vars) raxFree(bound_vars);
	array_free(arguments);
}

