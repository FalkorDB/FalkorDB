/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "../ops/ops.h"
#include "../../query_ctx.h"
#include "../execution_plan.h"
#include "execution_plan_util.h"
#include "execution_plan_modify.h"
#include "execution_plan_construct.h"
#include "../../util/rax_extensions.h"
#include "../../ast/ast_build_op_contexts.h"

static void _buildMergeCreateStream(ExecutionPlan *plan, AST_MergeContext *merge_ctx,
									const char **arguments) {
	/* If we have bound variables, we must ensure that all of our created entities are unique. Consider:
	 * UNWIND [1, 1] AS x MERGE ({val: x})
	 * Exactly one node should be created in the UNWIND...MERGE query. */
	OpBase *merge_create = NewMergeCreateOp(plan, merge_ctx->nodes_to_merge, merge_ctx->edges_to_merge);
	ExecutionPlan_AddOp(plan->root, merge_create); // Add MergeCreate op to stream.

	// If we have bound variables, push an Argument tap beneath the Create op.
	if(arguments) {
		OpBase *create_argument = NewArgumentOp(plan, arguments);
		ExecutionPlan_AddOp(merge_create, create_argument); // Add Argument op to stream.
	}
}

void buildMergeOp(ExecutionPlan *plan, AST *ast, const cypher_astnode_t *clause, GraphContext *gc) {
	/*
	 * A MERGE clause provides a single path that must exist or be created.
	 * If we have built ops already, they will form the first stream into the Merge op.
	 * A clone of the Record produced by this stream will be passed into the other Merge streams
	 * so that they properly work with bound variables.
	 *
	 * As with paths in a MATCH query, build the appropriate traversal operations
	 * and add them as another stream into Merge.
	 *
	 * Finally, we'll add a last stream that creates the pattern if it did not get matched.
	 *
	 * Simple case (2 streams, no bound variables):
	 * MERGE (:A {val: 5})
	 *                           Merge
	 *                          /     \
	 *                     Filter    Create
	 *                      /
	 *                Label Scan
	 *
	 * Complex case:
	 * MATCH (a:A) MERGE (a)-[:E]->(:B)
	 *                                  Merge
	 *                           /        |        \
	 *                    LabelScan CondTraverse  Create
	 *                                    |          \
	 *                                Argument     Argument
	 */

	// Collect the variables that are bound at this point, as MERGE shouldn't construct them.
	rax *bound_vars = NULL;
	const char **arguments = NULL;
	if(plan->root) {
		bound_vars = raxNew();
		// Rather than cloning the record map, collect the bound variables along with their
		// parser-generated constant strings.
		ExecutionPlan_BoundVariables(plan->root, bound_vars, plan);
		// Collect the variable names from bound_vars to populate the Argument ops we will build.
		arguments = (const char **)raxValues(bound_vars);
	}

	// Convert all AST data required to populate our operations tree.
	AST_MergeContext merge_ctx = AST_PrepareMergeOp(clause, gc, plan->query_graph, bound_vars);

	// Build the Match stream as a Merge child.
	const cypher_astnode_t *path = cypher_ast_merge_get_pattern_path(clause);
	OpBase *match_stream = ExecutionPlan_BuildOpsFromPath(plan, arguments, path);

	// Create a Merge operation. It will store no information at this time except for any graph updates
	// it should make due to ON MATCH and ON CREATE SET directives in the query.
	OpBase *merge_op = NewMergeOp(plan, merge_ctx.on_match, merge_ctx.on_create);
	// Set Merge op as new root and add previously-built ops, if any, as Merge's first stream.
	ExecutionPlan_UpdateRoot(plan, merge_op);
	ExecutionPlan_AddOp(merge_op, match_stream); // Add Match stream to Merge op.

	// Build the Create stream as a Merge child.
	_buildMergeCreateStream(plan, &merge_ctx, arguments);

	if(bound_vars) raxFree(bound_vars);
	array_free(arguments);
}

