/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "../execution_plan.h"
#include "../../ast/ast.h"
#include "../ops/op_filter.h"

// Build a Skip operation from SKIP clause
OpBase *buildSkipOp(ExecutionPlan *plan, const cypher_astnode_t *skip);

// Build a Limit operation from LIMIT clause
OpBase *buildLimitOp(ExecutionPlan *plan, const cypher_astnode_t *limit);

// Build a procedure call operation from CALL clause
void buildCallOp(AST *ast, ExecutionPlan *plan, const cypher_astnode_t *call_clause);

// Convert a MATCH clause into a sequence of scan and traverse ops
void buildMatchOpTree(ExecutionPlan *plan, AST *ast, const cypher_astnode_t *clause);

// Convert a RETURN clause into a Project or Aggregate op
void buildReturnOps(ExecutionPlan *plan, const cypher_astnode_t *clause);

// Convert a WITH clause into a Project or Aggregate op
void buildWithOps(ExecutionPlan *plan, const cypher_astnode_t *clause);

// Convert a MERGE clause into a matching traversal and creation op tree
void buildMergeOp(ExecutionPlan *plan, AST *ast, const cypher_astnode_t *clause, GraphContext *gc);

// Reduce a filter operation into an apply operation
void ExecutionPlan_ReduceFilterToApply(ExecutionPlan *plan, OpFilter *filter);

// Place filter ops at the appropriate positions within the op tree
void ExecutionPlan_PlaceFilterOps(ExecutionPlan *plan, OpBase *root, const OpBase *recurse_limit,
								  FT_FilterNode *ft);

// Convert a clause into the appropriate sequence of ops
void ExecutionPlanSegment_ConvertClause(GraphContext *gc, AST *ast, ExecutionPlan *plan,
										const cypher_astnode_t *clause);

// Build pattern comprehension plan operations
void buildPatternComprehensionOps(
	ExecutionPlan *plan,
	OpBase *root,
	const cypher_astnode_t *ast
);

// Build pattern path plan operations
void buildPatternPathOps(
	ExecutionPlan *plan,
	OpBase *root,
	const cypher_astnode_t *ast
);

// given an AST path pattern, generate the tree of scan, traverse,
// and filter operations required to represent it.
OpBase *ExecutionPlan_BuildOpsFromPath
(
	ExecutionPlan *plan,
	const char **vars,
	const cypher_astnode_t *path
);
