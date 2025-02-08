/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../execution_plan.h"
#include "../../ast/ast.h"
#include "../ops/op_filter.h"

// build a Skip operation from SKIP clause
OpBase *buildSkipOp
(
	ExecutionPlan *plan,
	const cypher_astnode_t *skip
);

// build a Limit operation from LIMIT clause
OpBase *buildLimitOp
(
	ExecutionPlan *plan,
	const cypher_astnode_t *limit
);

// build a procedure call operation from CALL clause
void buildCallOp
(
	AST *ast,
	ExecutionPlan *plan,
	const cypher_astnode_t *call_clause
);

// convert a MATCH clause into a sequence of scan and traverse ops
void buildMatchOpTree
(
	ExecutionPlan *plan,
	AST *ast,
	const cypher_astnode_t *clause
);

// convert a RETURN clause into a Project or Aggregate op
void buildReturnOps
(
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
);

// convert a WITH clause into a Project or Aggregate op
void buildWithOps
(
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
);

// convert a MERGE clause into a matching traversal and creation op tree
void buildMergeOp
(
	ExecutionPlan *plan,
	AST *ast,
	const cypher_astnode_t *clause,
	GraphContext *gc
);

// reduce a filter operation into an apply operation
void ExecutionPlan_ReduceFilterToApply
(
	ExecutionPlan *plan,
	OpFilter *filter
);

// place filter ops at the appropriate positions within the op tree
void ExecutionPlan_PlaceFilterOps
(
	ExecutionPlan *plan,           // plan
	OpBase *root,                  // root
	const OpBase *recurse_limit,   // boundry
	FT_FilterNode *ft              // filter-tree to position
);

// convert a clause into the appropriate sequence of ops
void ExecutionPlanSegment_ConvertClause
(
	GraphContext *gc,
	AST *ast,
	ExecutionPlan *plan,
	const cypher_astnode_t *clause
);

// build pattern comprehension plan operations
void buildPatternComprehensionOps(
	ExecutionPlan *plan,
	OpBase *root,
	const cypher_astnode_t *ast
);

// build pattern path plan operations
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

