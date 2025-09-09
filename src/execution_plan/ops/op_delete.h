/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../execution_plan.h"
#include "../../graph/entities/node.h"
#include "../../resultset/resultset_statistics.h"
#include "cypher-parser.h"
#include "rax.h"

// deletes entities specified within the DELETE clause

typedef struct {
	OpBase op;
	GraphContext *gc;
	uint64_t rec_idx;                      // index of record to emit
	Record *records;                       // eagerly collected records
	AR_ExpNode **exps;                     // expressions evaluated to an entity about to be deleted
	uint exp_count;                        // number of expressions
	cypher_ast_delete_mode_t delete_mode;  // delete mode: DELETE, DETACH, or NODETACH
	Node *deleted_nodes;                   // array of nodes to be removed
	Edge *deleted_edges;                   // array of edges to be removed
} OpDelete;

OpBase *NewDeleteOp
(
	const ExecutionPlan *plan,
	AR_ExpNode **exps,
	cypher_ast_delete_mode_t delete_mode
);

