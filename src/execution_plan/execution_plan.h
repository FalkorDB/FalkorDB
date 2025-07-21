/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#pragma once

#include "./ops/op.h"
#include "../graph/graph.h"
#include "../resultset/resultset.h"
#include "../filter_tree/filter_tree.h"
#include "../util/object_pool/object_pool.h"

typedef struct ExecutionPlan ExecutionPlan;

struct ExecutionPlan {
	OpBase *root;                       // Root operation of overall ExecutionPlan.
	AST *ast_segment;                   // The segment which the current ExecutionPlan segment is built from.
	rax *record_map;                    // Mapping between identifiers and record indices.
	QueryGraph *query_graph;            // QueryGraph representing all graph entities in this segment.
	ObjectPool *record_pool;
	bool prepared;                      // Indicates if the execution plan is ready for execute.
};

// Creates a new execution plan from AST
ExecutionPlan *ExecutionPlan_FromTLS_AST(void);

// Prepare an execution plan for execution: optimize, initialize result set schema.
void ExecutionPlan_PreparePlan(ExecutionPlan *plan);

// Allocate a new ExecutionPlan segment.
ExecutionPlan *ExecutionPlan_NewEmptyExecutionPlan(void);

// Build a tree of operations that performs all the work required by the clauses of the current AST.
void ExecutionPlan_PopulateExecutionPlan(ExecutionPlan *plan);

// Re position filter op.
void ExecutionPlan_RePositionFilterOp(ExecutionPlan *plan, OpBase *lower_bound,
									  const OpBase *upper_bound, OpBase *filter);

// Retrieve the map of aliases to Record offsets in this ExecutionPlan segment.
rax *ExecutionPlan_GetMappings(const ExecutionPlan *plan);

// Retrieves a Record from the ExecutionPlan's Record pool.
Record ExecutionPlan_BorrowRecord(ExecutionPlan *plan);

// Free Record contents and return it to the Record pool.
void ExecutionPlan_ReturnRecord(const ExecutionPlan *plan, Record r);

// Prints execution plan.
void ExecutionPlan_Print(const ExecutionPlan *plan, RedisModuleCtx *ctx);

// Initialize all operations in an ExecutionPlan.
void ExecutionPlan_Init(ExecutionPlan *plan);

// Executes plan
ResultSet *ExecutionPlan_Execute(ExecutionPlan *plan);

// Checks if execution plan been drained
bool ExecutionPlan_Drained(ExecutionPlan *plan);

// Drains execution plan
void ExecutionPlan_Drain(ExecutionPlan *plan);

// Profile executes plan
ResultSet *ExecutionPlan_Profile(ExecutionPlan *plan);

// Free execution plan
void ExecutionPlan_Free(ExecutionPlan *plan);
