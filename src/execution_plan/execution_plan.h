/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "./ops/op.h"
#include "../graph/graph.h"
#include "../resultset/resultset.h"
#include "../filter_tree/filter_tree.h"
#include "../util/object_pool/object_pool.h"

typedef struct ExecutionPlan ExecutionPlan;

struct ExecutionPlan {
	OpBase *root;             // root operation of overall ExecutionPlan
	AST *ast_segment;         // the segment which the current ExecutionPlan segment is built from
	rax *record_map;          // mapping between identifiers and record indices
	dict *resolvers;          // mapping between identifiers and resolving op
	QueryGraph *query_graph;  // queryGraph representing all graph entities in this segment
	ObjectPool *record_pool;  // pool of reusable records
	bool prepared;            // indicates if the execution plan is ready for execute
};

// creates a new execution plan from AST
ExecutionPlan *ExecutionPlan_FromTLS_AST(void);

// allocate a new ExecutionPlan segment
ExecutionPlan *ExecutionPlan_NewEmptyExecutionPlan(void);

// prepare an execution plan for execution: optimize
// initialize result set schema
void ExecutionPlan_PreparePlan
(
	ExecutionPlan *plan
);

// clones an execution plan
ExecutionPlan *ExecutionPlan_Clone
(
	const ExecutionPlan *plan
);

// build a tree of operations that performs all the work required by the clauses
// in the current AST
void ExecutionPlan_PopulateExecutionPlan
(
	ExecutionPlan *plan
);

// re-position filter op
void ExecutionPlan_RePositionFilterOp
(
	ExecutionPlan *plan,
	OpBase *lower_bound,
	const OpBase *upper_bound,
	OpBase *filter
);

// retrieve the map of aliases to Record offsets in this ExecutionPlan segment
rax *ExecutionPlan_GetMappings
(
	const ExecutionPlan *plan
);

// add a new mappings from an alias to a Record ID
// returns the ID of the new mapping
int ExecutionPlan_AddMappings
(
	const ExecutionPlan *plan,  // plan to update
	const char *alias           // alias to map
);

// mark 'op' as a resolver for 'alias'
void ExecutionPlan_AddResolver
(
	ExecutionPlan *plan,  // plan to update
	const char *alias,    // resolved alias
	const OpBase *op      // resolver operation
);

// remove op from plan's resolver map
void ExecutionPlan_RemoveResolver
(
	ExecutionPlan *plan,  // plan to update
	const OpBase *op      // resolver operation
);

// retrieves a Record from the ExecutionPlan's Record pool
Record ExecutionPlan_BorrowRecord
(
	ExecutionPlan *plan
);

// free Record contents and return it to the Record pool
void ExecutionPlan_ReturnRecord
(
	const ExecutionPlan *plan,
	Record r
);

// prints execution plan
void ExecutionPlan_Print
(
	const ExecutionPlan *plan,
	RedisModuleCtx *ctx
);

// initialize all operations in an ExecutionPlan
void ExecutionPlan_Init
(
	ExecutionPlan *plan
);

// executes plan
ResultSet *ExecutionPlan_Execute
(
	ExecutionPlan *plan
);

// drains execution plan
void ExecutionPlan_Drain
(
	ExecutionPlan *plan
);

// checks if execution plan been drained
bool ExecutionPlan_Drained
(
	ExecutionPlan *plan
);

// profile executes plan
ResultSet *ExecutionPlan_Profile
(
	ExecutionPlan *plan
);

// free execution plan
void ExecutionPlan_Free
(
	ExecutionPlan *plan
);

