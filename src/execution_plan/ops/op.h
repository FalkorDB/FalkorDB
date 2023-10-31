/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "../record.h"
#include "../../util/arr.h"
#include "../../util/dict.h"
#include "../../redismodule.h"
#include "../../schema/schema.h"
#include "../../graph/query_graph.h"
#include "../../graph/entities/node.h"
#include "../../graph/entities/edge.h"

typedef enum {
	OPType_ALL_NODE_SCAN,
	OPType_NODE_BY_LABEL_SCAN,
	OPType_NODE_BY_INDEX_SCAN,
	OPType_EDGE_BY_INDEX_SCAN,
	OPType_NODE_BY_ID_SEEK,
	OPType_NODE_BY_LABEL_AND_ID_SCAN,
	OPType_EXPAND_INTO,
	OPType_CONDITIONAL_TRAVERSE,
	OPType_CONDITIONAL_VAR_LEN_TRAVERSE,
	OPType_CONDITIONAL_VAR_LEN_TRAVERSE_EXPAND_INTO,
	OPType_RESULTS,
	OPType_PROJECT,
	OPType_AGGREGATE,
	OPType_SORT,
	OPType_SKIP,
	OPType_LIMIT,
	OPType_DISTINCT,
	OPType_MERGE,
	OPType_MERGE_CREATE,
	OPType_FILTER,
	OPType_CREATE,
	OPType_UPDATE,
	OPType_DELETE,
	OPType_UNWIND,
	OPType_FOREACH,
	OPType_PROC_CALL,
	OPType_CALLSUBQUERY,
	OPType_ARGUMENT,
	OPType_ARGUMENT_LIST,
	OPType_CARTESIAN_PRODUCT,
	OPType_VALUE_HASH_JOIN,
	OPType_APPLY,
	OPType_JOIN,
	OPType_SEMI_APPLY,
	OPType_ANTI_SEMI_APPLY,
	OPType_OR_APPLY_MULTIPLEXER,
	OPType_AND_APPLY_MULTIPLEXER,
	OPType_OPTIONAL,
} OPType;

typedef enum {
	OP_DEPLETED = 1,
	OP_REFRESH = 2,
	OP_OK = 4,
	OP_ERR = 8,
} OpResult;

// macro for checking whether an operation is an Apply variant
#define OP_IS_APPLY(op) ((op)->type == OPType_OR_APPLY_MULTIPLEXER || (op)->type == OPType_AND_APPLY_MULTIPLEXER || (op)->type == OPType_SEMI_APPLY || (op)->type == OPType_ANTI_SEMI_APPLY)

#define PROJECT_OP_COUNT 2
static const OPType PROJECT_OPS[] = {
	OPType_PROJECT,
	OPType_AGGREGATE
};

#define TRAVERSE_OP_COUNT 2
static const OPType TRAVERSE_OPS[] = {
	OPType_CONDITIONAL_TRAVERSE,
	OPType_CONDITIONAL_VAR_LEN_TRAVERSE
};

#define SCAN_OP_COUNT 5
static const OPType SCAN_OPS[] = {
	OPType_ALL_NODE_SCAN,
	OPType_NODE_BY_LABEL_SCAN,
	OPType_NODE_BY_INDEX_SCAN,
	OPType_EDGE_BY_INDEX_SCAN,
	OPType_NODE_BY_ID_SEEK,
	OPType_NODE_BY_LABEL_AND_ID_SCAN
};

#define BLACKLIST_OP_COUNT 2
static const OPType FILTER_RECURSE_BLACKLIST[] = {
	//OPType_APPLY,
	OPType_OPTIONAL,
	OPType_MERGE
};

#define EAGER_OP_COUNT 7
static const OPType EAGER_OPERATIONS[] = {
	OPType_AGGREGATE,
	OPType_CREATE,
	OPType_DELETE,
	OPType_UPDATE,
	OPType_MERGE,
	OPType_FOREACH,
	OPType_SORT
};

struct OpBase;
struct ExecutionPlan;

// operation function pointers
typedef void (*fpFree)(struct OpBase *);      // free operation
typedef OpResult(*fpInit)(struct OpBase *);   // initialize operation
typedef Record(*fpConsume)(struct OpBase *);  // consume operation
typedef OpResult(*fpReset)(struct OpBase *);  // reset operation
typedef void (*fpToString)(const struct OpBase *, sds *);
typedef struct OpBase *(*fpClone)(struct ExecutionPlan *, const struct OpBase *);

// execution plan operation statistics
typedef struct {
	int profileRecordCount;  // number of records generated
	double profileExecTime;  // operation total execution time in ms
}  OpStats;

// operation base
struct OpBase {
	OPType type;                // type of operation
	fpInit init;                // called once before execution
	fpFree free;                // free operation
	fpReset reset;              // reset operation state
	fpClone clone;              // operation clone
	fpConsume consume;          // produce next record
	fpConsume profile;          // profiled version of consume
	fpToString toString;        // operation string representation
	const char *name;           // operation name
	int childCount;             // number of children
	struct OpBase **children;   // child operations
	const char **modifies;      // list of entities this op modifies
	OpStats *stats;             // profiling statistics
	struct OpBase *parent;      // parent operations
	bool writer;                // indicates this is a writer operation
	struct ExecutionPlan *plan; // executionPlan this operation is part of
	dict *aware;                // identifiers available to this operation
};
typedef struct OpBase OpBase;

// initialize op
void OpBase_Init
(
	OpBase *op,                 // op to initialize
	OPType type,                // op type
	const char *name,           // op name
	fpInit init,                // op's init function
	fpConsume consume,          // op's consume function
	fpReset reset,              // op's reset function
	fpToString toString,        // op's toString function
	fpClone clone,              // op's clone function
	fpFree free,                // op's free function
	bool writer,  			    // writer indicator
	struct ExecutionPlan *plan  // op's execution plan
);

// consume op
Record OpBase_Consume
(
	OpBase *op  // operation to consume
);

// profile op
Record OpBase_Profile
(
	OpBase *op  // operation to profile
);

void OpBase_ToString
(
	const OpBase *op,
	sds *buff
);

// clone opertion
OpBase *OpBase_Clone
(
	struct ExecutionPlan *plan,
	const OpBase *op
);

// returns operation type
OPType OpBase_Type
(
	const OpBase *op
);

void OpBase_SetParent
(
	OpBase *op,     // op to set parent for
	OpBase *parent  // parent to set
);

// add child
void OpBase_AddChild
(
	OpBase *parent,  // parent op
	OpBase *child    // new child op
);

// add child at specific index
void OpBase_AddChildAt
(
	OpBase *parent,  // parent op
	OpBase *child,   // child op
	uint idx         // index of op
);

// remove child from parent
void OpBase_RemoveChild
(
	OpBase *op,            // parent op
	OpBase *child,         // child to remove
	bool inharit_children  // if true, child's children will be inharited
);

// locate child in parent's children array
// returns true if child was found, false otherwise
// sets 'idx' to the index of child in parent's children array
bool OpBase_LocateChild
(
	const OpBase *parent,  // parent op
	const OpBase *child,   // child op to locate
	int *idx               // [optional out] index of child in parent
);

// returns op's number of children
uint OpBase_ChildCount
(
	const OpBase *op
);

// returns op's i'th child
OpBase *OpBase_GetChild
(
	OpBase *join,  // op
	uint i         // child index
);

// mark alias as being modified by operation
// returns the ID associated with alias
int OpBase_Modifies
(
	OpBase *op,
	const char *alias
);

// returns op's modifiers
const char **OpBase_GetModifiers
(
	const OpBase *op,  // op to get modifiers from
	int *n             // number of modifiers
);

// returns true if any of an op's children are aware of the given alias
bool OpBase_ChildrenAware
(
	OpBase *op,
	const char *alias,
	int *idx
);

// returns true if op is aware of alias
// an operation is aware of all aliases it modifies and all aliases
// modified by prior operation within its segment
bool OpBase_Aware
(
	OpBase *op,
	const char *alias,
	int *idx
);

// computes op awareness
// an op is aware of all alises its children are aware of in addition to
// aliases it modifies
void OpBase_ComputeAwareness
(
	OpBase *op  // op to compute awareness for
);

// sends reset request to each operation up the chain
void OpBase_PropagateReset
(
	OpBase *op
);

// indicates if the operation is a writer operation
bool OpBase_IsWriter
(
	OpBase *op
);

// update operation consume function
void OpBase_UpdateConsume
(
	OpBase *op,
	fpConsume consume
);

// updates the plan of an operation
void OpBase_BindOpToPlan
(
	OpBase *op,
	struct ExecutionPlan *plan
);

// creates a new record that will be populated during execution
Record OpBase_CreateRecord
(
	const OpBase *op
);

// clones given record
Record OpBase_CloneRecord
(
	Record r
);

// deep clones given record
Record OpBase_DeepCloneRecord
(
	Record r
);

// release record
void OpBase_DeleteRecord
(
	Record r
);

// free op
void OpBase_Free
(
	OpBase *op  // operation to free
);

