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

#define OP_REQUIRE_NEW_DATA(opRes) (opRes & (OP_DEPLETED | OP_REFRESH)) > 0

#define OP_JOIN_MULTIPLE_STREAMS(op)                     \
	(OpBase_Type((op)) == OPType_JOIN            ||      \
	 OpBase_Type((op)) == OPType_VALUE_HASH_JOIN ||      \
	 OpBase_Type((op)) == OPType_CARTESIAN_PRODUCT)

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
	OPType_OPTIONAL_CONDITIONAL_TRAVERSE,
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
	OPType_LOAD_CSV,
	OPType_SUBQUERY_FOREACH,
	OPType_EMPTY_ROW,
	OPType_EAGER
} OPType;

typedef enum {
	OP_DEPLETED = 1,
	OP_REFRESH = 2,
	OP_OK = 4,
	OP_ERR = 8,
} OpResult;

// Macro for checking whether an operation is an Apply variant.
#define OP_IS_APPLY(op) ((op)->type == OPType_OR_APPLY_MULTIPLEXER || (op)->type == OPType_AND_APPLY_MULTIPLEXER || (op)->type == OPType_SEMI_APPLY || (op)->type == OPType_ANTI_SEMI_APPLY)

#define PROJECT_OP_COUNT 2
static const OPType PROJECT_OPS[] = {
	OPType_PROJECT,
	OPType_AGGREGATE
};

// TODO: add OPType_OPTIONAL_CONDITIONAL_TRAVERSE to TRAVERSE_OPS
// and adjust relevent optimizations accordingly
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
	OPType_APPLY,
	OPType_MERGE
};

#define EAGER_OP_COUNT 8
static const OPType EAGER_OPERATIONS[] = {
	OPType_AGGREGATE,
	OPType_CREATE,
	OPType_DELETE,
	OPType_UPDATE,
	OPType_MERGE,
	OPType_FOREACH,
	OPType_SORT,
	OPType_EAGER
};

#define MODIFYING_OP_COUNT 4
static const OPType MODIFYING_OPERATIONS[] = {
	OPType_CREATE,
	OPType_DELETE,
	OPType_UPDATE,
	OPType_MERGE
};

struct OpBase;
struct ExecutionPlan;

typedef void (*fpFree)(struct OpBase *);
typedef OpResult(*fpInit)(struct OpBase *);
typedef Record(*fpConsume)(struct OpBase *);
typedef OpResult(*fpReset)(struct OpBase *);
typedef void (*fpToString)(const struct OpBase *, sds *);
typedef struct OpBase *(*fpClone)(const struct ExecutionPlan *, const struct OpBase *);

// Execution plan operation statistics.
typedef struct {
	int profileRecordCount;     // Number of records generated.
	double profileExecTime;     // Operation total execution time in ms.
}  OpStats;

struct OpBase {
	OPType type;                       // type of operation
	fpInit init;                       // called once before execution
	fpFree free;                       // free operation
	fpReset reset;                     // reset operation state
	fpClone clone;                     // operation clone
	fpConsume consume;                 // produce next record
	fpConsume _consume;                // backup for the original consume func
	fpToString toString;               // operation string representation
	const char *name;                  // operation name
	int childCount;                    // number of children
	struct OpBase **children;          // child operations
	const char **modifies;             // list of entities this op modifies
	dict *awareness;                   // variables this op is aware of
	OpStats *stats;                    // profiling statistics
	struct OpBase *parent;             // parent operations
	const struct ExecutionPlan *plan;  // executionPlan this operation is part of
	bool writer;                       // indicates this is a writer operation
};
typedef struct OpBase OpBase;

// initialize op
void OpBase_Init
(
	OpBase *op,
	OPType type,
	const char *name,
	fpInit init,
	fpConsume consume,
	fpReset reset,
	fpToString toString,
	fpClone,
	fpFree free,
	bool writer,
	const struct ExecutionPlan *plan
);

// free op
void OpBase_Free
(
	OpBase *op
);

// consume op
Record OpBase_Consume
(
	OpBase *op
);

// profile op
Record OpBase_Profile
(
	OpBase *op
);

void OpBase_ToString
(
	const OpBase *op,
	sds *buff
);

OpBase *OpBase_Clone
(
	const struct ExecutionPlan *plan,
	const OpBase *op
);

// returns operation type
OPType OpBase_Type
(
	const OpBase *op
);

// returns the number of children of the op
uint OpBase_ChildCount
(
	const OpBase *op
);

// returns the i'th child of the op
OpBase *OpBase_GetChild
(
	const OpBase *op,  // op
	uint i             // child index
);

// returns op's parent
OpBase *OpBase_Parent
(
	const OpBase *op  // op
);

// returns true if operation is aware of all aliases
bool OpBase_Aware
(
	const OpBase *op,      // op
	const char **aliases,  // aliases
	uint n                 // number of aliases
);

// mark alias as being modified by operation
// returns the ID associated with alias
int OpBase_Modifies
(
	OpBase *op,
	const char *alias
);

// adds an alias to an existing modifier
// such that record[modifier] = record[alias]
int OpBase_AliasModifier
(
	OpBase *op,            // op
	const char *modifier,  // existing alias
	const char *alias      // new alias
);

// returns true if any of an op's children are aware of the given alias
bool OpBase_ChildrenAware
(
	const OpBase *op,
	const char *alias,
	int *idx
);

// returns true if alias is mapped
bool OpBase_AliasMapping
(
	const OpBase *op,   // op
	const char *alias,  // alias
	int *idx            // alias map id
);

// sends reset request to each operation up the chain
void OpBase_PropagateReset
(
	OpBase *op
);

// indicates if the operation is a writer operation
bool OpBase_IsWriter
(
	const OpBase *op
);

// indicates if the operation is an eager operation
bool OpBase_IsEager
(
	const OpBase *op
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
	const struct ExecutionPlan *plan
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

// release record
void OpBase_DeleteRecord
(
	Record *r
);

// merge src into dest and deletes src
void OpBase_MergeRecords
(
	Record dest,  // entries are merged into this record
	Record *src   // entries are merged from this record
);

