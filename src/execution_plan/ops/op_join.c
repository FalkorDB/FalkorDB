/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

// the OpJoin operation joins multiple sub queries
// records pass through as is
// all records must have the same structure
//
// MATCH (n)-[:R]->(m)
// RETURN n.v AS value
// SORT BY n.k
//
// UNION
//
// MATCH (n)-[:S]->(m)->[]->()
// RETURN m.v + n.v AS value

#include "RG.h"
#include "op_join.h"
#include "../../query_ctx.h"

// forward declarations
static Record JoinConsume(OpBase *opBase);
static OpResult JoinInit(OpBase *opBase);
static OpBase *JoinClone(const ExecutionPlan *plan, const OpBase *opBase);
static OpResult JoinReset(OpBase *opBase);

OpBase *NewJoinOp
(
	const ExecutionPlan *plan
) {
	OpJoin *op = rm_calloc(1, sizeof(OpJoin));

	op->stream    = NULL;
	op->streamIdx = 0;

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_JOIN, "Join", JoinInit, JoinConsume, 
		JoinReset, NULL, JoinClone, NULL, false, plan);

	return (OpBase *)op;
}

static OpResult JoinInit
(
	OpBase *opBase
) {
	OpJoin *op = (OpJoin *)opBase;

	// start pulling from first stream
	op->streamIdx = 0;
	op->stream    = OpBase_GetChild(opBase, op->streamIdx);

	return OP_OK;
}

static Record JoinConsume
(
	OpBase *opBase
) {
	OpJoin *op = (OpJoin*)opBase;

	Record r = NULL;

	while(!(r = OpBase_Consume(op->stream))) {
		// stream depleted
		// propagate reset to release RediSearch index lock if any exists
		OpBase_PropagateReset(op->stream);

		// try moving to next stream
		op->streamIdx++;
		if(op->streamIdx >= OpBase_ChildCount(opBase)) {
			// no more streams to pull from, depleted!
			break;
		}

		// update stream
		op->stream = op->op.children[op->streamIdx];
	}

	return r;
}

static inline OpBase *JoinClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_JOIN);
	return NewJoinOp(plan);
}

static OpResult JoinReset
(
	OpBase *opBase
) {
	OpJoin *op = (OpJoin*)opBase;

	op->streamIdx = 0;
	op->stream    = OpBase_GetChild(opBase, op->streamIdx);

	return OP_OK;
}

