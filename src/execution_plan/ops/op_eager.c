/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_eager.h"

// forward declarations
static Record EagerConsume(OpBase *opBase);
static OpResult EagerReset(OpBase *opBase);
static void EagerFree(OpBase *opBase);
static OpBase *EagerClone(const ExecutionPlan *plan, const OpBase *opBase);

OpBase *NewEagerOp
(
	const ExecutionPlan *plan
) {
	// validate inputs
	ASSERT (plan != NULL) ;

	OpEager *op = rm_calloc (1, sizeof (OpEager)) ;

	// set operations
	OpBase_Init((OpBase *)op, OPType_EAGER, "Eager", NULL, EagerConsume,
			EagerReset, NULL, EagerClone, EagerFree, false, plan);

	return (OpBase *)op;
}

static Record EagerConsume
(
	OpBase *opBase
) {
	OpEager *op = (OpEager *)opBase;

handoff:
	if (op->records != NULL) {
		if (op->rec_idx < array_len (op->records)) {
			return op->records[op->rec_idx++] ;
		}

		return NULL ;
	}

	ASSERT (OpBase_ChildCount (opBase) == 1) ;

	op->records = array_new (Record, 1) ;
	OpBase *child = OpBase_GetChild (opBase, 0) ;

	// drain stream
	Record r = NULL ;
	while ((r = OpBase_Consume (child))) {
		array_append (op->records, r) ;
	}

	OpBase_PropagateReset (child) ;

	goto handoff ;
}

static OpResult EagerReset
(
	OpBase *ctx
) {
	OpEager *op = (OpEager *)ctx;
	
	if (op->records != NULL) {
		uint n = array_len (op->records) ;
		for (uint i = op->rec_idx; i < n; i++) {
			OpBase_DeleteRecord (op->records + i) ;
		}

		array_free (op->records) ;
		op->records = NULL ;
	}

	op->rec_idx = 0 ;

	return OP_OK;
}

static inline OpBase *EagerClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT (opBase->type == OPType_EAGER) ;

	return NewEagerOp (plan) ;
}

static void EagerFree
(
	OpBase *opBase
) {
	OpEager *op = (OpEager *)opBase;

	if (op->records != NULL) {
		uint n = array_len (op->records) ;
		for (uint i = op->rec_idx; i < n; i++) {
			OpBase_DeleteRecord (op->records + i) ;
		}

		array_free (op->records) ;
		op->records = NULL ;
	}
}

