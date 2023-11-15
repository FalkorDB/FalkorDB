/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_apply.h"
#include "op_argument.h"

// forward declarations
static OpResult ArgumentInit(OpBase *opBase);
static Record ArgumentConsume(OpBase *opBase);
static OpResult ArgumentReset(OpBase *opBase);
static OpBase *ArgumentClone(const ExecutionPlan *plan, const OpBase *opBase);

OpBase *NewArgumentOp
(
	const ExecutionPlan *plan,
	const char **variables
) {
	Argument *op = rm_calloc(1, sizeof(Argument));

	// set our Op operations
	OpBase_Init((OpBase *)op, OPType_ARGUMENT, "Argument", ArgumentInit,
			ArgumentConsume, ArgumentReset, NULL, ArgumentClone, NULL, false,
			plan);

	uint variable_count = array_len(variables);
	for(uint i = 0; i < variable_count; i ++) {
		OpBase_Modifies((OpBase *)op, variables[i]);
	}

	return (OpBase *)op;
}

static OpResult ArgumentInit
(
	OpBase *opBase
) {
	Argument *arg = (Argument*)opBase;

	ASSERT(arg != NULL);
	ASSERT(arg->producer == NULL);

	//--------------------------------------------------------------------------
	// locate producer
	//--------------------------------------------------------------------------

	// scan down the parent tree looking for an Apply operation
	// which doesn't contains us as its child

	OpBase *parent = opBase->parent;
	while(parent != NULL) {
		OPType t = OpBase_Type(parent);
		if(t == OPType_APPLY           ||
		   t == OPType_SEMI_APPLY      ||
		   t == OPType_ANTI_SEMI_APPLY ||
		   t == OPType_MERGE) {
			// make sure we're not the apply's left child
			if(OpBase_GetChild(parent, 0) != opBase) {
				arg->producer = parent;
				break;
			}
		}

		// advance
		parent = parent->parent;
	}

	// make sure producer was found
	ASSERT(arg->producer != NULL);

	return OP_OK;
}

static Record ArgumentNullConsume
(
	OpBase *opBase
) {
	return NULL;
}

static Record ArgumentConsume
(
	OpBase *opBase
) {
	Argument *arg = (Argument *)opBase;

	// emit the record only once
	// update consume function to a function which always returns NULL
	Record r = Apply_PullArgRecord(arg->producer);
	OpBase_UpdateConsume(opBase, ArgumentNullConsume);
	return r;
}

static OpResult ArgumentReset
(
	OpBase *opBase
) {
	Argument *arg = (Argument *)opBase;
	// reset consume function
	OpBase_UpdateConsume(opBase, ArgumentConsume);
	return OP_OK;
}

static inline OpBase *ArgumentClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_ARGUMENT);
	return NewArgumentOp(plan, opBase->modifies);
}

