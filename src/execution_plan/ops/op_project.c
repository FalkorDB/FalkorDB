/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_project.h"
#include "../../util/arr.h"
#include "../../query_ctx.h"
#include "../../util/rmalloc.h"
#include "../../../deps/rax/rax.h"
#include "../../util/rax_extensions.h"

// forward declarations
static RecordBatch ProjectConsume(OpBase *opBase);
static OpResult ProjectReset(OpBase *opBase);
static OpBase *ProjectClone(const ExecutionPlan *plan, const OpBase *opBase);
static void ProjectFree(OpBase *opBase);

// create a new projection operation
OpBase *NewProjectOp
(
	const ExecutionPlan *plan,  // op's plan
	AR_ExpNode **exps           // projection expression
) {
	OpProject *op = rm_calloc (1, sizeof (OpProject)) ;

	op->exps           = exps ;
	op->exp_count      = array_len (exps) ;
	op->record_offsets = array_new (uint, op->exp_count) ;

	// set our Op operations
	OpBase_Init ((OpBase *)op, OPType_PROJECT, "Project", NULL, ProjectConsume,
				ProjectReset, NULL, ProjectClone, ProjectFree, false, plan) ;

	for(uint i = 0; i < op->exp_count; i ++) {
		// the projected record will associate values with their resolved name
		// to ensure that space is allocated for each entry
		int record_idx = OpBase_Modifies ((OpBase *)op,
				op->exps[i]->resolved_name) ;
		array_append (op->record_offsets, record_idx) ;
	}

	return (OpBase *)op;
}

static RecordBatch ProjectConsume
(
	OpBase *opBase
) {
	OpProject *op = (OpProject *)opBase ;
	ASSERT (op->batch == NULL) ;

	if (op->op.childCount) {
		OpBase *child = op->op.children[0] ;
		op->batch = OpBase_Consume (child) ;
		if (op->batch == NULL) {
			return NULL ;
		}
	} else {
		// QUERY: RETURN 1+2
		// Return a single record followed by NULL on the second call.
		if (op->singleResponse) {
			return NULL ;
		}

		op->singleResponse = true ;
		op->batch = OpBase_CreateRecordBatch (opBase, 1) ;
	}

	ASSERT (op->batch != NULL) ;

	// allocate projected batch
	uint16_t batch_size = RecordBatch_Size (op->batch) ;
	op->projection = OpBase_CreateRecordBatch (opBase, batch_size) ;

	for (uint i = 0 ; i < batch_size; i++) {
		Record input  = op->batch[i] ;
		Record output = op->projection[i] ;

		for (uint j = 0; j < op->exp_count; j++) {
			AR_ExpNode *exp = op->exps[j] ;
			SIValue v = AR_EXP_Evaluate (exp, input) ;

			// persisting a value is only necessary when
			// 'v' refers to a scalar held in Record 'r'
			// graph entities don't need to be persisted here as
			// Record_Add will copy them internally
			//
			// the RETURN projection here requires persistence:
			// MATCH (a) WITH toUpper(a.name) AS e RETURN e
			// TODO: this is a rare case;
			// the logic of when to persist can be improved
			if (!(v.type & SI_GRAPHENTITY)) {
				SIValue_Persist (&v) ;
			}

			int rec_idx = op->record_offsets[j] ;
			Record_Add (output, rec_idx, v) ;

			// if the value was a graph entity with its own allocation
			// as with a query like:
			// MATCH p = (src) RETURN nodes(p)[0]
			// ensure that the allocation is freed here
			if ((v.type & SI_GRAPHENTITY)) {
				SIValue_Free (v) ;
			}
		}
	}

	// release input batch
	RecordBatch_Free (&op->batch) ;

	// emit the projected batch
	RecordBatch projection = op->projection ;
	op->projection = NULL ;
	return projection ;
}

static OpResult ProjectReset
(
	OpBase *opBase
) {
	OpProject *op = (OpProject *)opBase;
	op->singleResponse = false;

	if (op->batch != NULL) {
		RecordBatch_Free (&op->batch) ;
	}

	if (op->projection != NULL) {
		RecordBatch_Free (&op->projection) ;
	}

	return OP_OK;
}

static OpBase *ProjectClone
(
	const ExecutionPlan *plan,
	const OpBase *opBase
) {
	ASSERT(opBase->type == OPType_PROJECT);
	OpProject *op = (OpProject *)opBase;
	AR_ExpNode **exps;
	array_clone_with_cb(exps, op->exps, AR_EXP_Clone);
	return NewProjectOp (plan, exps) ;
}

void ProjectBindToPlan
(
	OpBase *opBase,            // op to bind
	const ExecutionPlan *plan  // plan to bind the op to
) {
	OpProject *op = (OpProject *)opBase;
	opBase->plan = plan;

	// introduce the projected aliases to the plan record-mapping, and reset the
	// record offsets to the correct indexes
	array_clear(op->record_offsets);

	for(uint i = 0; i < op->exp_count; i ++) {
		// The projected record will associate values with their resolved name
		// to ensure that space is allocated for each entry.
		int record_idx = OpBase_Modifies((OpBase *)op, op->exps[i]->resolved_name);
		array_append(op->record_offsets, record_idx);
	}
}

static void ProjectFree
(
	OpBase *ctx
) {
	OpProject *op = (OpProject *)ctx ;

	if (op->exps != NULL) {
		for (uint i = 0; i < op->exp_count; i++) {
			AR_EXP_Free (op->exps[i]) ;
		}
		array_free (op->exps) ;
		op->exps = NULL ;
	}

	if (op->record_offsets != NULL) {
		array_free (op->record_offsets) ;
		op->record_offsets = NULL ;
	}

	if (op->batch != NULL) {
		RecordBatch_Free (&op->batch) ;
	}

	if (op->projection != NULL) {
		RecordBatch_Free (&op->projection) ;
	}
}

