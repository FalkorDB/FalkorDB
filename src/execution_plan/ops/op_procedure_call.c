/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "op_procedure_call.h"
#include "../../util/arr.h"
#include "../../util/rmalloc.h"
#include "../../query_ctx.h"

// forward declarations
static Record ProcCallConsume(OpBase *opBase);
static OpResult ProcCallReset(OpBase *opBase);
static OpBase *ProcCallClone(const ExecutionPlan *plan, const OpBase *opBase);
static void ProcCallFree(OpBase *opBase);

static Record _yield(OpProcCall *op) {
	SIValue *outputs = Proc_Step(op->procedure);
	if(outputs == NULL) return NULL;

	Record clone = OpBase_CloneRecord(op->r);
	for(uint i = 0; i < array_len(op->output); i++) {
		int idx = op->yield_map[i].rec_idx;
		uint proc_out_idx = op->yield_map[i].proc_out_idx;
		SIValue val = outputs[proc_out_idx];
		Record_Add(clone, idx, val);
	}

	return clone;
}

static void _evaluate_proc_args
(
	OpProcCall *op
) {
	// evaluate arguments, free args from previous call
	uint arg_count = array_len(op->args);
	for(uint i = 0; i < arg_count; i++) {
		SIValue_Free(op->args[i]);
	}

	array_clear(op->args);

	for(uint i = 0; i < op->arg_count; i++) {
		array_append(op->args, AR_EXP_Evaluate(op->arg_exps[i], op->r));
	}
}

OpBase *NewProcCallOp
(
	const ExecutionPlan *plan,
	const char *proc_name,
	AR_ExpNode **arg_exps,
	AR_ExpNode **yield_exps
) {

	ASSERT(plan       != NULL);
	ASSERT(proc_name  != NULL);
	ASSERT(arg_exps   != NULL);
	ASSERT(yield_exps != NULL);

	OpProcCall *op = rm_calloc (1, sizeof(OpProcCall)) ;

	op->args       = array_new(SIValue, array_len(arg_exps));
	op->arg_exps   = arg_exps;
	op->arg_count  = array_len(arg_exps);
	op->proc_name  = proc_name;
	op->first_call = true;
	op->yield_exps = yield_exps;

	// procedure must exist
	op->procedure = Proc_Get(proc_name);
	ASSERT(op->procedure != NULL);

	uint yield_count = array_len(yield_exps);
	op->output = array_new(const char *, yield_count);
	op->yield_map = rm_malloc(sizeof(OutputMap) * yield_count);

	// set callbacks
	OpBase_Init((OpBase *)op, OPType_PROC_CALL, "ProcedureCall",
				NULL, ProcCallConsume, ProcCallReset, NULL, ProcCallClone,
				ProcCallFree, !Procedure_IsReadOnly(op->procedure), plan);

	// set modifiers
	for(uint i = 0; i < yield_count; i ++) {
		const char *alias = yield_exps[i]->resolved_name;
		const char *yield = yield_exps[i]->operand.variadic.entity_alias;

		array_append(op->output, yield);
		int rec_idx = OpBase_Modifies((OpBase *)op, alias);
		op->yield_map[i].rec_idx = rec_idx;
		op->yield_map[i].proc_out_idx = i;
	}

	return (OpBase *)op;
}

static Record ProcCallConsume
(
	OpBase *opBase
) {
	OpProcCall *op = (OpProcCall *)opBase;

	Record yield_record = NULL;
	while(!(yield_record = _yield(op))) {
		// free old record
		if(op->r != NULL) {
			OpBase_DeleteRecord(&op->r);
		}

		if(op->op.childCount == 0) {
			// "Static evaluation", return data for first call only!
			if(!op->first_call) return NULL;
			op->first_call = false;
			op->r = OpBase_CreateRecord((OpBase *)op);
		} else {
			OpBase *child = op->op.children[0];
			op->r = OpBase_Consume(child);
			if(op->r == NULL) return NULL;
		}

		_evaluate_proc_args(op);

		// free previous invocation
		// TODO: replace with Proc_Reset
		Proc_Free(op->procedure);
		op->procedure = Proc_Get(op->proc_name);

		// at the moment the only two procedures that can modify the graph are:
		// proc_fulltext_create_index
		// proc_fulltext_drop_index
		// both perform the modification once invoked without returning any
		// additional data (consume/step) function
		// this is why acquiring the write lock as we do below works
		// we will have to revisit this logic once new "write" procedures are
		// introduced

		// lock if procedure can modify the graph
		if(!Procedure_IsReadOnly(op->procedure)) QueryCtx_LockForCommit();

		ProcedureResult res = Proc_Invoke(op->procedure, op->args, op->output);

		/* TODO: should rise run-time exception?
		 * op->r will be freed in ProcCallFree. */
		if(res != PROCEDURE_OK) return NULL;
	}

	return yield_record;
}

static OpResult ProcCallReset(OpBase *ctx) {
	OpProcCall *op = (OpProcCall *)ctx;
	op->first_call = true;
	return OP_OK;
}

static OpBase *ProcCallClone(const ExecutionPlan *plan, const OpBase *opBase) {
	ASSERT(opBase->type == OPType_PROC_CALL);
	OpProcCall *op = (OpProcCall *)opBase;
	AR_ExpNode **args_exp;
	AR_ExpNode **yield_exps;
	array_clone_with_cb(args_exp, op->arg_exps, AR_EXP_Clone);
	array_clone_with_cb(yield_exps, op->yield_exps, AR_EXP_Clone);
	return NewProcCallOp(plan, op->proc_name, args_exp, yield_exps);
}

static void ProcCallFree(OpBase *ctx) {
	OpProcCall *op = (OpProcCall *)ctx;

	if(op->r) {
		OpBase_DeleteRecord(&op->r);
	}

	if(op->procedure) {
		Proc_Free(op->procedure);
		op->procedure = NULL;
	}

	if(op->yield_map) {
		rm_free(op->yield_map);
		op->yield_map = NULL;
	}

	if(op->args) {
		uint arg_count = array_len(op->args);
		for(uint i = 0; i < arg_count; i++) SIValue_Free(op->args[i]);
		array_free(op->args);
		op->args = NULL;
	}

	if(op->arg_exps) {
		for(uint i = 0; i < op->arg_count; i++) AR_EXP_Free(op->arg_exps[i]);
		array_free(op->arg_exps);
		op->arg_exps = NULL;
	}

	if(op->output) {
		array_free(op->output);
		op->output = NULL;
	}

	if(op->yield_exps) {
		uint yield_count = array_len(op->yield_exps);
		for(uint i = 0; i < yield_count; i ++) AR_EXP_Free(op->yield_exps[i]);
		array_free(op->yield_exps);
		op->yield_exps = NULL;
	}
}

