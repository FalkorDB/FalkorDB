/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#pragma once

#include "op.h"
#include "../../ast/ast.h"
#include "../execution_plan.h"
#include "../../procedures/procedure.h"

/* Maps procedure output to record index.
 * yield element I is mapped to procedure output J
 * which will be stored within Record at position K. */
typedef struct {
	uint proc_out_idx;  // Index into procedure output.
	uint rec_idx;       // Index into record.
} OutputMap;

/* OpProcCall, */
typedef struct {
	OpBase op;                  // Base op.
    Record r;                   // Current record.
    uint arg_count;             // Number of arguments.
    AR_ExpNode **arg_exps;      // Expression representing arguments to procedure.
    SIValue *args;              // Computed arguments.
	const char **output;        // Procedure output.
	const char *proc_name;      // Procedure name.
    AR_ExpNode **yield_exps;    // Yield expressions.
	ProcedureCtx *procedure;    // Procedure to call.
	OutputMap *yield_map;       // Maps between yield to procedure output and record idx.
    bool first_call;            // Indicate first call.
} OpProcCall;

OpBase *NewProcCallOp(
	const ExecutionPlan *plan,  // Execution plan this operation belongs to.
	const char *proc_name,      // Procedure name.
    AR_ExpNode **arg_exps,      // Arguments passed to procedure invocation.
	AR_ExpNode **yield_exps     // Procedure output.
);
