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

// maps procedure output to record index
// yield element I is mapped to procedure output J
// which will be stored within Record at position K
typedef struct {
	uint proc_out_idx;  // index into procedure output
	uint rec_idx;       // index into record
} OutputMap;

// OpProcCall
typedef struct {
	OpBase op;                // base op
    Record r;                 // current record
    uint arg_count;           // number of arguments
    AR_ExpNode **arg_exps;    // expression representing arguments to procedure
    SIValue *args;            // computed arguments
	const char **output;      // procedure output
	const char *proc_name;    // procedure name
    AR_ExpNode **yield_exps;  // yield expressions
	ProcedureCtx *procedure;  // procedure to call
	OutputMap *yield_map;     // maps between yield to procedure output and record idx
    bool first_call;          // indicate first call
} OpProcCall;

OpBase *NewProcCallOp(
	ExecutionPlan *plan,     // execution plan this operation belongs to
	const char *proc_name,   // procedure name
    AR_ExpNode **arg_exps,   // arguments passed to procedure invocation
	AR_ExpNode **yield_exps  // procedure output
);

