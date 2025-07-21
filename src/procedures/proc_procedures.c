/*
 * Copyright (c) 2006-Present, Redis Ltd.
 * All rights reserved.
 *
 * Licensed under your choice of the Redis Source Available License 2.0
 * (RSALv2); or (b) the Server Side Public License v1 (SSPLv1); or (c) the
 * GNU Affero General Public License v3 (AGPLv3).
*/

#include "../util/arr.h"
#include "rax.h"
#include "../RG.h"
#include "procedure.h"
#include "proc_procedures.h"

extern rax *__procedures;

// CALL dbms.procedures()

typedef struct {
	SIValue *output;      // array with a maximum of 2 entries: [name, mode]
	raxIterator iter;     // procedures iterator
	SIValue *yield_name;  // yield name
	SIValue *yield_mode;  // yield mode
} ProcProceduresPrivateData;

static void _process_yield
(
	ProcProceduresPrivateData *ctx,
	const char **yield
) {
	ctx->yield_name = NULL;
	ctx->yield_mode = NULL;
	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("name", yield[i]) == 0) {
			ctx->yield_name = ctx->output + idx;
			idx++;
			continue;
		}
		if(strcasecmp("mode", yield[i]) == 0) {
			ctx->yield_mode = ctx->output + idx;
			idx++;
			continue;
		}
	}
}

ProcedureResult Proc_ProceduresInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	if(array_len((SIValue *)args) != 0) return PROCEDURE_ERR;

	ProcProceduresPrivateData *pdata = rm_malloc(sizeof(ProcProceduresPrivateData));

	// initialize an iterator to the rax that contains all procedures
	rax *procedures = __procedures;
	raxStart(&pdata->iter, procedures);
	raxSeek(&pdata->iter, "^", NULL, 0);
	pdata->output = array_new(SIValue, 2);
	_process_yield(pdata, yield);

	ctx->privateData = pdata;
	return PROCEDURE_OK;
}

// promote the rax iterator to the next procedure and return its name and mode.
SIValue *Proc_ProceduresStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData);

	ProcProceduresPrivateData *pdata = (ProcProceduresPrivateData *)ctx->privateData;
	// depleted?
	if(!raxNext(&pdata->iter)) return NULL;

	// returns the current procedure's name and mode
	ProcGenerator gen = pdata->iter.data;
	ProcedureCtx *curr_proc_ctx = gen();

	if(pdata->yield_name) *pdata->yield_name =
			SI_ConstStringVal(Procedure_GetName(curr_proc_ctx));
	if(pdata->yield_mode) *pdata->yield_mode =
			Procedure_IsReadOnly(curr_proc_ctx) ? SI_ConstStringVal("READ") :
			SI_ConstStringVal("WRITE");

	Proc_Free(curr_proc_ctx);
	return pdata->output;
}

ProcedureResult Proc_ProceduresFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData) {
		ProcProceduresPrivateData *pdata = ctx->privateData;
		array_free(pdata->output);
		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_ProceduresCtx() {
	void *privateData = NULL;

	ProcedureOutput *outputs = array_new(ProcedureOutput, 2);
	ProcedureOutput out_name = {.name = "name", .type = T_STRING};
	ProcedureOutput out_mode = {.name = "mode", .type = T_STRING};
	array_append(outputs, out_name);
	array_append(outputs, out_mode);

	ProcedureCtx *ctx = ProcCtxNew("dbms.procedures",
								   0,
								   outputs,
								   Proc_ProceduresStep,
								   Proc_ProceduresInvoke,
								   Proc_ProceduresFree,
								   privateData,
								   true);
	return ctx;
}

