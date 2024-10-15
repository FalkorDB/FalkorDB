/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */



#include "../util/arr.h"
#include "rax.h"
#include "../RG.h"
#include "procedure.h"
#include "proc_functions.h"

extern rax *__aeRegisteredFuncs;

// CALL dbms.functions()

typedef struct {
	SIValue *output;              // array with a one entry: [name]
	raxIterator iter;             // procedures iterator
	SIValue *yield_name;          // yield name
    SIValue *yield_signature;     // yield the function signature in the format name: t1 ... tn -> tn+1
    SIValue *yield_description;   // yield document string
} ProcFunctionsPrivateData;

static void _process_yield
(
	ProcFunctionsPrivateData *ctx,
	const char **yield
) {
	ctx->yield_name = NULL;
	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("name", yield[i]) == 0) {
			ctx->yield_name = ctx->output + idx;
			idx++;
			continue;
		}
        if(strcasecmp("signature", yield[i]) == 0) {
			ctx->yield_signature = ctx->output + idx;
			idx++;
			continue;
		}
        if(strcasecmp("description", yield[i]) == 0) {
			ctx->yield_description = ctx->output + idx;
			idx++;
			continue;
		}

	}
}

ProcedureResult Proc_FunctionsInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	if(array_len((SIValue *)args) != 0) return PROCEDURE_ERR;

	ProcFunctionsPrivateData *pdata = rm_malloc(sizeof(ProcFunctionsPrivateData));
	memset(pdata, 0, sizeof(ProcFunctionsPrivateData));

	// initialize an iterator to the rax that contains all functions
	rax *functions = __aeRegisteredFuncs;
	raxStart(&pdata->iter, functions);
	raxSeek(&pdata->iter, "^", NULL, 0);
	pdata->output = array_new(SIValue, 3);
	_process_yield(pdata, yield);

	ctx->privateData = pdata;
	return PROCEDURE_OK;
}


// promote the rax iterator to the next function and return its name and mode.
SIValue *Proc_FunctionsStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx->privateData);
	ASSERT(ctx->privateData != NULL);
    
    size_t bufferLen = MULTIPLE_TYPE_STRING_BUFFER_SIZE * 15;
	char buf[bufferLen];

	ProcFunctionsPrivateData *pdata = (ProcFunctionsPrivateData *)ctx->privateData;

	// filter out all internal functions
    while(raxNext(&pdata->iter)){

        AR_FuncDesc *func = (AR_FuncDesc*)pdata->iter.data;
        if(!func->internal){      
            if(pdata->yield_name && func->name ){
                *pdata->yield_name = SI_ConstStringVal(func->name);
            }
            if(pdata->yield_signature){
                // get the function signature into buf
                SITypes_SignatureToString(func->name, func->ret_type, func->types, buf, bufferLen);
                *pdata->yield_signature = SI_DuplicateStringVal(buf); 
            }
            if(pdata->yield_description && func->description){
                *pdata->yield_description = func->description ? SI_ConstStringVal(func->description) : SI_ConstStringVal("No description available");
            }
            return pdata->output;
        }

    }
    return NULL;
}

ProcedureResult Proc_FunctionsFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData) {
		ProcFunctionsPrivateData *pdata = ctx->privateData;
		raxStop(&pdata->iter);
		array_free(pdata->output);
		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

ProcedureCtx *Proc_FunctionsCtx() {
	void *privateData = NULL;

    ProcedureOutput *outputs = array_new(ProcedureOutput, 3);
    ProcedureOutput out_name = {.name = "name", .type = T_STRING};
    ProcedureOutput out_signature = {.name = "signature", .type = T_STRING};
    ProcedureOutput out_description = {.name = "description", .type = T_STRING};

	array_append(outputs, out_name);
    array_append(outputs, out_signature);
    array_append(outputs, out_description);

    ProcedureCtx *ctx = ProcCtxNew("dbms.functions",
								   0,
								   outputs,
								   Proc_FunctionsStep,
								   Proc_FunctionsInvoke,
								   Proc_FunctionsFree,
								   privateData,
								   true);
	return ctx;
}




