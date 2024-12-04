/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */



#include "../util/arr.h"
#include "rax.h"
#include "../RG.h"
#include "procedure.h"
#include "proc_functions.h"
#include "../datatypes/array.h"

extern rax *__aeRegisteredFuncs;

// CALL dbms.functions()

typedef struct {
    SIValue *output;                      // array with a one entry: [name]
    raxIterator iter;                     // procedures iterator
    SIValue *yield_name;                  // yield name
    SIValue *yield_signature;             // yield the function signature in the format name: t1 ... tn -> tn+1
    SIValue *yield_description;           // yield document string
    SIValue *yield_return_type;           // yield the return type of the function
    SIValue *yield_arguments_type;        // yield the function arguments type
} ProcFunctionsPrivateData;

static void _process_yield
(
	ProcFunctionsPrivateData *ctx,
	const char **yield
) {
    ctx->yield_name                 = NULL;
    ctx->yield_signature            = NULL;
    ctx->yield_description          = NULL;
    ctx->yield_return_type          = NULL;
    ctx->yield_arguments_type       = NULL;

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
        if(strcasecmp("return_type", yield[i]) == 0) {
            ctx->yield_return_type = ctx->output + idx;
            idx++;
            continue;
        }
        if(strcasecmp("arguments_type", yield[i]) == 0) {
            ctx->yield_arguments_type = ctx->output + idx;
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
    pdata->output = array_new(SIValue, 5);
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

    // looping over all the yield outputs for this function and yielding the data if requested
    // filtering out internal functions.
    while(raxNext(&pdata->iter)){

        AR_FuncDesc *func = (AR_FuncDesc*)pdata->iter.data;
        // filter out all internal functions
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
            if(pdata->yield_return_type && func->ret_type){
                // get the return value description into buf
                SIType_ToMultipleTypeStringSimple(func->ret_type, '|', buf, bufferLen);
                *pdata->yield_return_type = SI_ConstStringVal(buf);
            }
            if(pdata->yield_arguments_type && func->types){
                int arg_count = array_len(func->types);
                if(0 < arg_count) {
                    *pdata->yield_arguments_type = SI_Array(arg_count);
                    for (int i = 0; i < arg_count; i++) {
                        // get the argument description into buf
                        SIType_ToMultipleTypeStringSimple(func->types[i], '|', buf, bufferLen);
                        SIValue value = SI_ConstStringVal(buf);
                        SIArray_Append(pdata->yield_arguments_type, value);
                    }
                } else {
                    *pdata->yield_arguments_type = SI_Array(0); 
                }
            }
            // no more data to yield for this function return the output
            return pdata->output;
        }
    }
    // this function is internal, skip it
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

    ProcedureOutput *outputs = array_new(ProcedureOutput, 5);
    ProcedureOutput out_name = {.name = "name", .type = T_STRING};
    ProcedureOutput out_signature = {.name = "signature", .type = T_STRING};
    ProcedureOutput out_description = {.name = "description", .type = T_STRING};
    ProcedureOutput out_return_type = {.name = "return_type", .type = T_STRING};
    ProcedureOutput out_arguments_type = {.name = "arguments_type", .type = T_ARRAY};

    array_append(outputs, out_name);
    array_append(outputs, out_signature);
    array_append(outputs, out_description);
    array_append(outputs, out_return_type);
    array_append(outputs, out_arguments_type);

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
