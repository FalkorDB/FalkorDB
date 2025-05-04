/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "./procedure.h"
#include "procedures.h"
#include "rax.h"
#include "../RG.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "../util/strutil.h"
#include "../graph/graphcontext.h"

rax *__procedures = NULL;

static void _procRegister
(
	const char *procedure,
	ProcGenerator gen
) {
	char lowercase_proc_name[128];
	size_t lowercase_proc_name_len = 128;
	str_tolower_ascii(procedure, lowercase_proc_name, &lowercase_proc_name_len);
	raxInsert(__procedures, (unsigned char *)lowercase_proc_name,
		   lowercase_proc_name_len, gen, NULL);
}

// register procedures
void Proc_Register() {
	__procedures = raxNew();
	_procRegister("db.labels",            Proc_LabelsCtx);
	_procRegister("db.indexes",           Proc_IndexesCtx);
	_procRegister("db.constraints",       Proc_ConstraintsCtx);
	_procRegister("db.propertyKeys",      Proc_PropKeysCtx);
	_procRegister("dbms.procedures",      Proc_ProceduresCtx);
	_procRegister("db.relationshipTypes", Proc_RelationsCtx);

	// register graph algorithms
	_procRegister("algo.BFS",         Proc_BFS_Ctx);
	_procRegister("algo.WCC",         Proc_WCCCtx);
	_procRegister("algo.pageRank",    Proc_PagerankCtx);
	_procRegister("algo.SPpaths",     Proc_SPpathCtx);
	_procRegister("algo.SSpaths",     Proc_SSpathCtx);
	_procRegister("algo.betweenness", Proc_BetweennessCtx);

	// register FullText Search generator
	_procRegister("db.idx.fulltext.drop",               Proc_FulltextDropIdxGen);
	_procRegister("db.idx.fulltext.queryNodes",         Proc_FulltextQueryNodeGen);
	_procRegister("db.idx.fulltext.createNodeIndex",    Proc_FulltextCreateNodeIdxGen);
	_procRegister("db.idx.fulltext.queryRelationships", Proc_FulltextQueryRelationshipGen);

	// register vector search generator
	_procRegister("db.idx.vector.queryNodes",         Proc_VectorQueryNodeCtx);
	_procRegister("db.idx.vector.queryRelationships", Proc_VectorQueryRelCtx);
}

ProcedureCtx *ProcCtxNew
(
	const char *name,
	unsigned int argc,
	ProcedureOutput *output,
	ProcStep fStep,
	ProcInvoke fInvoke,
	ProcFree fFree,
	void *privateData,
	bool readOnly
) {
	ProcedureCtx *ctx = rm_malloc(sizeof(ProcedureCtx));

	ctx->argc        = argc;
	ctx->name        = name;
	ctx->Step        = fStep;
	ctx->Free        = fFree;
	ctx->output      = output;
	ctx->Invoke      = fInvoke;
	ctx->readOnly    = readOnly;
	ctx->privateData = privateData;

	return ctx;
}

ProcedureCtx *Proc_Get
(
	const char *proc_name
) {
	if(!__procedures) return NULL;

	size_t proc_name_len = strlen(proc_name) + 1;
	char proc_name_lowercase [proc_name_len];
	str_tolower_ascii(proc_name, proc_name_lowercase, &proc_name_len);
	ProcGenerator gen = raxFind(__procedures, (unsigned char *)proc_name_lowercase,
	  			proc_name_len);
	if(gen == raxNotFound) return NULL;
	ProcedureCtx *ctx = gen();

	// set procedure state to not initialized
	ctx->state = PROCEDURE_NOT_INIT;
	return ctx;
}

ProcedureResult Proc_Invoke
(
	ProcedureCtx *proc,
	const SIValue *args,
	const char **yield
) {
	ASSERT(proc != NULL);

	// procedure is expected to be in the `PROCEDURE_NOT_INIT` state
	if(proc->state != PROCEDURE_NOT_INIT) {
		proc->state = PROCEDURE_ERROR;
		return PROCEDURE_ERR;
	}

	if(proc->argc != PROCEDURE_VARIABLE_ARG_COUNT) {
		uint argc = array_len((SIValue *)args);
		ASSERT(proc->argc == argc);
	}

	ProcedureResult res = proc->Invoke(proc, args, yield);
	// set state to initialized
	if(res == PROCEDURE_OK) proc->state = PROCEDURE_INIT;
	return res;
}

SIValue *Proc_Step
(
	ProcedureCtx *proc
) {
	ASSERT(proc != NULL);
	// validate procedure state, can only consumed if state is initialized
	if(proc->state != PROCEDURE_INIT) return NULL;

	SIValue *val = proc->Step(proc);
	// set procedure state to depleted if NULL is returned
	// NOTE: we might have errored
	if(val == NULL) proc->state = PROCEDURE_DEPLETED;
	return val;
}

uint Procedure_Argc
(
	const ProcedureCtx *proc
) {
	ASSERT(proc != NULL);
	return proc->argc;
}

uint Procedure_OutputCount
(
	const ProcedureCtx *proc
) {
	ASSERT(proc != NULL);
	return array_len(proc->output);
}

const char *Procedure_GetOutput
(
	const ProcedureCtx *proc,
	uint output_idx
) {
	ASSERT(proc != NULL);
	ASSERT(output_idx < Procedure_OutputCount(proc));
	return proc->output[output_idx].name;
}

bool Procedure_ContainsOutput
(
	const ProcedureCtx *proc,
	const char *output
) {
	ASSERT(proc != NULL);
	ASSERT(output != NULL);
	uint output_count = array_len(proc->output);
	for(uint i = 0; i < output_count; i++) {
		if(strcmp(proc->output[i].name, output) == 0) return true;
	}
	return false;
}

const char *Procedure_GetName
(
	const ProcedureCtx *proc
) {
	ASSERT(proc != NULL);
	return proc->name;
}

bool Procedure_IsReadOnly
(
	const ProcedureCtx *proc
) {
	ASSERT(proc != NULL);
	return proc->readOnly;
}

void Proc_Free
(
	ProcedureCtx *proc
) {
	if(proc == NULL) {
		return;
	}

	if(proc->Free != NULL) {
		proc->Free(proc);
	}

	if(proc->output != NULL) {
		array_free(proc->output);
	}

	rm_free(proc);
}

