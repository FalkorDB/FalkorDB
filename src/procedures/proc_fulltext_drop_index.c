/*
 * Copyright Redis Ltd. 2018 - present
 * Licensed under your choice of the Redis Source Available License 2.0 (RSALv2) or
 * the Server Side Public License v1 (SSPLv1).
 */

#include "proc_fulltext_drop_index.h"
#include "../query_ctx.h"
#include "../value.h"
#include "../util/arr.h"
#include "../util/rmalloc.h"
#include "../errors/errors.h"
#include "../index/indexer.h"
#include "../graph/graphcontext.h"

//------------------------------------------------------------------------------
// fulltext dropNodeIndex
//------------------------------------------------------------------------------

ProcedureResult Proc_FulltextDropIndexInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	// argument validations
	int argc = array_len((SIValue *)args);
	ASSERT(argc == 1);

	// expecting arg[0] to be a string
	SIValue arg = args[0];
	if(!(SI_TYPE(arg) & T_STRING)) {
		return PROCEDURE_ERR;
	}

	// try to get relevant index
	const char *lbl = arg.stringval;
	GraphContext *gc = QueryCtx_GetGraphCtx();
	Schema *s = GraphContext_GetSchema(gc, lbl, SCHEMA_NODE);

	if(s == NULL) {
		ErrorCtx_SetError(EMSG_FULLTEXT_DROP_INDEX, lbl);
		return PROCEDURE_ERR;
	}

	Index idx = Schema_GetIndex(s, NULL, 0, INDEX_FLD_ANY, true);
	if(idx == NULL) {
		ErrorCtx_SetError(EMSG_FULLTEXT_DROP_INDEX, lbl);
		return PROCEDURE_ERR;
	}

	const IndexField *fields = Index_GetFields(idx);
	int n = array_len((IndexField*)fields);
	// drop only fulltext fields
	for(int i = 0; i < n; i++) {
		const IndexField *f = fields + i;
		if(IndexField_GetType(f) & INDEX_FLD_FULLTEXT) {
			int res = GraphContext_DeleteIndex(gc, SCHEMA_NODE, lbl,
					IndexField_GetName(f), INDEX_FLD_FULLTEXT);
			ASSERT(res == INDEX_OK);
		}
	}

	return PROCEDURE_OK;
}

SIValue *Proc_FulltextDropIndexStep
(
	ProcedureCtx *ctx
) {
	return NULL;
}

// deprecated
// CALL db.idx.fulltext.drop(label)
// CALL db.idx.fulltext.drop('books')
ProcedureCtx *Proc_FulltextDropIdxGen() {
	void *privateData = NULL;
	ProcedureOutput *output = array_new(ProcedureOutput, 0);
	ProcedureCtx *ctx = ProcCtxNew("db.idx.fulltext.drop",
								   1,
								   output,
								   Proc_FulltextDropIndexStep,
								   Proc_FulltextDropIndexInvoke,
								   NULL,
								   privateData,
								   false);

	return ctx;
}

