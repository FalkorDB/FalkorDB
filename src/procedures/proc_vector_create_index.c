/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "../value.h"
#include "../query_ctx.h"
#include "../index/indexer.h"
#include "../datatypes/map.h"
#include "../errors/errors.h"
#include "../graph/graphcontext.h"
#include "proc_vector_create_index.h"

// parse procedure arguments
static bool _parseArgs
(
	const SIValue *args,     // arguments
	SchemaType *type,        // (node/edge)
	const char **label,      // label to index
	const char **attribute,  // attribute to index
	uint32_t *dimension      // vector length
) {
	// expecting a single MAP argument
	if(array_len((SIValue*)args) != 1) {
		return false;
	}

	SIValue arg = args[0];

	if(SI_TYPE(arg) != T_MAP) {
		return false;
	}
	
	// expecting a map with the following fields:
	// {
	//     type:'NODE',
	//     label:'Person',
	//     attribute:'embeddings',
	//     length:538,
	//     similarityFunction:'euclidean'
	//  }

	// extract fields from the map
	SIValue val;

	//--------------------------------------------------------------------------
	// extract type
	//--------------------------------------------------------------------------

	if(!MAP_GET(arg, "type", val) || SI_TYPE(val) != T_STRING) {
		return false;
	}

	// make sure type is either NODE or RELATIONSHIP
	if(strcasecmp(val.stringval, "NODE") == 0) {
		*type = SCHEMA_NODE;
	} else if(strcasecmp(val.stringval, "RELATIONSHIP") == 0) {
		*type = SCHEMA_EDGE;
	} else {
		return false;
	}

	//--------------------------------------------------------------------------
	// extract label
	//--------------------------------------------------------------------------

	if(!MAP_GET(arg, "label", val) || SI_TYPE(val) != T_STRING) {
		return false;
	}
	*label = val.stringval;

	//--------------------------------------------------------------------------
	// extract attribute
	//--------------------------------------------------------------------------

	if(!MAP_GET(arg, "attribute", val) || SI_TYPE(val) != T_STRING) {
		return false;
	}
	*attribute = val.stringval;

	//--------------------------------------------------------------------------
	// extract vector length
	//--------------------------------------------------------------------------

	if(!MAP_GET(arg, "dim", val) || SI_TYPE(val) != T_INT64) {
		return false;
	}
	*dimension = val.longval;

	//--------------------------------------------------------------------------
	// extract similarity function
	//--------------------------------------------------------------------------

	if(!MAP_GET(arg, "similarityFunction", val) || SI_TYPE(val) != T_STRING) {
		return false;
	}

	// at the moment only euclidean distance is supported
	if(strcasecmp(val.stringval, "euclidean") != 0) {
		return false;
	}

	return true;
}

ProcedureResult Proc_VectorCreateIdxInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	ASSERT(ctx  != NULL);

	ProcedureResult res = PROCEDURE_OK;

	// arguments
	const char*     label       = NULL;            // label to index
	const char*     attribute   = NULL;            // attribute to index
	uint32_t        dimension   = 0;               // vector length
	SchemaType      schema_type = SCHEMA_NODE;     // (node/edge) 

	//--------------------------------------------------------------------------
	// parse arguments
	//--------------------------------------------------------------------------

	if(!_parseArgs(args, &schema_type, &label, &attribute, &dimension)) {
		ErrorCtx_SetError(EMSG_VECTOR_IDX_CREATE_INVALID_CALL);
		res = PROCEDURE_ERR;
		goto cleanup;
	}

	Index idx = NULL;
	GraphContext *gc = QueryCtx_GetGraphCtx();

	//--------------------------------------------------------------------------
	// create index
	//--------------------------------------------------------------------------

	if(GraphContext_AddVectorIndex(&idx, gc, schema_type, label, attribute,
				dimension) == false) {
		ErrorCtx_SetError(EMSG_VECTOR_IDX_CREATE_FAIL);
		res = PROCEDURE_ERR;
		goto cleanup;
	}

	//--------------------------------------------------------------------------
	// populate index asynchornously
	//--------------------------------------------------------------------------

	Schema *s = GraphContext_GetSchema(gc, label, schema_type);
	ASSERT(s != NULL);

	Indexer_PopulateIndex(gc, s, idx);

cleanup:
	return res;
}

SIValue *Proc_VectorCreateIdxStep
(
	ProcedureCtx *ctx
) {
	return NULL;
}

// procedure context for the vector index creation
// 
// procedure call example
// CALL db.idx.vector.createIndex(
//     {type:'NODE'/'RELATIONSHIP',
//     label:'Person',
//     attribute:'embeddings',
//     dim:538,
//     similarityFunction:'euclidean'})
//
ProcedureCtx *Proc_VectorCreateIdxGen(void) {
	return ProcCtxNew("db.idx.vector.createIndex", 1, NULL,
			Proc_VectorCreateIdxStep, Proc_VectorCreateIdxInvoke, NULL, NULL,
			false);
}

