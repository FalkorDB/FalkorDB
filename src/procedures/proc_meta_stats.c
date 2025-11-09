/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "proc_ctx.h"
#include "../value.h"
#include "../query_ctx.h"
#include "../datatypes/map.h"
#include "../graph/graphcontext.h"

// CALL db.meta.stats()
// yields:
//  labels           - map containing each label and its node count
//  relTypes         - map containing each relType and its edge count
//  relCount         - number of edges in the graph
//  nodeCount        - number of nodes in the graph
//  labelCount       - number of labels in the graph
//  relTypeCount     - number of relationshipTypes in the graph
//  propertyKeyCount - number of attributeKeys in the graph

typedef struct {
	bool depleted;                    // procedure depleted
	SIValue output[7];                // procedure outputs
	SIValue *yield_labels;            // labels & counts
	SIValue *yield_relTypes;          // relTypes
	SIValue *yield_relCount;          // relationshipTypes count
	SIValue *yield_nodeCount;         // node count
	SIValue *yield_labelCount;        // label count
	SIValue *yield_relTypeCount;      // relationshipTypes & counts
	SIValue *yield_propertyKeyCount;  // number of attributeKeys
} MetaStatsCtx;

static void _process_yield
(
	MetaStatsCtx *ctx,
	const char **yield
) {
	ASSERT (ctx   != NULL) ;
	ASSERT (yield != NULL) ;

	int idx = 0;

	for (uint i = 0; i < array_len(yield); i++) {
		if (strcasecmp ("labels", yield[i]) == 0) {
			ctx->yield_labels = ctx->output + idx++ ;
			continue ;
		}

		if (strcasecmp ("relTypes", yield[i]) == 0) {
			ctx->yield_relTypes = ctx->output + idx++ ;
			continue ;
		}

		if (strcasecmp ("relCount", yield[i]) == 0) {
			ctx->yield_relCount = ctx->output + idx++ ;
			continue ;
		}

		if (strcasecmp ("nodeCount", yield[i]) == 0) {
			ctx->yield_nodeCount = ctx->output + idx++ ;
			continue ;
		}

		if (strcasecmp ("labelCount", yield[i]) == 0) {
			ctx->yield_labelCount = ctx->output + idx++ ;
			continue ;
		}

		if (strcasecmp ("relTypeCount", yield[i]) == 0) {
			ctx->yield_relTypeCount = ctx->output + idx++ ;
			continue ;
		}

		if (strcasecmp ("propertyKeyCount", yield[i]) == 0) {
			ctx->yield_propertyKeyCount = ctx->output + idx++ ;
			continue ;
		}

		ASSERT (false && "unknown yield") ;
	}
}

ProcedureResult Proc_MetaStatsInvoke
(
	ProcedureCtx *ctx,
	const SIValue *args,
	const char **yield
) {
	ASSERT (ctx   != NULL) ;
	ASSERT (args  != NULL) ;
	ASSERT (yield != NULL) ;

	// no arguments
	if (array_len ((SIValue *)args) != 0) {
		return PROCEDURE_ERR ;
	}

	// setup context
	MetaStatsCtx *pdata = rm_calloc (1, sizeof (MetaStatsCtx)) ;

	pdata->depleted = false ;

	_process_yield (pdata, yield) ;

	ctx->privateData = pdata ;

	return PROCEDURE_OK ;
}

SIValue *Proc_MetaStatsStep
(
	ProcedureCtx *ctx
) {
	ASSERT (ctx != NULL) ;

	MetaStatsCtx *pdata = (MetaStatsCtx *)ctx->privateData ;

	// depleted?
	if (pdata->depleted) {
		return NULL ;
	}
	
	pdata->depleted = true ;

	Graph        *g  = QueryCtx_GetGraph    () ;
	GraphContext *gc = QueryCtx_GetGraphCtx () ;

	if (pdata->yield_labels != NULL) {
		// create a map
		// {Person: 100, City: 20...}
		unsigned short n = GraphContext_SchemaCount (gc, SCHEMA_NODE) ;
		SIValue lbls = SI_Map (n) ;

		for (LabelID i = 0; i < n; i++) {
			Schema *s = GraphContext_GetSchemaByID (gc, i, SCHEMA_NODE) ;

			const char *name = Schema_GetName (s) ;
			uint64_t    cnt  = Graph_LabeledNodeCount (g, i) ;

			Map_AddNoClone (&lbls, SI_ConstStringVal (name), SI_LongVal (cnt)) ;
		}

		*pdata->yield_labels = lbls ;
	}

	if (pdata->yield_relTypes != NULL) {
		// create a map
		// {Knows: 100, Contact: 20...}
		unsigned short n = GraphContext_SchemaCount (gc, SCHEMA_EDGE) ;
		SIValue rels = SI_Map (n) ;

		for (RelationID i = 0; i < n; i++) {
			Schema *s = GraphContext_GetSchemaByID (gc, i, SCHEMA_EDGE) ;

			const char *name = Schema_GetName (s) ;
			uint64_t    cnt  = Graph_RelationEdgeCount (g, i) ;

			Map_AddNoClone (&rels, SI_ConstStringVal (name), SI_LongVal (cnt)) ;
		}

		*pdata->yield_relTypes = rels ;
	}

	if (pdata->yield_relCount != NULL) {
		*pdata->yield_relCount = SI_LongVal (Graph_EdgeCount (g)) ;
	}

	if (pdata->yield_nodeCount != NULL) {
		*pdata->yield_nodeCount = SI_LongVal (Graph_NodeCount (g)) ;
	}

	if (pdata->yield_labelCount != NULL) {
		*pdata->yield_labelCount = SI_LongVal (GraphContext_SchemaCount (gc,
					SCHEMA_NODE)) ;
	}

	if (pdata->yield_relTypeCount != NULL) {
		*pdata->yield_relTypeCount = SI_LongVal (GraphContext_SchemaCount (gc,
					SCHEMA_EDGE)) ;
	}

	if (pdata->yield_propertyKeyCount != NULL) {
		*pdata->yield_propertyKeyCount =
			SI_LongVal (GraphContext_AttributeCount (gc)) ;
	}

	return pdata->output ;
}

ProcedureResult Proc_MetaStatsFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if (ctx->privateData) {
		rm_free (ctx->privateData) ;
		ctx->privateData = NULL ;
	}

	return PROCEDURE_OK ;
}

ProcedureCtx *Proc_MetaStatsCtx(void) {
	ProcedureOutput output ;
	ProcedureOutput *outputs = array_new (ProcedureOutput, 7) ;

	array_append(outputs,
		((ProcedureOutput) {.name = "labels", .type = T_MAP })) ;

	array_append (outputs,
		((ProcedureOutput) {.name = "relTypes", .type = T_MAP })) ;

	array_append (outputs,
		((ProcedureOutput) {.name = "relCount", .type = T_INT64})) ;

	array_append (outputs,
		((ProcedureOutput) {.name = "nodeCount", .type = T_INT64})) ;

	array_append (outputs,
		((ProcedureOutput) {.name = "labelCount", .type = T_INT64})) ;

	array_append (outputs,
		((ProcedureOutput) {.name = "relTypeCount", .type = T_INT64 })) ;

	array_append (outputs,
		((ProcedureOutput) {.name = "propertyKeyCount", .type = T_INT64})) ;

	ProcedureCtx *ctx = ProcCtxNew("db.meta.stats",
								   0,
								   outputs,
								   Proc_MetaStatsStep,
								   Proc_MetaStatsInvoke,
								   Proc_MetaStatsFree,
								   NULL,
								   true);
	return ctx;
}

