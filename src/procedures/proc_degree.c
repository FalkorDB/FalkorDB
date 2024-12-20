/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "proc_degree.h"

#include "../value.h"
#include "../query_ctx.h"
#include "../graph/graph.h"
#include "../datatypes/map.h"
#include "../algorithms/degree.h"
#include "../graph/graphcontext.h"
#include "../graph/delta_matrix/delta_matrix_iter.h"

// degree procedure context
typedef struct {
	const Graph *g;         // graph
	bool depleted;          // procedure depleted
	Node node;              // current node being yield
	GrB_Vector degree;      // 1xn vector containing node degree
	GxB_Iterator it;        // iterator over the degree matrix
	SIValue *output;        // array with up to two entries [node, degree]
	SIValue *yield_node;    // yield node
	SIValue *yield_degree;  // yield degree
} DegreeContext;

// setup procedure outputs according to yield
// CALL algo.degree({}) YIELD node, degree
static void _process_yield
(
	DegreeContext *ctx,
	const char **yield
) {
	ASSERT(ctx              != NULL);
	ASSERT(yield            != NULL);
	ASSERT(array_len(yield) <= 2);

	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		// yield node
		if(strcasecmp("node", yield[i]) == 0) {
			ctx->yield_node = ctx->output + idx;
			idx++;
		}

		// yield degree
		else if(strcasecmp("degree", yield[i]) == 0) {
			ctx->yield_degree = ctx->output + idx;
			idx++;
		}
	}
}

// parse algo.degree configuration map
//
//	{
//		'source':      <label>,
//		'dir':         'incoming' / 'outgoing',
//		'relation':    <relationship-type>,
//		'destination': <label>,
//	}
static bool parse_arguments
(
	SIValue config,          // input configuration map
	const char **src_label,  // src label
	GRAPH_EDGE_DIR *dir,     // edge direction
	const char **rel_type,   // edge relationship type
	const char **dest_label  // destination label
) {
	ASSERT(dir        != NULL);
	ASSERT(rel_type   != NULL);
	ASSERT(src_label  != NULL);
	ASSERT(dest_label != NULL);

	// init inputs to default values
	*src_label  = NULL;
	*rel_type   = NULL;
	*dest_label = NULL;
	*dir        = GRAPH_EDGE_DIR_OUTGOING;

	// validate config is a map
	if(SI_TYPE(config) != T_MAP) {
		return false;
	}

	SIValue tmp;      // current value
	int matches = 0;  // number of keys matched

	//--------------------------------------------------------------------------
	// get source label
	//--------------------------------------------------------------------------

	if(Map_Get(config, SI_ConstStringVal("source"), &tmp)) {
		if(SI_TYPE(tmp) != T_STRING) {
			return false;
		}

		*src_label = tmp.stringval;
		matches++;
	}

	//--------------------------------------------------------------------------
	// get edge direction
	//--------------------------------------------------------------------------

	if(Map_Get(config, SI_ConstStringVal("dir"), &tmp)) {
		if(SI_TYPE(tmp) == T_STRING) {
			if(strcmp(tmp.stringval, "incoming") == 0) {
				*dir = GRAPH_EDGE_DIR_INCOMING;
			} else if(strcmp(tmp.stringval, "outgoing") == 0) {
				*dir = GRAPH_EDGE_DIR_OUTGOING;
			} else {
				// unknown edge direction, fail
				return false;
			}
			matches++;
		}
	}

	//--------------------------------------------------------------------------
	// get edge relationship type
	//--------------------------------------------------------------------------

	if(Map_Get(config, SI_ConstStringVal("relation"), &tmp)) {
		if(SI_TYPE(tmp) != T_STRING) {
			return false;
		}

		*rel_type = tmp.stringval;
		matches++;
	}

	//--------------------------------------------------------------------------
	// get destination label
	//--------------------------------------------------------------------------

	if(Map_Get(config, SI_ConstStringVal("destination"), &tmp)) {
		if(SI_TYPE(tmp) != T_STRING) {
			return false;
		}

		*dest_label = tmp.stringval;
		matches++;
	}

	// make sure config map doesn't contains unexpected keys
	if(Map_KeyCount(config) > matches) {
		// map contains unmatched keys
		// e.g.
		// {'sources': 'Person', 'unexpected_key': 4}
		return false;
	}

	return true;
}

// procedure invoke callback
ProcedureResult Proc_DegreeInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // invocation args
	const char **yield    // output(s)
) {
	// expecting a single map argument
	if(array_len((SIValue *)args) != 1) return PROCEDURE_ERR;

	// configuration
	GRAPH_EDGE_DIR dir;      // edge direction
	const char *src_label;   // src label
	const char *rel_type;    // edge relationship type
	const char *dest_label;  // destination label

	// parse configuration
	if(!parse_arguments(args[0], &src_label, &dir, &rel_type, &dest_label)) {
		// failed to parse input configuration
		// emit an error and return
		ErrorCtx_RaiseRuntimeException(EMSG_PROC_INVALID_ARGUMENTS, "algo.degree");
		return PROCEDURE_ERR;
	}

	Schema *s;
	GraphContext *gc        = QueryCtx_GetGraphCtx();
	Graph        *g         = GraphContext_GetGraph(gc);
	bool          transpose = (dir == GRAPH_EDGE_DIR_INCOMING);

	Delta_Matrix S = NULL;  // source label matrix
	Delta_Matrix R = NULL;  // relation adjacency matrix
	Delta_Matrix D = NULL;  // destination label matrix

	//--------------------------------------------------------------------------
	// get source label matrix
	//--------------------------------------------------------------------------

	if(src_label != NULL) {
		s = GraphContext_GetSchema(gc, src_label, SCHEMA_NODE);
		if(s != NULL) {
			S = Graph_GetLabelMatrix(g, Schema_GetID(s));
		} else {
			S = Graph_GetZeroMatrix(g);
		}
	}

	//--------------------------------------------------------------------------
	// get relationship type matrix
	//--------------------------------------------------------------------------

	if(rel_type != NULL) {
		s = GraphContext_GetSchema(gc, rel_type, SCHEMA_EDGE);
		if(s != NULL) {
			R = Graph_GetRelationMatrix(g, Schema_GetID(s), transpose);
			// error in case relationship contains tensors
			// TODO: support tensors
			if(Graph_RelationshipContainsMultiEdge(g, Schema_GetID(s))) {
				ErrorCtx_RaiseRuntimeException("'algo.degree' can't run against a graph containing tensors");
				return PROCEDURE_ERR;
			}
		} else {
			R = Graph_GetZeroMatrix(g);
		}
	} else {
		// default adjacency matrix
		R = Graph_GetAdjacencyMatrix(g, transpose);

		GrB_Index nvals;
		Delta_Matrix_nvals(&nvals, R);

		if(Graph_EdgeCount(g) != nvals) {
			// error in case graph contains tensors
			ErrorCtx_RaiseRuntimeException("'algo.degree' can't run against a graph containing tensors");
			return PROCEDURE_ERR;
		}
	}

	//--------------------------------------------------------------------------
	// get destination label matrix
	//--------------------------------------------------------------------------

	if(dest_label != NULL) {
		s = GraphContext_GetSchema(gc, dest_label, SCHEMA_NODE);
		if(s != NULL) {
			D = Graph_GetLabelMatrix(g, Schema_GetID(s));
		} else {
			D = Graph_GetZeroMatrix(g);
		}
	}

	//--------------------------------------------------------------------------
	// compute adjacency matrix
	//--------------------------------------------------------------------------

	// either S or D or both are specified
	GrB_Info     info;
	Delta_Matrix ADJ;
	size_t n = Graph_RequiredMatrixDim(g);

	info = Delta_Matrix_new(&ADJ , GrB_UINT64, n, n, false);
	ASSERT(info == GrB_SUCCESS);

	if(S != NULL || D != NULL) {
		// set up I
		GrB_Vector diag;
		info = GrB_Vector_new(&diag, GrB_BOOL, n);
		ASSERT(info == GrB_SUCCESS);

		// diag = ones (n, 1)
		info = GrB_assign(diag, NULL, NULL, 1, GrB_ALL, n, NULL);
		ASSERT(info == GrB_SUCCESS);

		GrB_Matrix _I = Delta_Matrix_M(ADJ);
		info = GxB_Matrix_diag(_I, diag, 0, NULL);
		ASSERT(info == GrB_SUCCESS);

		Delta_Matrix identity = ADJ;

		if(S != NULL && D != NULL) {
			// ADJ = I * S * R * D
			info = Delta_mxm(ADJ, GxB_ANY_PAIR_BOOL, identity, S);
			ASSERT(info == GrB_SUCCESS);

			info = Delta_mxm(ADJ, GxB_ANY_PAIR_BOOL, ADJ, R);
			ASSERT(info == GrB_SUCCESS);

			info = Delta_mxm(ADJ, GxB_ANY_PAIR_BOOL, ADJ, D);
			ASSERT(info == GrB_SUCCESS);
		} else if(S != NULL) {
			// ADJ == I * S * R
			info = Delta_mxm(ADJ, GxB_ANY_PAIR_BOOL, identity, S);
			ASSERT(info == GrB_SUCCESS);

			info = Delta_mxm(ADJ, GxB_ANY_PAIR_BOOL, ADJ, R);
			ASSERT(info == GrB_SUCCESS);
		} else {
			// ADJ == I * R * D
			info = Delta_mxm(ADJ, GxB_ANY_PAIR_BOOL, identity, R);
			ASSERT(info == GrB_SUCCESS);

			info = Delta_mxm(ADJ, GxB_ANY_PAIR_BOOL, ADJ, D);
			ASSERT(info == GrB_SUCCESS);
		}

		GrB_free(&diag);
	} else {
		// ADJ is a copy of R
		info = Delta_Matrix_copy(ADJ, R);
		ASSERT(info == GrB_SUCCESS);
	}

	// make sure ADJ doesn't contains any pendding changes
	info = Delta_Matrix_wait(ADJ, true);
	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// compute degrees
	//--------------------------------------------------------------------------

	GrB_Matrix _ADJ = Delta_Matrix_M(ADJ);

	GrB_Vector degree;
	info = Degree(&degree, _ADJ);
	ASSERT(info == GrB_SUCCESS);
	ASSERT(degree != NULL);

	Delta_Matrix_free(&ADJ);

	//--------------------------------------------------------------------------
	// initialize procedure context
	//--------------------------------------------------------------------------
	
	DegreeContext *pdata = rm_calloc(1, sizeof(DegreeContext));
	ctx->privateData = pdata;

	pdata->g      = QueryCtx_GetGraph();
	pdata->degree = degree;
	pdata->output = rm_malloc(sizeof(SIValue) * 2);

	_process_yield(pdata, yield);

	// attach iterator to matrix
	info = GxB_Iterator_new(&pdata->it);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_Vector_Iterator_attach(pdata->it, degree, NULL);
	ASSERT(info == GrB_SUCCESS);

	pdata->depleted = (GxB_Vector_Iterator_seek(pdata->it, 0) != GrB_SUCCESS);

	return PROCEDURE_OK;
}

// procedure step function
SIValue *Proc_DegreeStep
(
	ProcedureCtx *ctx
) {
	ASSERT(ctx              != NULL);
	ASSERT(ctx->privateData != NULL);

	DegreeContext *pdata = (DegreeContext*)ctx->privateData;

	// return if iterator depleted
	if(pdata->depleted) {
		return NULL;
	}

	// get current entry
	NodeID   id     = GxB_Vector_Iterator_getIndex(pdata->it);  // node id
	uint64_t degree = GxB_Iterator_get_UINT64(pdata->it);       // node degree

	// yield node
	if(pdata->yield_node) {
		Graph_GetNode(pdata->g, id, &pdata->node);
		*pdata->yield_node = SI_Node(&pdata->node);
	}

	// yield degree
	if(pdata->yield_degree) {
		*pdata->yield_degree = SI_DoubleVal(degree);
	}

	// advance to next entry
	pdata->depleted = (GxB_Vector_Iterator_next(pdata->it) != GrB_SUCCESS);

	return pdata->output;
}

// procedure context free callback
ProcedureResult Proc_DegreeFree
(
	ProcedureCtx *ctx
) {
	DegreeContext *pdata = (DegreeContext*)ctx->privateData;

	if(pdata != NULL) {
		GrB_Info info = GxB_Iterator_free(&pdata->it);
		ASSERT(info == GrB_SUCCESS);

		if(pdata->degree != NULL) {
			info = GrB_free(&pdata->degree);
			ASSERT(info == GrB_SUCCESS);
		}

		if(pdata->output) {
			rm_free(pdata->output);
		}

		rm_free(pdata);
	}

	return PROCEDURE_OK;
}

// define the Degree procedure
// procedure input:
//	{
//		'source':      <label>,
//		'dir':         'incoming' / 'outgoing',
//		'relation':    <relationship-type>,
//		'destination': <label>,
//	}
//
//  source      - [optional] [string] type of nodes for which degree is computed
//  dir         - [optional] [string] edge direction: 'incoming' or 'outgoing'
//  relation    - [optional] [string] the type of edges to consider
//  destination - [optional] [string] type of reachable nodes
//
//  examples:
//
//  CALL algo.degree({})
//  CALL algo.degree({source: 'L', relation: 'R', dir: 'outgoing', destination: 'M'})
ProcedureCtx *Proc_DegreeCtx()
{
	void *privateData = NULL;

	ProcedureOutput *outputs      = array_new(ProcedureOutput, 1);
	ProcedureOutput output_node   = {.name = "node",   .type = T_NODE};
	ProcedureOutput output_degree = {.name = "degree", .type = T_DOUBLE};

	array_append(outputs, output_node);
	array_append(outputs, output_degree);

	return ProcCtxNew("algo.degree", 1, outputs, Proc_DegreeStep,
			Proc_DegreeInvoke, Proc_DegreeFree, privateData, true);
}

