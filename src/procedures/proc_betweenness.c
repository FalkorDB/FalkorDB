/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "LAGraph.h"
#include "GraphBLAS.h"

#include "proc_betweenness.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"
#include "../graph/graphcontext.h"

#define BETWEENNESS_DEFAULT_SAMPLE_SIZE 32

// CALL algo.betweenness({}) YIELD node, score
// CALL algo.betweenness(NULL) YIELD node, score
// CALL algo.betweenness({nodeLabels: ['L', 'P']}) YIELD node, score
// CALL algo.betweenness({relationshipTypes: ['R', 'E']}) YIELD node, score
// CALL algo.betweenness({nodeLabels: ['L'], relationshipTypes: ['E']}) YIELD node, score
// CALL algo.betweenness({nodeLabels: ['L'], samplingSize:20, samplingSeed: 10})

typedef struct {
	Graph *g;              // graph
	GrB_Vector nodes;      // nodes participating in computation
	GrB_Vector centrality; // centrality(i): betweeness centrality of node i
	GrB_Info info;         // iterator state
	GxB_Iterator it;       // centrality iterator
	Node node;             // node
	SIValue output[2];     // array with up to 2 entries [node, score]
	SIValue *yield_node;   // yield node
	SIValue *yield_score;  // yield score
} Betweenness_Context;

// pick random set of source nodes
// the set is returned as a GrB_Index array, it is the callers responsibility
// to free the array
static GrB_Index* _Random_Sources
(
	GrB_Matrix AT,          // transposed adjacency matrix
	int32_t *samplingSize,  // size of sample
	int32_t samplingSeed    // random seed
) {
	GrB_Info info;

	// pick random nodes from AT's j array
	// AT->j[i] contains the ID of a reachable source node
	// to gain access to AT->j we need to unload AT into a container
	// within the container j is refered to as i

	GxB_Container container;
	info = GxB_Container_new(&container);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_unload_Matrix_into_Container(AT, container, NULL);
	ASSERT(info == GrB_SUCCESS);

	*samplingSize = MIN(container->nvals, *samplingSize);

	// allocate sources array
	GrB_Index *sources = rm_malloc(sizeof(GrB_Index) * (*samplingSize));

	// pick random sources
	for(int i = 0; i < *samplingSize; i++) {
		uint64_t x;
		GrB_Index idx = rand_r(&samplingSeed) % container->nvals;

		info = GrB_Vector_extractElement(&x, container->i, idx);
		ASSERT(info == GrB_SUCCESS);

		sources[i] = x;
	}

	// load AT back from the container
	info = GxB_load_Matrix_from_Container(AT, container, NULL);
	ASSERT(info == GrB_SUCCESS);

	// discard container
	info = GxB_Container_free(&container);
	ASSERT(info == GrB_SUCCESS);

	return sources;
}

// build adjacency matrix which will be used to compute Betweenness-Centrality
static GrB_Matrix _Build_Matrix
(
	const Graph *g,         // graph
	GrB_Vector *N,          // list nodes participating in betweenness
	const LabelID *lbls,    // [optional] labels to consider
	unsigned short n_lbls,  // number of labels
	const RelationID *rels, // [optional] relationships to consider
	unsigned short n_rels   // number of relationships
) {
	GrB_Info info;
	Delta_Matrix D;       // graph delta matrix
	GrB_Index nrows;      // number of rows in matrix
	GrB_Index ncols;      // number of columns in matrix
	GrB_Matrix A = NULL;  // returned betweenness matrix

	// if no relationships are specified use the adjacency matrix
	// otherwise use specified relation matrices
	if(n_rels == 0) {
		D = Graph_GetAdjacencyMatrix(g, false);
	} else {
		RelationID id = rels[0];
		D = Graph_GetRelationMatrix(g, id, false);
	}

	// export relation matrix to A
	info = Delta_Matrix_export(&A, D);
	ASSERT(info == GrB_SUCCESS);

	// in case there are multiple relation types, include them in A
	for(unsigned short i = 1; i < n_rels; i++) {
		D = Graph_GetRelationMatrix(g, rels[i], false);

		GrB_Matrix M;
		info = Delta_Matrix_export(&M, D);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_Matrix_eWiseAdd_Semiring(A, NULL, NULL, GxB_ANY_PAIR_BOOL,
				A, M, NULL);
		ASSERT(info == GrB_SUCCESS);

		GrB_Matrix_free(&M);
	}

	info = GrB_Matrix_nrows(&nrows, A);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols, A);
	ASSERT(info == GrB_SUCCESS);

	// expecting a square matrix
	ASSERT(nrows == ncols);

	// create vector N denoting all nodes participating in the algorithm
	info = GrB_Vector_new(N, GrB_BOOL, nrows);
	ASSERT(info == GrB_SUCCESS);

	// enforce labels
	if(n_lbls > 0) {
		Delta_Matrix DL = Graph_GetLabelMatrix(g, lbls[0]);

		GrB_Matrix L;
		info = Delta_Matrix_export(&L, DL);
		ASSERT(info == GrB_SUCCESS);

		// L = L U M
		for(unsigned short i = 1; i < n_lbls; i++) {
			DL = Graph_GetLabelMatrix(g, lbls[i]);

			GrB_Matrix M;
			info = Delta_Matrix_export(&M, DL);
			ASSERT(info == GrB_SUCCESS);

			info = GrB_Matrix_eWiseAdd_Semiring(L, NULL, NULL,
					GxB_ANY_PAIR_BOOL, L, M, NULL);
			ASSERT(info == GrB_SUCCESS);

			GrB_Matrix_free(&M);
		}

		// A = L * A * L
		info = GrB_mxm(A, NULL, NULL, GxB_ANY_PAIR_BOOL, L, A, NULL);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_mxm(A, NULL, NULL, GxB_ANY_PAIR_BOOL, A, L, NULL);
		ASSERT(info == GrB_SUCCESS);

		// set N to L's main diagonal denoting all participating nodes 
		info = GxB_Vector_diag(*N, L, 0, NULL);
		ASSERT(info == GrB_SUCCESS);

		// free L matrix
		info = GrB_Matrix_free(&L);
		ASSERT(info == GrB_SUCCESS);
	} else {
		// N = [1,....1]
		GrB_Scalar scalar;
		info = GrB_Scalar_new(&scalar, GrB_BOOL);
		ASSERT(info == GrB_SUCCESS);

		info = GxB_Scalar_setElement_BOOL(scalar, true);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_Vector_assign_Scalar(*N, NULL, NULL, scalar, GrB_ALL, nrows,
				NULL);
		ASSERT(info == GrB_SUCCESS);
    
		info = GrB_free(&scalar);
		ASSERT(info == GrB_SUCCESS);
	}

	return A;
}

// process procedure yield
static void _process_yield
(
	Betweenness_Context *ctx,
	const char **yield
) {
	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("node", yield[i]) == 0) {
			ctx->yield_node = ctx->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("score", yield[i]) == 0) {
			ctx->yield_score = ctx->output + idx;
			idx++;
			continue;
		}
	}
}

// process procedure configuration argument
static bool _read_config
(
	SIValue config,         // procedure configuration
	LabelID **lbls,         // [output] labels
	RelationID **rels,      // [output] relationships
	int32_t *samplingSize,  // [output] number of source vertices
	int32_t *samplingSeed	// [output] random number generator seed
) {
	// expecting configuration to be a map
	ASSERT(lbls            != NULL);
	ASSERT(rels            != NULL);
	ASSERT(samplingSize    != NULL);
	ASSERT(samplingSeed    != NULL);
	ASSERT(SI_TYPE(config) == T_MAP);

	// set outputs to NULL
	*lbls = NULL;
	*rels = NULL;
	*samplingSize = -1;
	*samplingSeed = -1;

	uint match_fields = 0;
	uint n = Map_KeyCount(config);
	if(n > 4) {
		// error config contains unknown key
		ErrorCtx_SetError("invalid betweenness configuration");
		return false;
	}

	SIValue v;
	LabelID *_lbls    = NULL;
	GraphContext *gc  = QueryCtx_GetGraphCtx();
	RelationID *_rels = NULL;

	if(MAP_GETCASEINSENSITIVE(config, "samplingSize", v)) {
		if(SI_TYPE(v) != T_INT64 || v.longval <= 0) {
			ErrorCtx_SetError("betweenness configuration, 'samplingSize' should be a positive integer");
			return false;
		}
		
		*samplingSize = v.longval;
		match_fields++;
	}

	if(MAP_GETCASEINSENSITIVE(config, "samplingSeed", v)) {
		if(SI_TYPE(v) != T_INT64) {
			ErrorCtx_SetError("betweenness configuration, 'samplingSeed' should be an integer");
			return false;
		}
		
		*samplingSeed = v.longval;
		match_fields++;
	}

	if(MAP_GETCASEINSENSITIVE(config, "nodeLabels", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("betweenness configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			// error
			ErrorCtx_SetError("betweenness configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		_lbls = array_new(LabelID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue lbl = SIArray_Get(v, i);
			const char *label = lbl.stringval;
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
			if(s == NULL) {
				ErrorCtx_SetError("betweenness configuration, unknown label %s", label);
				goto error;
			}

			LabelID lbl_id = Schema_GetID(s);
			array_append(_lbls, lbl_id);
		}
		*lbls = _lbls;

		match_fields++;
	}

	if(MAP_GETCASEINSENSITIVE(config, "relationshipTypes", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("betweenness configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			ErrorCtx_SetError("betweenness configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		_rels = array_new(RelationID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue rel = SIArray_Get(v, i);
			const char *relation = rel.stringval;
			Schema *s = GraphContext_GetSchema(gc, relation, SCHEMA_EDGE);
			if(s == NULL) {
				ErrorCtx_SetError("betweenness configuration, unknown relationship-type %s", relation);
				goto error;
			}

			RelationID rel_id = Schema_GetID(s);
			array_append(_rels, rel_id);
		}
		*rels = _rels;

		match_fields++;
	}

	if(n != match_fields) {
		ErrorCtx_SetError("betweenness configuration contains unknown key");
		goto error;
	}

	return true;

error:
	if(_lbls != NULL) {
		array_free(_lbls);
		*lbls = NULL;
	}

	if(_rels != NULL) {
		array_free(_rels);
		*rels = NULL;
	}

	return false;
}

// invoke the procedure
ProcedureResult Proc_BetweennessInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // procedure arguments
	const char **yield    // procedure outputs
) {
	// expecting a single argument

	size_t l = array_len((SIValue *)args);

	if(l > 1) return PROCEDURE_ERR;

	SIValue config;

	if(SIValue_IsNull(args[0])) {
		config = SI_Map(0);
	} else {
		config = SI_CloneValue(args[0]);
	}

	// arg0 can be either a map or NULL
	SIType t = SI_TYPE(config);
	if(t != T_MAP) {
		SIValue_Free(config);

		ErrorCtx_SetError("invalid argument to algo.betweenness");
		return PROCEDURE_ERR;
	}

	// read betweenness invoke configuration
	// {
	//	nodeLabels: ['A', 'B'],
	//	relationshipTypes: ['R'],
	//	samplingSize: 64,
	//	samplingSeed: 12
	// }

	LabelID    *lbls = NULL;
	RelationID *rels = NULL;
	int32_t samplingSize = -1;
	int32_t samplingSeed = -1;

	//--------------------------------------------------------------------------
	// load configuration map
	//--------------------------------------------------------------------------

	bool config_ok = _read_config(config, &lbls, &rels, &samplingSize,
			&samplingSeed);

	SIValue_Free(config);

	if(!config_ok) {
		return PROCEDURE_ERR;
	}

	// assign default values for missing configuration

	if(samplingSeed == -1) {
		samplingSeed = (int32_t)time(NULL);
	}

	if(samplingSize == -1) {
		samplingSize = BETWEENNESS_DEFAULT_SAMPLE_SIZE;
	}

	//--------------------------------------------------------------------------
	// setup procedure context
	//--------------------------------------------------------------------------

	Graph *g = QueryCtx_GetGraph();
	Betweenness_Context *pdata = rm_calloc(1, sizeof(Betweenness_Context));

	pdata->g = g;

	_process_yield(pdata, yield);

	// save private data
	ctx->privateData = pdata;

	// build adjacency matrix on which we'll run Betweenness Centrality
	GrB_Matrix A = _Build_Matrix(g, &pdata->nodes, lbls, array_len(lbls), rels,
			array_len(rels));

	// free build matrix inputs
	if(lbls != NULL) array_free(lbls);
	if(rels != NULL) array_free(rels);

	//--------------------------------------------------------------------------
	// build AT
	//--------------------------------------------------------------------------

	GrB_Info   info;
    GrB_Type   type;
    GrB_Index  nrows;
	GrB_Index  ncols;
	GrB_Matrix AT;

    info = GrB_Matrix_nrows(&nrows, A);
	ASSERT(info == GrB_SUCCESS);

    info = GrB_Matrix_ncols(&ncols, A);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_Matrix_type(&type, A);
	ASSERT(info == GrB_SUCCESS);

    info = GrB_Matrix_new(&AT, type, ncols, nrows);
	ASSERT(info == GrB_SUCCESS);

    info = GrB_transpose(AT, NULL, NULL, A, NULL);
	ASSERT(info == GrB_SUCCESS);

	// pick random set of source nodes
	GrB_Index *sources = _Random_Sources(AT, &samplingSize, samplingSeed);

	//--------------------------------------------------------------------------
	// run betweeness centrality
	//--------------------------------------------------------------------------

	char msg[LAGRAPH_MSG_LEN];
	LAGraph_Graph G; 

	LAGraph_New(&G, &A, LAGraph_ADJACENCY_DIRECTED, msg);
	G->AT = AT;  // set G->AT, required by the algorithm

	// execute Betweenness Centrality
	GrB_Info betweeness_res =
		LAGr_Betweenness(&pdata->centrality, G, sources, samplingSize, msg);

	// clean up algorithm inputs
	rm_free(sources);
	info = LAGraph_Delete(&G, msg);
	ASSERT(info == GrB_SUCCESS);

	if(betweeness_res != GrB_SUCCESS) {
		return PROCEDURE_ERR;
	}

	//--------------------------------------------------------------------------
	// initialize iterator
	//--------------------------------------------------------------------------

	info = GxB_Iterator_new(&pdata->it);
	ASSERT(info == GrB_SUCCESS);

	// iterate over participating nodes
	info = GxB_Vector_Iterator_attach(pdata->it, pdata->nodes, NULL);
	ASSERT(info == GrB_SUCCESS);

    pdata->info = GxB_Vector_Iterator_seek(pdata->it, 0);

	return PROCEDURE_OK;
}

// yield node and its score
// yields NULL if there are no additional nodes to return
SIValue *Proc_BetweennessStep
(
	ProcedureCtx *ctx  // procedure context
) {
	ASSERT(ctx->privateData != NULL);

	Betweenness_Context *pdata = (Betweenness_Context *)ctx->privateData;

	// retrieve node from graph
	GrB_Index node_id;
	while(pdata->info != GxB_EXHAUSTED) {
		// get current node id and its associated score
		node_id = GxB_Vector_Iterator_getIndex(pdata->it);

		if(Graph_GetNode(pdata->g, node_id, &pdata->node)) {
			break;
		}

		// move to the next entry in the components vector
		pdata->info = GxB_Vector_Iterator_next(pdata->it);
	}

	// depleted
	if(pdata->info == GxB_EXHAUSTED) {
		return NULL;
	}

	// prep for next call to Proc_BetweennessStep
	pdata->info = GxB_Vector_Iterator_next(pdata->it);

	double score;
	GrB_Info info = GrB_Vector_extractElement_FP64(&score, pdata->centrality,
			node_id);

	ASSERT(info == GrB_SUCCESS);

	//--------------------------------------------------------------------------
	// set outputs
	//--------------------------------------------------------------------------

	if(pdata->yield_node) {
		*pdata->yield_node = SI_Node(&pdata->node);
	}

	if(pdata->yield_score) {
		*pdata->yield_score = SI_DoubleVal(score);
	}

	return pdata->output;
}

ProcedureResult Proc_BetweennessFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData != NULL) {
		Betweenness_Context *pdata = ctx->privateData;

		if(pdata->it         != NULL) GrB_free(&pdata->it);
		if(pdata->nodes      != NULL) GrB_free(&pdata->nodes);
		if(pdata->centrality != NULL) GrB_free(&pdata->centrality);

		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

// CALL algo.betweenness({nodeLabels: ['Person'], relationshipTypes: ['KNOWS'],
// samplingSize:2000, samplingSeed: 10}) YIELD node, score
ProcedureCtx *Proc_BetweennessCtx(void) {
	void *privateData = NULL;

	ProcedureOutput *outputs         = array_new(ProcedureOutput, 2);
	ProcedureOutput output_node      = {.name = "node", .type = T_NODE};
	ProcedureOutput output_component = {.name = "score", .type = T_DOUBLE};

	array_append(outputs, output_node);
	array_append(outputs, output_component);

	ProcedureCtx *ctx = ProcCtxNew("algo.betweenness",
								   1,
								   outputs,
								   Proc_BetweennessStep,
								   Proc_BetweennessInvoke,
								   Proc_BetweennessFree,
								   privateData,
								   true);
	return ctx;
}

