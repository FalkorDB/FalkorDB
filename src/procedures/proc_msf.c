/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "LAGraphX.h"
#include "GraphBLAS.h"

#include "proc_msf.h"
#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"
#include "./utility/internal.h"
#include "../graph/graphcontext.h"

// MSF invoke examples:
//
// CALL algo.MSF() YIELD edge, weight
// CALL algo.MSF(NULL) YIELD edge, weight
// CALL algo.MSF({nodeLabels: ['L', 'P']}) YIELD edge, weight
// CALL algo.MSF({relationshipTypes: ['R', 'E']}) YIELD edge, weight
// CALL algo.MSF({nodeLabels: ['L'], relationshipTypes: ['E'], weightAttribute: 
//      'cost', objective: 'maximize'}) YIELD edge, weight
// CALL algo.MSF({nodeLabels: ['L'], objective: 'minimize'})


// MSF procedure context
typedef struct {
	const Graph *g;           // graph
	GrB_Matrix forest;        // The MSF
	GrB_Matrix w_forest;      // The weighted MSF
	RelationID *relationIDs;  // edge type(s) to traverse.
	int relationCount;        // length of relationIDs.
	GrB_Info info;            // iterator state
	GxB_Iterator it;          // edge iterator
	GxB_Iterator weight_it;   // weight iterator
	Edge edge;                // edge
	AttributeID weight_prop;  // weight attribute id
	SIValue output[2];        // array with up to 2 entries [edge, weight]
	SIValue *yield_edge;      // edges
	SIValue *yield_weight;    // edge weights
	Edge **tree_list;         // trees in each forest
	uint64_t *cc;             // trees in each forest
} MSF_Context;

// process procedure yield
static void _process_yield
(
	MSF_Context *ctx,
	const char **yield
) {
	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("edge", yield[i]) == 0) {
			ctx->yield_edge = ctx->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("weight", yield[i]) == 0) {
			ctx->yield_weight = ctx->output + idx;
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
	AttributeID *weightAtt, // [output] relationship used as weight
	bool *maxST             // [output] true if maximum spanning forest
) {
	ASSERT(lbls            != NULL);
	ASSERT(rels            != NULL);
	ASSERT(maxST           != NULL);
	ASSERT(weightAtt       != NULL);
	ASSERT(SI_TYPE(config) == T_MAP);  // expecting configuration to be a map

	// initialize outputs
	*lbls      = NULL;
	*rels      = NULL;
	*maxST     = false;
	*weightAtt = ATTRIBUTE_ID_NONE;

	uint match_fields = 0;
	uint n = Map_KeyCount(config);
	if(n > 4) {
		// error config contains unknown key
		ErrorCtx_SetError("invalid msf configuration");
		return false;
	}

	SIValue v;
	LabelID *_lbls    = NULL;
	GraphContext *gc  = QueryCtx_GetGraphCtx();
	RelationID *_rels = NULL;

	//--------------------------------------------------------------------------
	// read labels
	//--------------------------------------------------------------------------

	if(MAP_GETCASEINSENSITIVE(config, "nodeLabels", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("msf configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			// error
			ErrorCtx_SetError("msf configuration, 'nodeLabels' should be an array of strings");
			goto error;
		}

		_lbls = array_new(LabelID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue lbl = SIArray_Get(v, i);
			const char *label = lbl.stringval;
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
			if(s == NULL) {
				// error
				ErrorCtx_SetError(
					"msf configuration contains non-existent label:%s", label);
				goto error;
			}

			LabelID lbl_id = Schema_GetID(s);
			array_append(_lbls, lbl_id);
		}
		*lbls = _lbls;

		match_fields++;
	}

	//--------------------------------------------------------------------------
	// read relationship-types
	//--------------------------------------------------------------------------

	if(MAP_GETCASEINSENSITIVE(config, "relationshipTypes", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("msf configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			ErrorCtx_SetError("msf configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		_rels = array_new(RelationID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue rel = SIArray_Get(v, i);
			const char *relation = rel.stringval;
			Schema *s = GraphContext_GetSchema(gc, relation, SCHEMA_EDGE);
			if(s == NULL) {
				// error
				ErrorCtx_SetError(
					"msf configuration contains non-existent type:%s", relation);
				goto error;
			}

			RelationID rel_id = Schema_GetID(s);
			array_append(_rels, rel_id);
		}
		*rels = _rels;

		match_fields++;
	}

	//--------------------------------------------------------------------------
	// read weight attribute
	//--------------------------------------------------------------------------

	if(MAP_GETCASEINSENSITIVE(config, "weightAttribute", v)) {
		if(SI_TYPE(v) != T_STRING) {
			ErrorCtx_SetError("msf configuration, 'weightAttribute' should be a string");
			goto error;
		}

		const char *attr = v.stringval;
		*weightAtt = GraphContext_GetAttributeID(gc, attr);
		if(*weightAtt == ATTRIBUTE_ID_NONE) {
			ErrorCtx_SetError("msf configuration, unknown attribute: %s", attr);
			goto error;
		}
		match_fields++;
	}

	//--------------------------------------------------------------------------
	// read objective (min/max)
	//--------------------------------------------------------------------------

	if(MAP_GETCASEINSENSITIVE(config, "objective", v)) {
		if(SI_TYPE(v) != T_STRING) {
			ErrorCtx_SetError("msf configuration, 'objective' should be a string");
			goto error;
		}

		const char *objective = v.stringval;
		if (strcasecmp(objective, "minimize") == 0) {
			*maxST = false;
		} else if (strcasecmp(objective, "maximize") == 0) {
			*maxST = true;
		} else {
			ErrorCtx_SetError("msf configuration, unknown objective: %s", objective);
			goto error;
		}

		match_fields++;
	}

	// validate no unknown configuration fields
	if(n != match_fields) {
		ErrorCtx_SetError("msf configuration contains unknown key");
		goto error;
	}

	return true;

error:
	// clean up

	if (_lbls != NULL) {
		array_free(_lbls);
		*lbls = NULL;
	}

	if (_rels != NULL) {
		array_free(_rels);
		*rels = NULL;
	}

	return false;
}

void _get_trees_from_matrix(
	Edge ***trees_e,
	NodeID ***trees_n,
	const GrB_Matrix A,
	uint64_t *cc,
	GrB_Index cc_nvals
) {
	struct GB_Iterator_opaque _i;
	GrB_Index nrows;
	GrB_Index r;
	GrB_Index c;
	GrB_Info  it_info;
	GrB_OK(GrB_Matrix_nrows(&nrows, A));
#if RG_DEBUG
	GrB_Type ty;
	GxB_Matrix_type(&ty, A);
	ASSERT(ty == GrB_UINT64);
#endif
	NodeID **_trees_n = array_new(NodeID *, 0);

	for(uint k = 0; k < cc_nvals; k++) {
		// if k is not included in the forest (UINT64_MAX), skip
		if (cc[k] & MSB_MASK) continue;

		uint64_t grand_rep = cc[cc[k]];
		if ((grand_rep & MSB_MASK) == 0) {
			grand_rep = cc[cc[k]] = SET_MSB((uint64_t) array_len(_trees_n));
			array_append(_trees_n, array_new(NodeID, 1));
			array_append(array_tail(_trees_n), k); 
		} 
		cc[k] = grand_rep;

		ASSERT(cc[k] & MSB_MASK);
		ASSERT(CLEAR_MSB(cc[k]) < array_len(_trees_n));

		array_append(_trees_n[CLEAR_MSB(cc[k])], k);
	}

	Edge **_trees_e = array_new(Edge *, array_len(_trees_n));
	for(uint k = 0; k < array_len(_trees_n); k++) {
		array_append(_trees_e, array_new(Edge, array_len(_trees_n[k])));
	}

	GxB_Iterator i = &_i;
	GrB_OK(GxB_Matrix_Iterator_attach(i, A, NULL));
	it_info = GxB_Matrix_Iterator_seek(i, 0);
	
	while(it_info == GrB_SUCCESS) {
		GxB_Matrix_Iterator_getIndex(i, &r, &c);
		uint64_t j = CLEAR_MSB(cc[r]);
		Edge **tree = &_trees_e[j];

		// e points to a newly allocated edge at the end of tree
		*tree = array_grow(*tree, 1);
		Edge *e = &array_tail(*tree);
		EdgeID e_id = GxB_Iterator_get_UINT64(i);

		Edge_SetSrcNodeID(e, r);
		Edge_SetDestNodeID(e, c);
		e->id = e_id;

		it_info = GxB_Matrix_Iterator_next(i);
	}

	ASSERT(it_info == GxB_EXHAUSTED);
	if(trees_n != NULL) {
		*trees_n = _trees_n;
	} else {
		array_free(_trees_n);
	}

	if(trees_e != NULL) {
		*trees_e = _trees_e;
	} else {
		array_free(_trees_e);
	}
}

// invoke the procedure
ProcedureResult Proc_MSFInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // procedure arguments
	const char **yield    // procedure outputs
) {
	// expecting a single argument
	size_t l = array_len((SIValue *) args);

	if(l > 1) return PROCEDURE_ERR;

	SIValue config;

	// empty config map incase one wasn't provided
	if(l == 0 || SIValue_IsNull(args[0])) {
		config = SI_Map(0);
	} else {
		config = SI_CloneValue(args[0]);
	}

	// arg0 can be either a map or NULL
	SIType t = SI_TYPE(config);
	if(t != T_MAP) {
		SIValue_Free(config);

		ErrorCtx_SetError("invalid argument to algo.MSF");
		return PROCEDURE_ERR;
	}

	// read MSF invoke configuration
	// {
	//	nodeLabels: ['A', 'B'],
	//	relationshipTypes: ['R'],
	//	weightAttribute: 'score',
	//	objective: 'minimize'
	// }

	LabelID    *lbls      = NULL;   // filtered labels
	RelationID *rels      = NULL;   // filtered relationships
	bool        maxST     = false;  // true if objective is 'maximize'
	AttributeID weightAtt = ATTRIBUTE_ID_NONE;

	//--------------------------------------------------------------------------
	// load configuration map
	//--------------------------------------------------------------------------

	bool config_ok = _read_config(config, &lbls, &rels, &weightAtt, &maxST);
	
	SIValue_Free(config);

	if(!config_ok) {
		//error set by _read_config
		return PROCEDURE_ERR;
	}

	//--------------------------------------------------------------------------
	// setup procedure context
	//--------------------------------------------------------------------------

	Graph *g = QueryCtx_GetGraph();
	MSF_Context *pdata = rm_calloc(1, sizeof(MSF_Context));

	pdata->g             = g;
	pdata->weight_prop   = weightAtt;
	pdata->relationIDs   = rels;
	pdata->relationCount = array_len(rels);

	_process_yield(pdata, yield);

	// save private data
	ctx->privateData = pdata;

	//--------------------------------------------------------------------------
	// construct input matrix
	//--------------------------------------------------------------------------

	GrB_Matrix A     = NULL;  // edge ids of filtered edges
	GrB_Matrix A_w   = NULL;  // weight of filtered edges
	GrB_Vector cc    = NULL;
	GrB_Vector rows  = NULL;
	uint64_t cc_size;
	GrB_Type cc_t;
	uint64_t cc_n;
	int handle;

	// build input matrix
	GrB_OK (Build_Weighted_Matrix(&A, &A_w, &rows, g, lbls, array_len(lbls), rels,
			array_len(rels), weightAtt, maxST ? BWM_MAX : BWM_MIN, true, true));
	
	// free build matrix inputs
	if (lbls != NULL) array_free(lbls);

	//--------------------------------------------------------------------------
	// run MSF
	//--------------------------------------------------------------------------

	if (maxST) { // if we are optimizing for the max, make weights negative
		GrB_OK (GrB_Matrix_apply(A_w, NULL, NULL, GrB_AINV_FP64, A_w, NULL));
	}

	// execute Minimum Spanning Forest
	char msg[LAGRAPH_MSG_LEN];
	GrB_Info msf_res = LAGraph_msf(&pdata->w_forest, &cc, A_w, false, msg);

	// clean up algorithm inputs
	GrB_OK (GrB_free(&A_w));

	if (msf_res != GrB_SUCCESS) {
		GrB_free(&A);
		return PROCEDURE_ERR;
	}

	// negate weights again if maximizing
	if (maxST && pdata->yield_weight != NULL) {
		GrB_OK (GrB_Matrix_apply(pdata->w_forest, NULL, NULL, GrB_AINV_FP64,
				pdata->w_forest, NULL));
	}

	GrB_Index n;
	GrB_OK (GrB_Matrix_nrows(&n, pdata->w_forest));

	// mask out dropped edges
	GrB_OK (GrB_Matrix_assign(A, pdata->w_forest, NULL, A, GrB_ALL, n, GrB_ALL,
			n, GrB_DESC_RS));

	pdata->forest = A;
	
	//--------------------------------------------------------------------------
	// initialize iterators
	//--------------------------------------------------------------------------

	if (pdata->yield_weight) {
		GrB_OK (GxB_Iterator_new(&pdata->weight_it));

		GrB_OK (GxB_Matrix_Iterator_attach(pdata->weight_it, pdata->w_forest,
				NULL));

		pdata->info = GxB_Matrix_Iterator_seek(pdata->weight_it, 0);
	} else {
		// no need for the weight matrix
		GrB_free(&pdata->w_forest);
	}
	
	if (pdata->yield_edge) {
		GrB_OK (GxB_Iterator_new(&pdata->it));
		
		GrB_OK (GxB_Matrix_Iterator_attach(pdata->it, pdata->forest, NULL));

		pdata->info = GxB_Matrix_Iterator_seek(pdata->it, 0);
	}
	// set unused node ids to UINT64_MAX
	GrB_OK (GrB_Vector_assign_UINT64(cc, rows, NULL, UINT64_MAX, GrB_ALL, n,
			GrB_DESC_SC));
	GrB_OK (GxB_Vector_unload(cc, (void *) &pdata->cc, &cc_t, &cc_n, 
		&cc_size, &handle, NULL))	
	ASSERT(handle == GrB_DEFAULT);
	ASSERT(cc_t == GrB_UINT64); 
	Edge   **trees   = NULL;
	NodeID **trees_n = NULL;
	_get_trees_from_matrix(&trees, &trees_n, A, pdata->cc, cc_n);

	int tree_count = array_len(trees);
	for (size_t i = 0; i < tree_count; i++) {
		Edge *tree = trees[i];
		NodeID *tree_n = trees_n[i];
		printf("Tree %zu:\n", i);
		for (size_t j = 0; j < array_len(tree); j++) {
			Edge e = tree[j];
			printf("  Edge: src=%lu, dest=%lu, id=%lu\n", e.src_id, e.dest_id, e.id);
		}
		printf("  Nodes: ");
		for (size_t j = 0; j < array_len(tree_n); j++) {
			printf("%lu, ", tree_n[j]);
		}
		printf("\n");
		array_free(tree);
		array_free(tree_n);
		trees[i] = NULL;
		trees_n[i] = NULL;
	}

	array_free(trees);
	GrB_OK (GrB_free(&cc));
	GrB_OK (GrB_free(&rows));
	return PROCEDURE_OK;
}

// yield edge and its weight
// yields NULL if there are no additional edges to return
SIValue *Proc_MSFStep
(
	ProcedureCtx *ctx  // procedure context
) {
	ASSERT(ctx->privateData != NULL);

	MSF_Context *pdata = (MSF_Context *) ctx->privateData;

	// depleted
	if(pdata->info == GxB_EXHAUSTED) {
		return NULL;
	}

	//--------------------------------------------------------------------------
	// set outputs
	//--------------------------------------------------------------------------

	if (pdata->yield_edge) {
		// retrieve node from graph
		EdgeID edgeID = (EdgeID) GxB_Iterator_get_UINT64(pdata->it);
		ASSERT(SCALAR_ENTRY(edgeID));

		GrB_Index node_i;
		GrB_Index node_j;
		GxB_Matrix_Iterator_getIndex(pdata->it, &node_i, &node_j);

		bool edge_flag = Graph_GetEdge(pdata->g, edgeID, &pdata->edge);
		ASSERT(edge_flag);

		// initialize edge
		Edge_SetSrcNodeID(&pdata->edge,  node_i);
		Edge_SetDestNodeID(&pdata->edge, node_j);
		Edge_SetRelationID(&pdata->edge, GRAPH_UNKNOWN_RELATION);

		bool foundRel = Graph_LookupEdgeRelationID(pdata->g, &pdata->edge,
				pdata->relationIDs, pdata->relationCount);
		
		// it is possible for MSF to use a reversed edge, as it is operating on
		// a symetric matrix, in such case we'll have to switch e's src and dest
		// and preform a second lookup
		if (!foundRel) {
			// switch src and dest
			Edge_SetSrcNodeID(&pdata->edge,  node_j);
			Edge_SetDestNodeID(&pdata->edge, node_i);

			foundRel = Graph_LookupEdgeRelationID(pdata->g, &pdata->edge,
					pdata->relationIDs, pdata->relationCount);
		}

		ASSERT(foundRel);
		*pdata->yield_edge = SI_Edge(&pdata->edge);

		pdata->info = GxB_Matrix_Iterator_next(pdata->it);
	}

	if (pdata->yield_weight) {
		double weight_val = 0; 

		weight_val = GxB_Iterator_get_FP64(pdata->weight_it);
		if(weight_val == INFINITY || weight_val == -INFINITY) {
			*pdata->yield_weight = SI_NullVal();
		} else {
			*pdata->yield_weight = SI_DoubleVal(weight_val);
		}

		// advance weight iterator
		pdata->info = GxB_Matrix_Iterator_next(pdata->weight_it); 
	}

	return pdata->output;
}

ProcedureResult Proc_MSFFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData != NULL) {
		MSF_Context *pdata = ctx->privateData;

		if(pdata->it        != NULL) GrB_free(&pdata->it);
		if(pdata->forest    != NULL) GrB_free(&pdata->forest);
		if(pdata->w_forest  != NULL) GrB_free(&pdata->w_forest);
		if(pdata->weight_it != NULL) GrB_free(&pdata->weight_it);

		rm_free(pdata->cc);
		rm_free(ctx->privateData);
		array_free(pdata->relationIDs);
	}

	return PROCEDURE_OK;
}

// CALL algo.MSF({
//     nodeLabels:         ['Person'], 
//     relationshipTypes:  ['KNOWS'],
//     weightAttribute:     'Years', 
//     objective:           'Minimize'
// }) 
// YIELD edge, weight
ProcedureCtx *Proc_MSFCtx(void) {
	ProcedureOutput *outputs      = array_new(ProcedureOutput, 2);
	ProcedureOutput output_edge   = {.name = "edge",   .type = T_EDGE};
	ProcedureOutput output_weight = {.name = "weight", .type = T_DOUBLE};

	array_append(outputs, output_edge);
	array_append(outputs, output_weight);

	ProcedureCtx *ctx = ProcCtxNew("algo.MSF",
								   PROCEDURE_VARIABLE_ARG_COUNT,
								   outputs,
								   Proc_MSFStep,
								   Proc_MSFInvoke,
								   Proc_MSFFree,
								   NULL,
								   true);
	return ctx;
}

