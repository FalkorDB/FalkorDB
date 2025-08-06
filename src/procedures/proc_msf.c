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

#define _GET_ID(id, k) \
	if (id_map) {                                                  \
		int32_t _id;                                              \
		GrB_OK(GrB_Vector_extractElement_INT32(&(_id), id_map, k));\
		(id) = (NodeID) _id;                                       \
	} else {                                                       \
		(id) = (NodeID) (k);                                       \
	}

// MSF procedure context
typedef struct {
	const Graph *g;           // graph
	RelationID *relationIDs;  // edge type(s) to traverse.
	int relationCount;        // length of relationIDs.
	uint64_t idx;             // curent tree index
	AttributeID weight_prop;  // weight attribute id
	SIValue output[2];        // array with up to 2 entries [edge, weight]
	SIValue *yield_edges;     // edges
	SIValue *yield_nodes;     // nodes
	Edge **tree_list;         // trees in each forest
	NodeID **tree_nodes;      // trees in each forest
} MSF_Context;

// process procedure yield
static void _process_yield
(
	MSF_Context *ctx,
	const char **yield
) {
	int idx = 0;
	for(uint i = 0; i < array_len(yield); i++) {
		if(strcasecmp("edges", yield[i]) == 0) {
			ctx->yield_edges = ctx->output + idx;
			idx++;
			continue;
		}

		if(strcasecmp("nodes", yield[i]) == 0) {
			ctx->yield_nodes = ctx->output + idx;
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

// get the edges and nodes of the trees in the forest into buckets
void _get_trees_from_matrix (
	Edge ***trees_e,        // [output] tree edges
	NodeID ***trees_n,      // [output] tree nodes
	const GrB_Matrix A,     // tree matrix with edge id entries
	const GrB_Vector rows,  // the rows involved in the tree
	const Graph *g,         // graph
	uint64_t *cc,           // array of representatives
	GrB_Index cc_nvals      // number of representatives
) {
	GrB_Index nrows;    // number of rows in A
	GrB_Info  it_info;  // iterator info
	GxB_Container cont     = NULL;  // container to unpack rows
	int32_t       sparcity = 0;     // sparicty of rows vector
	GrB_Vector    id_map   = NULL;  // map from row index to node id

	bool ret_edges = (trees_e != NULL);
	bool ret_nodes = (trees_n != NULL);

	GrB_OK(GrB_Matrix_nrows(&nrows, A));
	ASSERT(nrows == cc_nvals);

	GrB_OK(GrB_Vector_nvals(&nrows, rows));
	ASSERT(nrows == cc_nvals);

	GrB_OK(GrB_Vector_get_INT32(rows, &sparcity, GxB_SPARSITY_STATUS));
	
	// if the rows vector is full, we don't need to build an id_map
	// if it is bitmap we make it sparse and unload it to access the id values
	// in cont->i.
	switch (sparcity)
	{
		case GxB_BITMAP:
			GrB_OK(GrB_Vector_set_INT32(rows, GxB_SPARSE, GxB_SPARSITY_CONTROL));
			// fall through

		case GxB_SPARSE:
			GrB_OK(GxB_Container_new(&cont));
			GrB_OK(GxB_unload_Vector_into_Container(rows, cont, NULL));
			id_map = cont->i;
			break;
		
		case GxB_FULL:
			// if the vector is full, we can just use the index
			id_map = NULL;
			break;

		default:
			ASSERT(false && "Unexpected vector sparcity");
			break;
	}

	GrB_Type ty;
	if(id_map != NULL) {
		// not expecting node count to go over 2^31
		GrB_OK(GxB_Vector_type(&ty, id_map));
		ASSERT(ty == GrB_INT32);
	}

	GrB_OK(GxB_Matrix_type(&ty, A));
	ASSERT(ty == GrB_UINT64);
	NodeID **_trees_n = array_new(NodeID *, 0);

	// this loop changes the ccs from pointing to a representative to pointing
	// to a place in _trees_n. The first node we encounter will have its whole 
	// component in _trees_n[0], the next node with a different component will
	// be placed in _trees_n[1], and so on.
	// the MSB is used to mark places that already point to the _trees_n array
	for(uint k = 0; k < cc_nvals; k++) {
		uint64_t rep = cc[k];
		NodeID id; // the id of node k in the graph

		_GET_ID(id, k);
		ASSERT(id < Graph_UncompactedNodeCount(g));

		// if k is already a representative, put into array and skip.
		if (rep & MSB_MASK) {
			array_append(_trees_n[CLEAR_MSB(rep)], id);
			continue;
		} 

		uint64_t grand_rep = cc[rep];

		// if its our first time seeing this representative, make a new array
		// and set the representative to point to it.
		if ((grand_rep & MSB_MASK) == 0) {
			grand_rep = cc[rep] = SET_MSB((uint64_t) array_len(_trees_n));
			array_append(_trees_n, array_new(NodeID, 1));
		} 

		ASSERT(grand_rep & MSB_MASK);
		ASSERT(CLEAR_MSB(grand_rep) < array_len(_trees_n));

		// to make the next loop life easier, have each entry point to its
		// coresponding tree index
		cc[k] = grand_rep;

		array_append(_trees_n[CLEAR_MSB(grand_rep)], id);
	}

	// The following code gets the branches of the trees. 
	// Skip if edges are not requested. Although this would not be common.
	if(ret_edges) {
		// trees_e has the same dimensions as trees_n, except with one less entry
		// per tree.
		int n_trees = array_len(_trees_n);
		Edge **_trees_e = array_new(Edge *, n_trees);
		for(uint k = 0; k < n_trees; k++) {
			array_append(_trees_e, array_new(Edge, array_len(_trees_n[k]) - 1));
		}

		struct GB_Iterator_opaque _i;  // stack allocated iterator
		GxB_Iterator  i        = &_i;  // iterates over the MSF matrix

		// populate the _trees_e arrays by looping through the MSF matrix 
		GrB_OK (GxB_Matrix_Iterator_attach(i, A, NULL));
		it_info = GxB_Matrix_Iterator_seek(i, 0);
		
		// iterate over the edges and place them into the correct tree
		while(it_info == GrB_SUCCESS) {
			GrB_Index r;  // row index
			GrB_Index c;  // col index

			// get the row and column indices (the nodes of the edge)
			GxB_Matrix_Iterator_getIndex(i, &r, &c);

			// get node ids from row and column index
			NodeID src_id;
			NodeID dest_id;

			_GET_ID(src_id, r);
			_GET_ID(dest_id, c);

			// Get the tree and append the edge to it.
			ASSERT(cc[r] == cc[c]); // src and dest need to be in the same tree

			uint64_t j = CLEAR_MSB(cc[r]); // tree index
			Edge **tree = _trees_e + j;

			// e points to the edge at the end of tree
			Edge *e = array_ensure_tail(tree, Edge);
			EdgeID e_id = GxB_Iterator_get_UINT64(i);

			// get the edge from the graph and set its source and destination
			// wait until yield to set the relation ID
			Graph_GetEdge(g, e_id, e);
			Edge_SetSrcNodeID(e, src_id);
			Edge_SetDestNodeID(e, dest_id);

			it_info = GxB_Matrix_Iterator_next(i);
		}

		// all done set the edge array(s) output
		ASSERT(it_info == GxB_EXHAUSTED);
		*trees_e = _trees_e;
	}

	// set outputs
	if(ret_nodes) {
		*trees_n = _trees_n;
	} else {
		// free _trees_n if not requested
		for(uint i = 0; i < array_len(_trees_n); i++) {
			array_free(_trees_n[i]);
		}
		array_free(_trees_n);
	}

	// don't free the id_map, it is in the container.
	id_map = NULL;
	if(cont != NULL) {
		GrB_OK(GxB_load_Vector_from_Container(rows, cont, NULL));
		GrB_OK(GrB_free(&cont));
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

	if(l > 1) {
		ErrorCtx_SetError("algo.MSF expects a single argument");
		return PROCEDURE_ERR;
	}

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

	int handle;        // cc's memory ownership
	GrB_Type cc_t;     // type of cc e.g. GrB_UINT64
	uint64_t cc_n;     // number of entries in cc
	uint64_t cc_size;  // cc size in bytes

	GrB_Matrix A       = NULL;  // edge ids of filtered edges
	GrB_Matrix A_w     = NULL;  // weight of filtered edges
	GrB_Vector cc      = NULL;  // connected components (trees)
	GrB_Vector rows    = NULL;  // nodes involved in the procedure
	uint64_t   *cc_arr = NULL;  // content cc

	// build input matrix
	GrB_OK (get_sub_weight_matrix(&A, &A_w, &rows, g, lbls, array_len(lbls), 
		rels, array_len(rels), weightAtt, maxST ? BWM_MAX : BWM_MIN, true));
	
	// free build matrix inputs
	if (lbls != NULL) array_free(lbls);

	//--------------------------------------------------------------------------
	// run MSF
	//--------------------------------------------------------------------------

	if (maxST) { // if we are optimizing for the max, make weights negative
		GrB_OK (GrB_Matrix_apply(A_w, NULL, NULL, GrB_AINV_FP64, A_w, NULL));
	}

	// execute Minimum Spanning Forest
	GrB_Matrix w_forest = NULL;
	char msg[LAGRAPH_MSG_LEN];
	GrB_Info msf_res = LAGraph_msf(&w_forest, &cc, A_w, false, msg);

	// clean up algorithm inputs
	GrB_OK (GrB_free(&A_w));

	if (msf_res != GrB_SUCCESS) {
		GrB_free(&A);
		return PROCEDURE_ERR;
	}

	GrB_Index n;
	GrB_OK (GrB_Matrix_nrows(&n, w_forest));

	// mask out dropped edges
	GrB_OK (GrB_Matrix_assign(A, w_forest, NULL, A, GrB_ALL, n, GrB_ALL,
			n, GrB_DESC_RS));
	GrB_free(&w_forest);
	
	//--------------------------------------------------------------------------
	// bucket nodes & edges
	//--------------------------------------------------------------------------

	GrB_OK (GxB_Vector_unload(cc, (void **) &cc_arr, &cc_t, &cc_n, 
		&cc_size, &handle, NULL));
	ASSERT(handle == GrB_DEFAULT);
	ASSERT(cc_t == GrB_UINT64);

	_get_trees_from_matrix(
		pdata->yield_edges != NULL ? &pdata->tree_list : NULL, 
		pdata->yield_nodes != NULL ? &pdata->tree_nodes : NULL,  
		A, rows, pdata->g, cc_arr, cc_n);

	GrB_OK (GxB_Vector_load(cc, (void **) &cc_arr, cc_t, cc_n, 
		cc_size, handle, NULL));

	GrB_OK (GrB_free(&A));
	GrB_OK (GrB_free(&cc));
	GrB_OK (GrB_free(&rows));

	ASSERT(pdata->tree_list != NULL || pdata->tree_nodes != NULL);
	if(pdata->yield_edges != NULL && pdata->yield_nodes != NULL) {
		ASSERT(array_len(pdata->tree_list) == array_len(pdata->tree_nodes));
	}
	
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

	uint32_t n = pdata->tree_list? 
		array_len(pdata->tree_list) : array_len(pdata->tree_nodes); 
	// depleted
	if(pdata->idx >= n) {
		return NULL;
	}
	
	//--------------------------------------------------------------------------
	// set outputs
	//--------------------------------------------------------------------------

	if (pdata->yield_edges != NULL) {
		Edge *tree_e = pdata->tree_list[pdata->idx];
		uint len = array_len(tree_e);
		*pdata->yield_edges = SI_Array(len);
		for (uint i = 0; i < len; i++) {
			Edge *e = &tree_e[i];
			
			// look up and set edge relation ID
			bool found_e = Graph_LookupEdgeRelationID(pdata->g, e, 
				pdata->relationIDs, pdata->relationCount) ;
			
			if (!found_e) {
				// msf thinks the graph is symetric, so it could have returned
				// a flipped edge
				EdgeID temp = Edge_GetSrcNodeID(e);
				Edge_SetSrcNodeID(e, Edge_GetDestNodeID(e));
				Edge_SetDestNodeID(e, temp);

				found_e = Graph_LookupEdgeRelationID(pdata->g, e, 
					pdata->relationIDs, pdata->relationCount);
			}
			ASSERT(found_e);

			SIArray_Append(pdata->yield_edges, SI_Edge(e));
		}
	}	

	if (pdata->yield_nodes != NULL) {
		NodeID *tree_n = pdata->tree_nodes[pdata->idx];
		uint len = array_len(tree_n);
		*pdata->yield_nodes = SI_Array(len);
		for (uint i = 0; i < len; i++) {
			Node n = GE_NEW_NODE();
			bool node_found = Graph_GetNode(pdata->g, tree_n[i], &n);
			ASSERT(node_found == true);

			SIValue v = SI_Node(&n);
			SIArray_Append(pdata->yield_nodes, v);
		}
	}

	// prepare for next step
	pdata->idx++;
	return pdata->output;
}

ProcedureResult Proc_MSFFree
(
	ProcedureCtx *ctx
) {
	// clean up
	if(ctx->privateData != NULL) {
		MSF_Context *pdata = ctx->privateData;
		
		if(pdata->tree_list != NULL) {
			for(uint i = 0; i < array_len(pdata->tree_list); i++) {
				array_free(pdata->tree_list[i]);
			}
			array_free(pdata->tree_list);
		}

		if(pdata->tree_nodes != NULL) {
			for(uint i = 0; i < array_len(pdata->tree_nodes); i++) {
				array_free(pdata->tree_nodes[i]);
			}
			array_free(pdata->tree_nodes);
		}

		array_free(pdata->relationIDs);

		rm_free(ctx->privateData);
	}

	return PROCEDURE_OK;
}

// CALL algo.MSF({
//     nodeLabels:         ['Person'], 
//     relationshipTypes:  ['KNOWS'],
//     weightAttribute:     'Years', 
//     objective:           'Minimize'
// }) 
// YIELD edges, nodes
ProcedureCtx *Proc_MSFCtx(void) {
	ProcedureOutput *outputs      = array_new(ProcedureOutput, 2);
	ProcedureOutput output_edge   = {.name = "edges", .type = T_ARRAY};
	ProcedureOutput output_weight = {.name = "nodes", .type = T_ARRAY};

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

