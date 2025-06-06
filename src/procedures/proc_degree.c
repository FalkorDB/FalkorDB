/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "proc_degree.h"

#include "../value.h"
#include "../util/arr.h"
#include "../query_ctx.h"
#include "../util/rmalloc.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"
#include "../algorithms/degree.h"
#include "../graph/graphcontext.h"
#include "../graph/delta_matrix/delta_matrix_iter.h"

// degree procedure context
typedef struct {
	const Graph *g;         // graph
	GrB_Info info;          // iterator info
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

static bool _read_config
(
	SIValue config,    // procedure configuration
	LabelID **src_lbls,    // [output] labels
	LabelID **dest_lbls,    // [output] labels
	RelationID **rels,  // [output] relationships
	GRAPH_EDGE_DIR *dir     // edge direction
) {
	// expecting configuration to be a map
	ASSERT(src_lbls        != NULL);
	ASSERT(dest_lbls      != NULL);
	ASSERT(rels            != NULL);
	ASSERT(SI_TYPE(config) == T_MAP);

	// set outputs to NULL
	*src_lbls 	= NULL;
	*dest_lbls 	= NULL;
	*rels 		= NULL;

	uint match_fields = 0;
	uint n = Map_KeyCount(config);
	if(n > 4) {
		// error config contains unknown key
		ErrorCtx_SetError("invalid degree configuration");
		return false;
	}

	SIValue v;
	GraphContext *gc  = QueryCtx_GetGraphCtx();
	RelationID *_rels    = NULL;
	LabelID *_slbls = NULL;
	LabelID *_dlbls = NULL;
	// TODO: make case insensitive.
	if(MAP_GET(config, "srcLabels", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("degree configuration, 'srcLabels' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			// error
			ErrorCtx_SetError("degree configuration, 'srcLabels' should be an array of strings");
			goto error;
		}

		_slbls = array_new(LabelID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue lbl = SIArray_Get(v, i);
			const char *label = lbl.stringval;
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
			if(s == NULL) {
				// ignore non-existing labels
				continue;
			}

			LabelID lbl_id = Schema_GetID(s);
			array_append(_slbls, lbl_id);
		}
		*src_lbls = _slbls;

		match_fields++;
	}
	if(MAP_GET(config, "destLabels", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("degree configuration, 'destLabels' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			// error
			ErrorCtx_SetError("degree configuration, 'destLabels' should be an array of strings");
			goto error;
		}

		_dlbls = array_new(LabelID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue lbl = SIArray_Get(v, i);
			const char *label = lbl.stringval;
			Schema *s = GraphContext_GetSchema(gc, label, SCHEMA_NODE);
			if(s == NULL) {
				// ignore non-existing labels
				continue;
			}

			LabelID lbl_id = Schema_GetID(s);
			array_append(_dlbls, lbl_id);
		}
		*dest_lbls = _dlbls;

		match_fields++;
	}

	if(MAP_GET(config, "relationshipTypes", v)) {
		if(SI_TYPE(v) != T_ARRAY) {
			ErrorCtx_SetError("degree configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		if(!SIArray_AllOfType(v, T_STRING)) {
			ErrorCtx_SetError("degree configuration, 'relationshipTypes' should be an array of strings");
			goto error;
		}

		_rels = array_new(RelationID, 0);
		u_int32_t l = SIArray_Length(v);
		for(u_int32_t i = 0; i < l; i++) {
			SIValue rel = SIArray_Get(v, i);
			const char *relation = rel.stringval;
			Schema *s = GraphContext_GetSchema(gc, relation, SCHEMA_EDGE);
			if(s == NULL) {
				// ignore non-existing relationships
				continue;
			}

			RelationID rel_id = Schema_GetID(s);
			array_append(_rels, rel_id);
		}
		*rels = _rels;

		match_fields++;
	}
	if(MAP_GET(config, "dir", v)) {
		if(SI_TYPE(v) == T_STRING) {
			if(strcmp(v.stringval, "incoming") == 0) {
				*dir = GRAPH_EDGE_DIR_INCOMING;
			} else if(strcmp(v.stringval, "outgoing") == 0) {
				*dir = GRAPH_EDGE_DIR_OUTGOING;
			} else {
				// unknown edge direction, fail
				ErrorCtx_SetError("Unknown direction");
				goto error;
			}
		}
		match_fields++;
	}

	if(n != match_fields) {
		ErrorCtx_SetError("wcc configuration contains unknown key");
		goto error;
	}

	return true;

error:
	if(_slbls != NULL) {
		array_free(_slbls);
		*src_lbls = NULL;
	}

	if(_dlbls != NULL) {
		array_free(_dlbls);
		*dest_lbls = NULL;
	}

	if(_rels != NULL) {
		array_free(_rels);
		*rels = NULL;
	}

	return false;
}

// procedure invoke callback
ProcedureResult Proc_DegreeInvoke
(
	ProcedureCtx *ctx,    // procedure context
	const SIValue *args,  // invocation args
	const char **yield    // output(s)
) {
	// expecting a single map argument
	SIValue config;
	size_t l = array_len((SIValue *)args);

	if(l > 1) return PROCEDURE_ERR;
	if(l == 0 || SIValue_IsNull(args[0])) {
		config = SI_Map(0);
	} else {
		config = SI_CloneValue(args[0]);
	}

	// arg0 can be either a map or NULL
	SIType t = SI_TYPE(config);
	if(!(t & T_MAP)) {
		SIValue_Free(config);

		ErrorCtx_SetError("invalid argument to algo.degree");
		return PROCEDURE_ERR;
	}
	GRAPH_EDGE_DIR dir;      		// edge direction
	LabelID *src_labels = NULL;   	// src label
	LabelID *dest_labels = NULL;   	// src label
	RelationID *rel_types = NULL;   // edge relationship type

	// parse configuration
	if(!_read_config(config, &src_labels, &dest_labels, &rel_types, &dir)) {
		// failed to parse input configuration
		// emit an error and return
		ErrorCtx_SetError(EMSG_PROC_INVALID_ARGUMENTS, "algo.degree");
		return PROCEDURE_ERR;
	}
	unsigned short  n_lbls = array_len(src_labels);
	unsigned short  n_rels = array_len(rel_types);
	GraphContext *gc        = QueryCtx_GetGraphCtx();
	Graph        *g         = GraphContext_GetGraph(gc);
	int           direction = (dir == GRAPH_EDGE_DIR_INCOMING)? DEG_INDEGREE : DEG_DEFAULT;

	Tensor		 R 		= NULL;  // relation adjacency matrix
	GrB_Info 	 info  	= GrB_SUCCESS;
	GrB_Vector 	 degree = NULL; // degree vector
	GrB_Vector 	 src 	= NULL; // src vector
	GrB_Vector 	 dest 	= NULL; // dest vector

	//--------------------------------------------------------------------------
	// get source label matrix
	//--------------------------------------------------------------------------
	info = GrB_Vector_new(&degree, GrB_UINT64, Graph_RequiredMatrixDim(g));
	ASSERT(info == GrB_SUCCESS);
	if(src_labels != NULL) {
		
		Delta_Matrix DL = Graph_GetLabelMatrix(g, src_labels[0]);

		GrB_Matrix L;
		info = Delta_Matrix_export(&L, DL, GrB_BOOL);
		ASSERT(info == GrB_SUCCESS);

		// L = L U M
		for(unsigned short i = 1; i < n_lbls; i++) {
			DL = Graph_GetLabelMatrix(g, src_labels[i]);

			GrB_Matrix M;
			info = Delta_Matrix_export(&M, DL, GrB_BOOL);
			ASSERT(info == GrB_SUCCESS);

			info = GrB_Matrix_eWiseAdd_Monoid(L, NULL, NULL,
					GxB_ANY_BOOL_MONOID, L, M, NULL);
			ASSERT(info == GrB_SUCCESS);

			GrB_Matrix_free(&M);
		}
		info = GrB_Vector_new(&src, GrB_BOOL, Graph_RequiredMatrixDim(g));
		ASSERT(info == GrB_SUCCESS);
		info = GxB_Vector_diag(src, L, 0, NULL);
		ASSERT(info == GrB_SUCCESS);
		info = GrB_Vector_assign_UINT64(
			degree, src, NULL, 0, GrB_ALL, 0, GrB_DESC_S);
		ASSERT(info == GrB_SUCCESS);

		// free L matrix
		info = GrB_Matrix_free(&L);
		ASSERT(info == GrB_SUCCESS);
		info = GrB_Vector_free(&src);
		ASSERT(info == GrB_SUCCESS);
	}
	else {
		// no source label specified, use all nodes
		info = GrB_Vector_assign_UINT64(
			degree, NULL, NULL, 0, GrB_ALL, 0, NULL);
		ASSERT(info == GrB_SUCCESS);
	}
	//--------------------------------------------------------------------------
	// get destination label matrix TODO
	//--------------------------------------------------------------------------
	info = GrB_Vector_new(&dest, GrB_BOOL, Graph_RequiredMatrixDim(g));
	ASSERT(info == GrB_SUCCESS);
	if(dest_labels != NULL) {
		
		Delta_Matrix DL = Graph_GetLabelMatrix(g, dest_labels[0]);

		GrB_Matrix L;
		info = Delta_Matrix_export(&L, DL, GrB_BOOL);
		ASSERT(info == GrB_SUCCESS);

		// L = L U M
		for(unsigned short i = 1; i < n_lbls; i++) {
			DL = Graph_GetLabelMatrix(g, dest_labels[i]);

			GrB_Matrix M;
			info = Delta_Matrix_export(&M, DL, GrB_BOOL);
			ASSERT(info == GrB_SUCCESS);

			info = GrB_Matrix_eWiseAdd_Monoid(L, NULL, NULL,
					GxB_ANY_BOOL_MONOID, L, M, NULL);
			ASSERT(info == GrB_SUCCESS);

			GrB_Matrix_free(&M);
		}
		ASSERT(info == GrB_SUCCESS);
		info = GxB_Vector_diag(dest, L, 0, NULL);
		ASSERT(info == GrB_SUCCESS);

		// free L matrix
		info = GrB_Matrix_free(&L);
		ASSERT(info == GrB_SUCCESS);
	}
	else {
		// no source label specified, use all nodes
		info = GrB_Vector_assign_BOOL(
			dest, NULL, NULL, (bool) 1, GrB_ALL, 0, NULL);
		ASSERT(info == GrB_SUCCESS);
	}
	//--------------------------------------------------------------------------
	// get relationship type matrix
	//--------------------------------------------------------------------------
	if(rel_types != NULL) {
		for(unsigned short i = 0; i < n_rels; i++) {
			R = Graph_GetRelationMatrix(g, rel_types[i], false);
			int opt = Graph_RelationshipContainsMultiEdge(g, rel_types[i])?
					DEG_TENSOR : DEG_DEFAULT;
			TesorDegree(degree, dest, R, direction | opt);
		}
	} else {
		n_rels = Graph_RelationTypeCount(g);
		RelationID current;
		for(unsigned short i = 0; i < n_rels; i++) {
			R = Graph_GetRelationMatrix(g, i, false);
			int opt = Graph_RelationshipContainsMultiEdge(g, i)?
					DEG_TENSOR : DEG_DEFAULT;
			TesorDegree(degree, dest, R, direction | opt);
		}
	}
	info = GrB_Vector_resize(degree, Graph_UncompactedNodeCount(g));

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

	pdata->info = GxB_Vector_Iterator_seek(pdata->it, 0);

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
	
	// get current entry
	NodeID   id     = -1;  		// node id
	uint64_t degree = -1;       // node degree
	
	while(pdata->info != GxB_EXHAUSTED) {
		// get current node id and its associated component id
		id = GxB_Vector_Iterator_getIndex(pdata->it);
		if(Graph_GetNode(pdata->g, id, &pdata->node)) {
			degree = GxB_Iterator_get_UINT64(pdata->it);	
			break;
		}
		// move to the next entry in the components vector
		pdata->info = GxB_Vector_Iterator_next(pdata->it);
	}

	// depleted
	if(pdata->info == GxB_EXHAUSTED) {
		return NULL;
	}

	

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
	pdata->info = GxB_Vector_Iterator_next(pdata->it);

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
	ProcedureOutput output_degree = {.name = "degree", .type = T_INT64};

	array_append(outputs, output_node);
	array_append(outputs, output_degree);

	return ProcCtxNew("algo.degree", 1, outputs, Proc_DegreeStep,
			Proc_DegreeInvoke, Proc_DegreeFree, privateData, true);
}

