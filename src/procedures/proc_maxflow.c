/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// proc_max_flow.c
//
// implements the `algo.maxFlow` stored procedure
//
// given a directed weighted graph, a set of source nodes,
// and a set of sink nodes, this procedure computes the maximum flow through
// the network and returns the participating nodes, edges, and per-edge flow
//
// usage:
//   CALL algo.maxFlow({
//     sourceNodes:        [s0, s1, s2],
//     targetNodes:        [t0],
//     capacityProperty:  'cap',
//     nodeLabels:        ['Intersection'],
//     relationshipTypes: ['CONNECTS']
//   })
//   YIELD nodes, edges, edgeFlows, maxFlow

#include "RG.h"
#include "LAGraph.h"
#include "proc_ctx.h"
#include "LAGraphX.h"
#include "GraphBLAS.h"
#include "../value.h"
#include "../query_ctx.h"
#include "../datatypes/map.h"
#include "../datatypes/array.h"
#include "./utility/internal.h"
#include "../graph/graphcontext.h"

// number of recognised keys in the configuration map
#define MAX_FLOW_CONFIG_KEY_COUNT 6

//------------------------------------------------------------------------------
// context
//------------------------------------------------------------------------------

// per-call state threaded through Invoke → Step → Free
typedef struct {
	RelationID rel_id ;         // edges relationship id
	GrB_Matrix R ;              // relationship matrix for the chosen rel-type
                                // used to resolve EdgeIDs during Step

	double max_flow ;           // network's max flow
	GrB_Matrix flow_mtx ;       // result matrix from LAGr_MaxFlow
                                // set to NULL after Step has consumed it

	SIValue output[4] ;         // packed output slots [nodes, edges, flow]
	SIValue *yield_nodes ;      // pointer into output[] for the nodes slot
	SIValue *yield_edges ;      // pointer into output[] for the edges slot
	SIValue *yield_edgeFlows ;  // pointer into output[] for the edge flows slot
	SIValue *yield_maxFlow ;    // pointer into output[] for the maxflow slot
} MaxFlow_Context ;

// context passed to the GraphBLAS IndexUnaryOp that maps each matrix entry to
// its capacity value
typedef struct {
	const Graph *g;        // the graph being queried
	AttributeID attr_id;   // attribute ID that holds the capacity value
	SIType expected_type;  // accepted SIType(s) for the capacity attribute
	double default_cap;    // capacity to use when the attribute is missing
						   // or has the wrong type (default: -1, meaning
						   // procedure error on missing or wrong attribute)
	atomic_bool *invalid;  // true if matrix has invalid values
} EdgeCapacityContext ;

// GraphBLAS IndexUnaryOp callback
// reads the capacity attribute from edge (i,j) and writes it to *z.
// NOTE: the edge value stored in the matrix is passed via x, but we resolve
//       the edge by its position (i, j) through the relationship matrix
//       instead. if the attribute is absent or has an unexpected type the
//       default capacity is used
static void _get_edge_capacity
(
	double *z,                    // [output] capacity value
	const void *x,                  // entry value (EdgeID) – currently unused
	GrB_Index i,                    // row index (source node ID)
	GrB_Index j,                    // column index (destination node ID)
	const EdgeCapacityContext *ctx  // user-supplied context (theta)
) {
	Edge e ;
	bool found = Graph_GetEdge (ctx->g, *(EdgeID*)x, &e) ;
	ASSERT (found == true) ;

	SIValue v ;
	GraphEntity_GetProperty ((GraphEntity *) &e, ctx->attr_id, &v) ;

	// set z to double cast if numeric. Set to default capacity otherwise.
	int res = SIValue_ToDouble(&v, z) ;
	*z = (res == 1) ? *z : ctx->default_cap ;

	// set invalid flag if value is negative
	bool valid = *z >= 0.0 ;
	if (!valid && !(*(ctx->invalid))) {
		atomic_store(ctx->invalid, true) ;
	}
}

//------------------------------------------------------------------------------
// configuration parsing
//------------------------------------------------------------------------------

// parse the configuration map supplied to algo.maxFlow
//
// expected keys (all optional individually, but sourceNode / targetNode /
// relationshipTypes are required by the caller):
//
//   nodeLabels        – array<string>  node labels to restrict the subgraph
//   relationshipTypes – array<string>  exactly one relationship type
//   sourceNode        – node           flow source
//   targetNode        – node           flow sink
//   capacityProperty  – string         edge attribute used as capacity
//
// returns true on success
// on failure an error is set via ErrorCtx_SetError
// and *lbls / *rels are freed before returning false

static bool _read_config
(
	SIValue config,         // procedure configuration map
	LabelID **lbls,         // [output] array of label IDs (caller must free)
	RelationID **rels,      // [output] array of relation IDs (caller must free)
	Node ***srcs,           // [output] source node
	Node ***sinks,          // [output] sink node
	AttributeID *attr_id,   // [output] capacity attribute ID
	double *default_cap     // [output] default capacity (-1 = error on missing)
) {
	ASSERT (srcs        != NULL) ;
	ASSERT (lbls        != NULL) ;
	ASSERT (rels        != NULL) ;
	ASSERT (sinks       != NULL) ;
	ASSERT (attr_id     != NULL) ;
	ASSERT (default_cap != NULL) ;
	ASSERT (SI_TYPE (config) == T_MAP) ;

	// initialise outputs
	*srcs        = NULL ;
	*lbls        = NULL ;
	*rels        = NULL ;
	*sinks       = NULL ;
	*attr_id     = ATTRIBUTE_ID_NONE ;
	*default_cap = -1 ;

	uint n = Map_KeyCount (config);
	if (n > MAX_FLOW_CONFIG_KEY_COUNT) {
		ErrorCtx_SetError ("invalid MaxFlow configuration") ;
		return false ;
	}

	SIValue v;
	LabelID *_lbls    = NULL ;
	RelationID *_rels = NULL ;
	GraphContext *gc  = QueryCtx_GetGraphCtx () ;
	uint matched = 0 ;  // number of recognised keys found

	//--------------------------------------------------------------------------
	// nodeLabels – optional array of node label strings
	//--------------------------------------------------------------------------

	if (MAP_GETCASEINSENSITIVE (config, "nodeLabels", v)) {
		if (SI_TYPE (v) != T_ARRAY || !SIArray_AllOfType (v, T_STRING)) {
			ErrorCtx_SetError ("MaxFlow configuration: 'nodeLabels' "
					"must be an array of strings") ;
			goto error ;
		}

		_lbls = array_new (LabelID, 0) ;
		uint32_t l = SIArray_Length (v) ;

		for (uint32_t i = 0; i < l; i++) {
			const char *label = SIArray_Get (v, i).stringval ;
			Schema *s = GraphContext_GetSchema (gc, label, SCHEMA_NODE) ;
			if (s == NULL) {
				ErrorCtx_SetError ("MaxFlow configuration, unknown label %s",
						label) ;
				goto error ;
			}

			array_append (_lbls, Schema_GetID (s)) ;
		}

		*lbls = _lbls ;
		matched++ ;
	}

	//--------------------------------------------------------------------------
	// relationshipTypes – required array containing exactly one rel-type string
	//--------------------------------------------------------------------------

	if (MAP_GETCASEINSENSITIVE (config, "relationshipTypes", v)) {
		if (SI_TYPE (v) != T_ARRAY || !SIArray_AllOfType (v, T_STRING)) {
			ErrorCtx_SetError ("MaxFlow configuration: 'relationshipTypes' "
					"must be an array of strings") ;
			goto error ;
		}

		if (SIArray_Length (v) != 1) {
			ErrorCtx_SetError ("MaxFlow configuration, 'relationshipTypes' "
					"should contain a single string") ;
			goto error ;
		}

		_rels = array_new (RelationID, 1) ;

		const char *relation = SIArray_Get (v, 0).stringval ;
		Schema *s = GraphContext_GetSchema (gc, relation, SCHEMA_EDGE) ;
		if (s == NULL) {
			ErrorCtx_SetError ("MaxFlow configuration, "
					"unknown relationship-type %s", relation) ;
			goto error ;
		}

		array_append (_rels, Schema_GetID (s)) ;
		*rels = _rels ;
		matched++ ;
	}

	//--------------------------------------------------------------------------
	// read sourceNode
	//--------------------------------------------------------------------------

	if (MAP_GETCASEINSENSITIVE (config, "sourceNodes", v)) {
		if (SI_TYPE (v) != T_ARRAY || !SIArray_AllOfType (v, T_NODE)) {
			ErrorCtx_SetError ("MaxFlow configuration, "
					"'sourceNodes' must be an array of nodes") ;
			goto error ;
		}

		*srcs = array_new (Node*, 1) ;
		uint32_t l = SIArray_Length (v) ;
		for (uint32_t i = 0; i < l; i++) {
			SIValue *src = SIArray_GetRef (v, i) ;
			array_append (*srcs, (Node*)src->ptrval) ;
		}

		matched++ ;
	}

	//--------------------------------------------------------------------------
	// read targetNode
	//--------------------------------------------------------------------------

	if (MAP_GETCASEINSENSITIVE (config, "targetNodes", v)) {
		if (SI_TYPE (v) != T_ARRAY || !SIArray_AllOfType (v, T_NODE)) {
			ErrorCtx_SetError ("MaxFlow configuration, "
					"'targetNodes' must be an array of nodes") ;
			goto error ;
		}

		*sinks = array_new (Node*, 1) ;
		uint32_t l = SIArray_Length (v) ;
		for (uint32_t i = 0; i < l; i++) {
			SIValue *t = SIArray_GetRef (v, i) ;
			array_append (*sinks, (Node*)t->ptrval) ;
		}
		
		matched++ ;
	}

	//--------------------------------------------------------------------------
	// read capacityProperty
	//--------------------------------------------------------------------------

	if (MAP_GETCASEINSENSITIVE (config, "capacityProperty", v)) {
		if (SI_TYPE (v) != T_STRING) {
			ErrorCtx_SetError ("MaxFlow configuration, "
					"'capacityProperty' should be a string") ;
			goto error ;
		}

		AttributeID _attr_id = GraphContext_GetAttributeID (gc, v.stringval) ;
		if (_attr_id == ATTRIBUTE_ID_NONE) {
			ErrorCtx_SetError ("MaxFlow configuration, 'capacityProperty' "
					"does not exists key") ;
			goto error ;
		}
		
		*attr_id = _attr_id ;
		matched++ ;
	}

	//--------------------------------------------------------------------------
	// read defaultCapacity – optional numeric fallback for missing/wrong-type
	//--------------------------------------------------------------------------

	if (MAP_GETCASEINSENSITIVE (config, "defaultCapacity", v)) {
		double d ;
		if (SIValue_ToDouble (&v, &d) != 1 || d < 0) {
			ErrorCtx_SetError ("MaxFlow configuration, "
					"'defaultCapacity' must be a non-negative number") ;
			goto error ;
		}

		*default_cap = d ;
		matched++ ;
	}

	// make sure all fields been matched
	if (n != matched) {
		ErrorCtx_SetError ("MaxFlow configuration contains unknown key");
		goto error ;
	}

	return true ;

error:
	if (_lbls != NULL) {
		array_free (_lbls) ;
		*lbls = NULL ;
	}

	if (_rels != NULL) {
		array_free (_rels) ;
		*rels = NULL ;
	}

	return false ;
}

//------------------------------------------------------------------------------
// yield slot resolution
//------------------------------------------------------------------------------

// map each requested yield name to a pointer into the output[] array
// slots are assigned in the order the names appear so that output[] is
// packed contiguously regardless of which subset is requested

static void _process_yield
(
	MaxFlow_Context *ctx,   // procedure context to update
	const char     **yield  // null-terminated list of requested output names
) {
	int slot = 0 ;

	for (uint i = 0; i < array_len (yield); i++) {
		if (strcasecmp ("nodes", yield [i]) == 0) {
			ctx->yield_nodes = ctx->output + slot++ ;
		}

		else if (strcasecmp("edges", yield [i]) == 0) {
			ctx->yield_edges = ctx->output + slot++ ;
		}

		else if (strcasecmp ("edgeFlows", yield [i]) == 0) {
			ctx->yield_edgeFlows = ctx->output + slot++ ;
		}

		else if (strcasecmp ("maxflow", yield [i]) == 0) {
			ctx->yield_maxFlow = ctx->output + slot++ ;
		}

		else {
			ASSERT (false && "unknown yield, should have errored at "
					"generic validation phase") ;
		}
	}
}

//------------------------------------------------------------------------------
// procedure entry points
//------------------------------------------------------------------------------

// Proc_MaxFlowInvoke
//
// validates arguments, builds the subgraph adjacency matrix, applies capacity
// weights, and runs the LAGraph MaxFlow algorithm
//
// expected argument: a single configuration map (see _read_config for keys)

ProcedureResult Proc_MaxFlowInvoke
(
	ProcedureCtx    *ctx,   // procedure context (receives privateData)
	const SIValue   *args,  // procedure arguments
	const char     **yield  // requested output columns
) {
	ASSERT (ctx  != NULL) ;
	ASSERT (args != NULL) ;

	// expecting a single argument
	if (array_len ((SIValue *) args) != 1 || SI_TYPE (args[0]) != T_MAP) {
		ErrorCtx_SetError ("algo.maxFlow expects a single configuration map") ;
		return PROCEDURE_ERR ;
	}

	Graph *g            = QueryCtx_GetGraph () ;
	LabelID     *lbls   = NULL ;
	RelationID  *rels   = NULL ;
	Node        **srcs  = NULL ;
	Node        **sinks = NULL ;
	AttributeID attr_id = ATTRIBUTE_ID_NONE ;
	double  default_cap = -1 ;

	//--------------------------------------------------------------------------
	// parse and validate configuration
	//--------------------------------------------------------------------------

	ProcedureResult res = PROCEDURE_OK ;

	if (!_read_config (
		args[0], &lbls, &rels, &srcs, &sinks, &attr_id, &default_cap)) {
		// error already set
		res = PROCEDURE_ERR ;
		goto cleanup ;
	}

	if (rels == NULL || array_len (rels) != 1) {
		ErrorCtx_SetError ("algo.maxFlow: 'relationshipTypes' is required "
				"and must contain exactly one type") ;

		res = PROCEDURE_ERR ;
		goto cleanup ;
	}

	if (Graph_RelationshipContainsMultiEdge (g, rels[0])) {
		ErrorCtx_SetError ("algo.maxFlow: relationship type must not "
				"contain multi-edges (tensors)") ;

		res = PROCEDURE_ERR ;
		goto cleanup ;
	}

	if (srcs              == NULL ||
		sinks             == NULL ||
		array_len (srcs)  == 0    ||
		array_len (sinks) == 0
	) {
		ErrorCtx_SetError ("algo.maxFlow: expects at least "
				"one source and one sink") ;

		res = PROCEDURE_ERR ;
		goto cleanup ;
	}

	// validate that source and sink sets are disjoint
	uint n_srcs  = array_len (srcs) ;
	uint n_sinks = array_len (sinks) ;
	bool disjoint = true ;

	if (n_srcs == 1 && n_sinks == 1) {
		// both singletons — direct ID comparison
		disjoint = ENTITY_GET_ID (srcs[0]) != ENTITY_GET_ID (sinks[0]) ;
	} else {
		// build a boolean vector for the larger set,
		// then probe each element of the smaller set
		GrB_Index dim = Graph_RequiredMatrixDim (g) ;
		GrB_Vector v  = NULL ;
		GrB_OK (GrB_Vector_new (&v, GrB_BOOL, dim)) ;

		Node **big   = (n_srcs >= n_sinks) ? srcs  : sinks ;
		Node **small = (n_srcs >= n_sinks) ? sinks : srcs  ;
		uint n_big   = (n_srcs >= n_sinks) ? n_srcs  : n_sinks ;
		uint n_small = (n_srcs >= n_sinks) ? n_sinks : n_srcs  ;

		for (uint i = 0; i < n_big; i++) {
			GrB_OK (GrB_Vector_setElement_BOOL (v, true,
						ENTITY_GET_ID (big[i]))) ;
		}

		for (uint i = 0; i < n_small && disjoint; i++) {
			GrB_Info info = GxB_Vector_isStoredElement(
					v, ENTITY_GET_ID (small[i])) ;
			disjoint = disjoint && (info == GrB_NO_VALUE) ;
		}

		GrB_OK (GrB_free (&v)) ;
	}

	if (!disjoint) {
		ErrorCtx_SetError ("algo.maxFlow: source and sink "
				"sets must be disjoint") ;
		res = PROCEDURE_ERR ;
		goto cleanup ;
	}

	if (attr_id == ATTRIBUTE_ID_NONE) {
		ErrorCtx_SetError ("algo.maxFlow: 'capacityProperty' is required") ;

		res = PROCEDURE_ERR ;
		goto cleanup ;
	}

	//--------------------------------------------------------------------------
	// initialise per-call context
	//--------------------------------------------------------------------------

	MaxFlow_Context *pdata = rm_calloc (1, sizeof (MaxFlow_Context)) ;
	_process_yield (pdata, yield) ;

	//--------------------------------------------------------------------------
	// build the subgraph adjacency matrix A (node × node, values = EdgeIDs)
	// then transpose into U so that U(i,j) represents the edge i→j
	//--------------------------------------------------------------------------

	GrB_Matrix U ;
	Delta_Matrix R = Graph_GetRelationMatrix (g, rels[0], false) ;
	GrB_OK (Delta_Matrix_export (&U, R, GrB_UINT64)) ;

	GrB_Matrix A ;
	GrB_OK (Build_Matrix (&A, NULL, g, lbls, array_len (lbls), rels,
			array_len(rels), false, false)) ;

	//--------------------------------------------------------------------------
	// cast A to uint64 matrix
	//--------------------------------------------------------------------------

	GrB_OK (GrB_transpose (U, A, NULL, U, GrB_DESC_RST0)) ;
	GrB_OK (GrB_free (&A)) ;

	// keep the relationship matrix so Step can resolve EdgeIDs
	pdata->R = U ;

	//--------------------------------------------------------------------------
	// apply capacity weights to U via a custom IndexUnaryOp
	//--------------------------------------------------------------------------
	atomic_bool invalid_attributes = false;

	EdgeCapacityContext cap_ctx = {
		.g             = g,
		.attr_id       = attr_id,
		.expected_type = T_INT64,
		.default_cap   = default_cap,
		.invalid       = &invalid_attributes
	} ;

	GrB_Type         cap_ctx_type = NULL ;
	GrB_Scalar       cap_ctx_s    = NULL ;
	GrB_IndexUnaryOp get_capacity = NULL ;

	GrB_OK (GrB_Type_new (&cap_ctx_type, sizeof (EdgeCapacityContext))) ;
	GrB_OK (GrB_Scalar_new (&cap_ctx_s, cap_ctx_type)) ;
	GrB_OK (GrB_Scalar_setElement_UDT (cap_ctx_s, (void *) &cap_ctx)) ;

	GrB_OK (GrB_IndexUnaryOp_new (
				&get_capacity,
				(GxB_index_unary_function) _get_edge_capacity,
				GrB_FP64, GrB_UINT64, cap_ctx_type)) ;

	GrB_Matrix C ;
	GrB_Index nrows, ncols ;
	GrB_OK (GrB_Matrix_nrows (&nrows, U)) ;
	GrB_OK (GrB_Matrix_ncols (&ncols, U)) ;
	GrB_OK (GrB_Matrix_new (&C, GrB_FP64, nrows, ncols)) ;

	GrB_OK (GrB_Matrix_apply_IndexOp_Scalar (
		C, NULL, NULL, get_capacity, U, cap_ctx_s, NULL)) ;

	GrB_OK (GrB_free (&cap_ctx_s)) ;
	GrB_OK (GrB_free (&cap_ctx_type)) ;
	GrB_OK (GrB_free (&get_capacity)) ;

	if (invalid_attributes) {
		ErrorCtx_SetError ("algo.maxFlow: invalid or missing attribute and no"
			"default attribute specified") ;

		res = PROCEDURE_ERR ;
		goto cleanup ;
	}

	//--------------------------------------------------------------------------
	// accommodate for multiple sources / sinks
	//--------------------------------------------------------------------------

	NodeID src_id  = INVALID_ENTITY_ID ;
	NodeID sink_id = INVALID_ENTITY_ID ;

	bool multi_srcs  = array_len (srcs)  > 1 ;
	bool multi_sinks = array_len (sinks) > 1 ;

	if (!multi_srcs)  src_id  = ENTITY_GET_ID (srcs  [0]) ;
	if (!multi_sinks) sink_id = ENTITY_GET_ID (sinks [0]) ;

	if (multi_srcs || multi_sinks) {
		// enlarge C to make room for a new source of sources and sink of sinks
		GrB_Index nrows, ncols ;
		GrB_OK (GrB_Matrix_nrows (&nrows, C)) ;
		GrB_OK (GrB_Matrix_ncols (&ncols, C)) ;
		ASSERT (nrows == ncols) ;

		GrB_Index n = nrows ;
		if (multi_srcs)  n++ ;
		if (multi_sinks) n++ ;

		GrB_OK (GrB_Matrix_resize (C, n, n)) ;

		// FIXME: LAGraph will hang on values spread accross many orders of
		// magnitude
		double x =  INT32_MAX;

		if (multi_srcs) {
			src_id = --n ; // source of sources id
			// connect source of sources to each individual source
			int l = array_len (srcs) ;
			for (int i = 0 ; i < l ; i++) {
				Node *s = srcs[i] ;
				NodeID s_id = ENTITY_GET_ID (s) ;
				GrB_OK (GrB_Matrix_setElement_FP64 (C, x, src_id, s_id)) ;
			}
		}

		if (multi_sinks) {
			sink_id = --n ; // sink of sinks id
			// connect each sink to sink of sinks
			int l = array_len (sinks) ;
			for (int i = 0 ; i < l ; i++) {
				Node *s = sinks[i] ;
				NodeID s_id = ENTITY_GET_ID (s) ;
				GrB_OK (GrB_Matrix_setElement_FP64 (C, x, s_id, sink_id)) ;
			}
		}
	}

	ASSERT (src_id  != INVALID_ENTITY_ID) ;
	ASSERT (sink_id != INVALID_ENTITY_ID) ;


	//--------------------------------------------------------------------------
	// build the LAGraph graph (includes AT and EMin caches required by MaxFlow)
	//--------------------------------------------------------------------------

	LAGraph_Graph G ;
	char msg [LAGRAPH_MSG_LEN] ;

	GrB_OK (LAGraph_New (&G, &C, LAGraph_ADJACENCY_DIRECTED, msg)) ;
	GrB_OK (LAGraph_Cached_AT (G, msg)) ;
	GrB_OK (LAGraph_Cached_EMin (G, msg)) ;

	//--------------------------------------------------------------------------
	// run MaxFlow
	//--------------------------------------------------------------------------

	// execute maxflow
	GrB_Matrix flow_mtx = NULL ;
	GrB_Info info = LAGr_MaxFlow (&pdata->max_flow, &flow_mtx, G, src_id,
			sink_id, msg) ;

	res = (info == GrB_SUCCESS) ? PROCEDURE_OK : PROCEDURE_ERR ;

	GrB_OK (LAGraph_Delete (&G, msg)) ;

	//--------------------------------------------------------------------------
	// clear temporary edges
	//--------------------------------------------------------------------------

	if (info == GrB_SUCCESS) {
		if (multi_srcs) {
			int l = array_len (srcs) ;
			for (int i = 0 ; i < l ; i++) {
				Node *s = srcs [i] ;
				NodeID s_id = ENTITY_GET_ID (s) ;
				GrB_OK (GrB_Matrix_removeElement (flow_mtx, src_id, s_id)) ;
			}
		}

		if (multi_sinks) {
			int l = array_len (sinks) ;
			for (int i = 0 ; i < l ; i++) {
				Node *s = sinks [i] ;
				NodeID s_id = ENTITY_GET_ID (s) ;
				GrB_OK (GrB_Matrix_removeElement (flow_mtx, s_id, sink_id)) ;
			}
		}
	}

	pdata->rel_id    = rels[0] ;
	pdata->flow_mtx  = flow_mtx ;
	ctx->privateData = pdata ;

cleanup:
	if (srcs != NULL) {
		array_free (srcs)  ;
	}

	if (sinks != NULL) {
		array_free (sinks) ;
	}

	if (lbls != NULL) {
		array_free (lbls) ;
	}

	if (rels != NULL) {
		array_free (rels) ;
	}

	return res ;
}

// Proc_MaxFlowStep
//
// called once after Invoke to collect and return results
// populates the requested output slots (nodes / edges / flow) from the
// flow matrix produced by LAGr_MaxFlow, then frees the flow matrix
// returns NULL on the second call (or if Invoke failed) to signal exhaustion
SIValue *Proc_MaxFlowStep
(
	ProcedureCtx *ctx  // procedure context
) {
	ASSERT (ctx->privateData != NULL) ;

	MaxFlow_Context *pdata = (MaxFlow_Context*) ctx->privateData ;

	// flow_matrix is consumed on the first call and set to NULL; a NULL here
	// means either Invoke failed or Step has already been called
	if (pdata->flow_mtx == NULL) {
		return NULL ;
	}

	Graph     *g        = QueryCtx_GetGraph () ;
	GrB_Matrix R        = pdata->R ;
	GrB_Matrix flow_mtx = pdata->flow_mtx ;

	GrB_Index nrows ;
	GrB_OK (GrB_Matrix_nrows (&nrows, flow_mtx)) ;

	//--------------------------------------------------------------------------
	// yield nodes
	// collect unique node IDs by OR-reducing the flow matrix along both axes
	//--------------------------------------------------------------------------

	if (pdata->yield_nodes != NULL) {
		GrB_Vector unique_nodes  ;
		GrB_OK (GrB_Vector_new (&unique_nodes, GrB_BOOL, nrows)) ;

		// full vector, all 1s
		GrB_Vector x ;
		GrB_OK (GrB_Vector_new (&x, GrB_BOOL, nrows)) ;
		GrB_OK (GrB_Vector_assign_BOOL (x, NULL, NULL, 1, GrB_ALL, 0, NULL)) ;

		// rows -> nodes that appear as a source in at least one flow edge
		GrB_OK (GrB_mxv (unique_nodes, NULL, NULL, GxB_ANY_PAIR_BOOL, flow_mtx,
					x, NULL)) ;

		// columns -> nodes that appear as a destination
		GrB_OK (GrB_vxm (unique_nodes, NULL, GrB_ONEB_BOOL, GxB_ANY_PAIR_BOOL,
					x, flow_mtx, NULL)) ;

		GrB_OK (GrB_free (&x)) ;

		// determine number of unique nodes
		GrB_Index nvals ;
		GrB_OK (GrB_Vector_nvals (&nvals, unique_nodes)) ;

		// allocate node arrays
		SIValue *nodes = array_new (SIValue, nvals) ;

		// start collecting nodes
		GxB_Iterator it ;
		GrB_OK (GxB_Iterator_new (&it)) ;
		GrB_OK (GxB_Vector_Iterator_attach (it, unique_nodes, NULL)) ;

		GrB_Info info = GxB_Vector_Iterator_seek (it, 0) ;
		int counter = 0 ;
		while (info != GxB_EXHAUSTED) {
			// get the entry v(i)
			GrB_Index node_id = GxB_Vector_Iterator_getIndex (it) ;

			// fetch node
			Node *n = rm_malloc (sizeof (Node)) ;
			Graph_GetNode (g, node_id, n) ;
			array_append (nodes, SI_Node (n)) ;

			// mark as owner of memory allocation
			SIValue_SetAllocationType (nodes + counter, M_SELF) ;
			counter++ ;

			// move to the next entry in v
			info = GxB_Vector_Iterator_next (it) ;
		}

		// clean up
		GrB_OK (GrB_free (&it)) ;
		GrB_OK (GrB_free (&unique_nodes)) ;

		*pdata->yield_nodes = SIArray_FromRaw (&nodes) ;
	}

	//--------------------------------------------------------------------------
	// yield edges and per-edge flow values
	//--------------------------------------------------------------------------

	GrB_Index edge_count ;
	GrB_OK (GrB_Matrix_nvals (&edge_count, flow_mtx)) ;

	SIValue *edges = NULL ;
	SIValue *flows = NULL ;

	if (pdata->yield_edges != NULL) {
		edges = array_new (SIValue, edge_count) ;
	}

	if (pdata->yield_edgeFlows != NULL) {
		flows = array_new (SIValue, edge_count) ;
	}

	if (pdata->yield_maxFlow != NULL) {
		*pdata->yield_maxFlow = SI_DoubleVal (pdata->max_flow) ;
	}

	//--------------------------------------------------------------------------
	// initialize iterator
	//--------------------------------------------------------------------------

	if (pdata->yield_edges != NULL || pdata->yield_edgeFlows != NULL) {
		GxB_Iterator it ;
		GrB_OK (GxB_Iterator_new (&it)) ;
		GrB_OK (GxB_Matrix_Iterator_attach (it, pdata->flow_mtx, NULL)) ;

		// seek to the first entry
		GrB_Info info = GxB_Matrix_Iterator_seek (it, 0) ;
		int counter = 0 ;
		while (info != GxB_EXHAUSTED) {
			// get current iterator indices
			GrB_Index i, j ;
			GxB_Matrix_Iterator_getIndex (it, &i, &j) ;

			if (pdata->yield_edges != NULL) {
				EdgeID id ;
				GrB_OK (GrB_Matrix_extractElement_UINT64 (&id, R, i, j)) ;

				Edge *e = rm_malloc (sizeof (Edge)) ;
				bool found = Graph_GetEdge (g, id, e) ;
				ASSERT (found == true) ;

				// initialize edge
				e->src_id     = i ;
				e->dest_id    = j ;
				e->relationID = pdata->rel_id ;

				array_append (edges, SI_Edge (e)) ;

				// mark as owner of memory allocation
				SIValue_SetAllocationType (edges + counter, M_SELF) ;
				counter++ ;
			}

			if (pdata->yield_edgeFlows != NULL) {
				// get the entry A(i,j)
				double flow_val = GxB_Iterator_get_FP64 (it) ;
				array_append (flows, SI_DoubleVal (flow_val)) ;
			}

			// move to the next entry
			info = GxB_Matrix_Iterator_next (it) ;
		}

		GrB_OK (GrB_free (&it)) ;
	}

	// release the flow matrix; setting to NULL marks this Step as consumed
	GrB_OK (GrB_free (&pdata->flow_mtx)) ;

	//--------------------------------------------------------------------------
	// write results into the output slots
	//--------------------------------------------------------------------------

	if (pdata->yield_edges != NULL) {
		*pdata->yield_edges = SIArray_FromRaw (&edges) ;
	}

	if (pdata->yield_edgeFlows != NULL) {
		*pdata->yield_edgeFlows = SIArray_FromRaw (&flows) ;
	}

	return pdata->output ;
}

// Proc_MaxFlowFree
//
// releases all resources owned by the per-call context
ProcedureResult Proc_MaxFlowFree
(
	ProcedureCtx *ctx
) {
	MaxFlow_Context *pdata = (MaxFlow_Context *) ctx->privateData ;

	if (pdata == NULL) {
		return PROCEDURE_OK ;
	}

	if (pdata->R != NULL) {
		GrB_OK (GrB_free (&pdata->R)) ;
	}

	// flow_matrix is normally freed in Step; free it here only if Step was
	// never reached (e.g. the query was cancelled after Invoke)
	if (pdata->flow_mtx != NULL) {
		GrB_OK (GrB_free (&pdata->flow_mtx)) ;
	}

	rm_free (ctx->privateData) ;

	return PROCEDURE_OK ;
}

//------------------------------------------------------------------------------
// procedure registration
//------------------------------------------------------------------------------

// returns a ProcedureCtx that registers algo.maxFlow with the procedure
// subsystem. the procedure accepts one map argument and can yield any
// combination of: nodes (T_ARRAY), edges (T_ARRAY), edgeFlows (T_ARRAY)
// and maxFlow (T_DOUBLE)
ProcedureCtx *Proc_MaxFlowCtx(void) {
	ProcedureOutput *outputs = array_new (ProcedureOutput, 4) ;

	ProcedureOutput output_nodes     = {.name = "nodes",     .type = T_ARRAY}  ;
	ProcedureOutput output_edges     = {.name = "edges",     .type = T_ARRAY}  ;
	ProcedureOutput output_maxFlow   = {.name = "maxFlow",   .type = T_DOUBLE} ;
	ProcedureOutput output_edgeFlows = {.name = "edgeFlows", .type = T_ARRAY}  ;

	array_append (outputs, output_nodes) ;
	array_append (outputs, output_edges) ;
	array_append (outputs, output_maxFlow) ;
	array_append (outputs, output_edgeFlows) ;

	ProcedureCtx *ctx = ProcCtxNew ("algo.maxFlow", 1,
								   outputs,
								   Proc_MaxFlowStep,
								   Proc_MaxFlowInvoke,
								   Proc_MaxFlowFree,
								   NULL,    // privateData – allocated in Invoke
								   true) ;  // procedure is read-only
	return ctx ;
}

