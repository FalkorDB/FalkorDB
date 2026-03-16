/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

// proc_max_flow.c
//
// implements the `algo.maxFlow` stored procedure
//
// given a directed weighted graph, a source node, and a sink node, this
// procedure computes the maximum flow through the network and returns the
// participating nodes, edges, and per-edge flow values
//
// usage:
//   CALL algo.maxFlow({
//     sourceNode:        s,
//     targetNode:        t,
//     capacityProperty:  'cap',
//     nodeLabels:        ['Intersection'],
//     relationshipTypes: ['CONNECTS']
//   })
//   YIELD nodes, edges, flow

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
#define MAX_FLOW_CONFIG_KEY_COUNT 5

//------------------------------------------------------------------------------
// context
//------------------------------------------------------------------------------

// per-call state threaded through Invoke → Step → Free
typedef struct {
	GrB_Matrix R ;          // relationship matrix for the chosen rel-type
					        // used to resolve EdgeIDs during Step

	GrB_Matrix flow_mtx ;   // result matrix from LAGr_MaxFlow
							// set to NULL after Step has consumed it

	SIValue output[3] ;     // packed output slots [nodes, edges, flow]
	SIValue *yield_nodes ;  // pointer into output[] for the nodes slot, or NULL
	SIValue *yield_edges ;  // pointer into output[] for the edges slot, or NULL
	SIValue *yield_flows ;  // pointer into output[] for the flow slot, or NULL
} MaxFlow_Context ;

// context passed to the GraphBLAS IndexUnaryOp that maps each matrix entry to
// its capacity value
typedef struct {
	const Graph *g;        // the graph being queried
	AttributeID attr_id;   // attribute ID that holds the capacity value
	SIType expected_type;  // accepted SIType(s) for the capacity attribute
	uint64_t default_cap;  // capacity to use when the attribute is missing
						   // or has the wrong type (default: 1)
} EdgeCapacityContext ;

// GraphBLAS IndexUnaryOp callback
// reads the capacity attribute from edge (i,j) and writes it to *z.
// NOTE: the edge value stored in the matrix is passed via x, but we resolve
//       the edge by its position (i, j) through the relationship matrix
//       instead. if the attribute is absent or has an unexpected type the
//       default capacity is used
static void _get_edge_capacity
(
	uint64_t *z,                    // [output] capacity value
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

	*z = (SI_TYPE (v) & ctx->expected_type) ?
		(uint64_t) v.longval
		: ctx->default_cap ;
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
	SIValue config,       // procedure configuration map
	LabelID **lbls,       // [output] array of label IDs (caller must free)
	RelationID **rels,    // [output] array of relation IDs (caller must free)
	Node **src,           // [output] source node
	Node **sink,          // [output] sink node
	AttributeID *attr_id  // [output] capacity attribute ID
) {
	ASSERT (src     != NULL) ;
	ASSERT (lbls    != NULL) ;
	ASSERT (rels    != NULL) ;
	ASSERT (sink    != NULL) ;
	ASSERT (attr_id != NULL) ;
	ASSERT (SI_TYPE (config) == T_MAP) ;

	// initialise outputs
	*src     = NULL ;
	*lbls    = NULL ;
	*rels    = NULL ;
	*sink    = NULL ;
	*attr_id = ATTRIBUTE_ID_NONE ;

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
		u_int32_t l = SIArray_Length (v) ;

		for (u_int32_t i = 0; i < l; i++) {
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

	if (MAP_GETCASEINSENSITIVE (config, "sourceNode", v)) {
		if (SI_TYPE (v) != T_NODE) {
			ErrorCtx_SetError ("MaxFlow configuration, "
					"'sourceNodes' should be a node") ;
			goto error ;
		}
		
		*src = v.ptrval ;
		matched++ ;
	}

	//--------------------------------------------------------------------------
	// read targetNode
	//--------------------------------------------------------------------------

	if (MAP_GETCASEINSENSITIVE (config, "targetNode", v)) {
		if (SI_TYPE (v) != T_NODE) {
			ErrorCtx_SetError ("MaxFlow configuration, "
					"'targetNode' should be a node") ;
			goto error ;
		}
		
		*sink = v.ptrval ;
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
			continue ;
		}

		else if (strcasecmp ("flow", yield [i]) == 0) {
			ctx->yield_flows = ctx->output + slot++ ;
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
	Node        *src    = NULL ;
	Node        *sink   = NULL ;
	AttributeID attr_id = ATTRIBUTE_ID_NONE ;

	//--------------------------------------------------------------------------
	// parse and validate configuration
	//--------------------------------------------------------------------------

	if (!_read_config (args[0], &lbls, &rels, &src, &sink, &attr_id)) {
		return PROCEDURE_ERR ;
	}

	if (rels == NULL || array_len (rels) != 1) {
		ErrorCtx_SetError ("algo.maxFlow: 'relationshipTypes' is required "
				"and must contain exactly one type") ;
		return PROCEDURE_ERR ;
	}

	if (Graph_RelationshipContainsMultiEdge (g, rels[0])) {
		ErrorCtx_SetError ("algo.maxFlow: relationship type must not "
				"contain multi-edges (tensors)") ;
		return PROCEDURE_ERR ;
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

	EdgeCapacityContext cap_ctx = {
		.g             = g,
		.attr_id       = attr_id,
		.expected_type = T_INT64,
		.default_cap   = 1
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
				GrB_UINT64, GrB_UINT64, cap_ctx_type)) ;

	GrB_Matrix C ;
	GrB_Index nrows, ncols ;
	GrB_OK (GrB_Matrix_nrows (&nrows, U)) ;
	GrB_OK (GrB_Matrix_ncols (&ncols, U)) ;
	GrB_OK (GrB_Matrix_new (&C, GrB_UINT64, nrows, ncols)) ;

	GrB_OK (GrB_Matrix_apply_IndexOp_Scalar (
		C, NULL, NULL, get_capacity, U, cap_ctx_s, NULL)) ;

	GrB_OK (GrB_free (&cap_ctx_s)) ;
	GrB_OK (GrB_free (&cap_ctx_type)) ;
	GrB_OK (GrB_free (&get_capacity)) ;

	// configuration arrays no longer needed
	if (lbls != NULL) {
		array_free (lbls) ;
	}

	if (rels != NULL) {
		array_free (rels) ;
	}

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

	// execute Betweenness Centrality
	double   flow ;
	GrB_Info info = LAGr_MaxFlow (&flow, &pdata->flow_mtx, G,
			ENTITY_GET_ID (src), ENTITY_GET_ID (sink), msg) ;

	GrB_OK (LAGraph_Delete (&G, msg)) ;

	ctx->privateData = pdata ;

	if (info != GrB_SUCCESS) {
		return PROCEDURE_ERR ;
	}

	return PROCEDURE_OK ;
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

		while (info != GxB_EXHAUSTED) {
			// get the entry v(i)
			GrB_Index node_id = GxB_Vector_Iterator_getIndex (it) ;

			// fetch node
			Node *n = rm_malloc (sizeof (Node)) ;
			Graph_GetNode (g, node_id, n) ;
			array_append (nodes, SI_Node (n)) ;

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

	if (pdata->yield_flows != NULL) {
		flows = array_new (SIValue, edge_count) ;
	}

	//--------------------------------------------------------------------------
	// initialize iterator
	//--------------------------------------------------------------------------

	GxB_Iterator it ;
	GrB_OK (GxB_Iterator_new (&it)) ;
	GrB_OK (GxB_Matrix_Iterator_attach (it, pdata->flow_mtx, NULL)) ;

	// seek to the first entry
	GrB_Info info = GxB_Matrix_Iterator_seek (it, 0) ;
	while (info != GxB_EXHAUSTED) {
		// get current iterator indices
		GrB_Index i, j ;
		GxB_Matrix_Iterator_getIndex (it, &i, &j) ;

		if (pdata->yield_edges != NULL) {
			EdgeID id ;
			GrB_OK (GrB_Matrix_extractElement_UINT64 (&id, R, i, j)) ;

			Edge *e = rm_malloc (sizeof (Edge)) ;
			Graph_GetEdge (g, id, e) ;
			array_append (edges, SI_Edge (e)) ;
		}

		if (pdata->yield_flows != NULL) {
			// get the entry A(i,j)
			double flow_val = GxB_Iterator_get_FP64 (it) ;
			array_append (flows, SI_DoubleVal (flow_val)) ;
		}

		// move to the next entry
		info = GxB_Matrix_Iterator_next (it) ;
	}

	GrB_OK (GrB_free (&it)) ;

	// release the flow matrix; setting to NULL marks this Step as consumed
	GrB_OK (GrB_free (&pdata->flow_mtx)) ;

	//--------------------------------------------------------------------------
	// write results into the output slots
	//--------------------------------------------------------------------------

	if (pdata->yield_edges != NULL) {
		*pdata->yield_edges = SIArray_FromRaw (&edges) ;
	}

	if (pdata->yield_flows != NULL) {
		*pdata->yield_flows = SIArray_FromRaw (&flows) ;
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
// combination of: nodes (T_ARRAY), edges (T_ARRAY), flow (T_DOUBLE).
ProcedureCtx *Proc_MaxFlowCtx(void) {
	ProcedureOutput *outputs = array_new (ProcedureOutput, 3) ;

	ProcedureOutput output_flow  = {.name = "flow",  .type = T_DOUBLE} ;
	ProcedureOutput output_nodes = {.name = "nodes", .type = T_ARRAY}  ;
	ProcedureOutput output_edges = {.name = "edges", .type = T_ARRAY}  ;

	array_append (outputs, output_flow) ;
	array_append (outputs, output_nodes) ;
	array_append (outputs, output_edges) ;

	ProcedureCtx *ctx = ProcCtxNew ("algo.maxFlow", 1,
								   outputs,
								   Proc_MaxFlowStep,
								   Proc_MaxFlowInvoke,
								   Proc_MaxFlowFree,
								   NULL,    // privateData – allocated in Invoke
								   true) ;  // procedure is read-only
	return ctx ;
}

