/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "quickjs.h"
#include "../query_ctx.h"
#include "../schema/schema.h"
#include "../arithmetic/algebraic_expression.h"

// converts a JavaScript array of strings (JSValueConst) into a dynamically
// allocated C array of strings (char**).
//
// this function handles error cases (e.g., input is not an array, or elements
// are not strings) and throws a JS exception on failure
// returns 0 on success, -1 on failure (exception thrown)
static int js_array_to_c_strings
(
    JSContext *js_ctx,      // js_ctx The QuickJS context
    JSValueConst js_array,  // js_array The JSValueConst representing the array
    char ***c_strings_out   // c_strings_out Pointer to the output char**
) {
    JSValue len_val = JS_UNDEFINED ;
    uint32_t len ;
    int ret = -1 ; // default return is failure

    // check if it's an array
    if (!JS_IsArray (js_ctx, js_array)) {
        JS_ThrowTypeError (js_ctx, "Expected an array of strings.") ;
        return -1 ;
    }

    // get the array length
    len_val = JS_GetPropertyStr (js_ctx, js_array, "length") ;
    if (JS_IsException (len_val) || JS_ToUint32 (js_ctx, &len, len_val)) {
        // error getting length or converting to uint32
        goto cleanup ;
    }

	*c_strings_out = array_new (char *, len) ;

    // initialize output count
    if (len == 0) {
        ret = 0 ; // success for empty array
        goto cleanup ;
    }

    // iterate and convert each element
    for (uint32_t i = 0; i < len; i++) {
        JSValue js_element = JS_GetPropertyUint32 (js_ctx, js_array, i) ;
        if (JS_IsException (js_element)) {
            // error getting property
            goto error_free_strings ;
        }

        // ensure element is a string before converting
        if (!JS_IsString (js_element)) {
            JS_FreeValue (js_ctx, js_element) ;
            JS_ThrowTypeError (js_ctx,
					"Array element at index %d is not a string.", i) ;
            goto error_free_strings ;
        }

        // convert to C string (copy is made)
        const char *c_str = JS_ToCString (js_ctx, js_element) ;
        JS_FreeValue (js_ctx, js_element) ; // free the JS string value

        if (!c_str) {
            // memory error or exception during conversion
            goto error_free_strings ;
        }

        // must allocate a copy to persist the string after JS_FreeCString
        array_append (*c_strings_out, strdup (c_str)) ;
        JS_FreeCString (js_ctx, c_str) ; // free the temp C string
    }

    // success
    ret = 0 ;
    goto cleanup ;

error_free_strings:
    // if an error occurred mid-loop, free the strings allocated so far
    for (uint32_t i = 0; i < array_len (*c_strings_out) ; i++) {
        free ((*c_strings_out)[i]) ;
    }
    array_free (*c_strings_out) ;

cleanup:
    JS_FreeValue (js_ctx, len_val) ;
    return ret ;
}

// initialize traversal configuration
bool traverse_init_config
(
	JSContext *js_ctx,          // javascript context
	int argc,                   // # arguments (expecting 0 or 1)
	JSValueConst *argv,         // config arguments (expecting a map)

	uint *distance,             // [output] traverse depth distance
	char ***labels,             // [output] restrict to reachable nodes of lbl
	char ***rel_types,          // [output] restrict to specified rel-types
	GRAPH_EDGE_DIR *dir,        // [output] traverse direction
	GraphEntityType *ret_type,  // [output] return either nodes or edges
	const char **err_msg        // [output] report error message
) {
	ASSERT (dir       != NULL) ;
	ASSERT (labels    != NULL) ;
	ASSERT (err_msg   != NULL) ;
	ASSERT (ret_type  != NULL) ;
	ASSERT (distance  != NULL) ;
	ASSERT (rel_types != NULL) ;

	//--------------------------------------------------------------------------
	// default config
	//--------------------------------------------------------------------------

	*dir       = GRAPH_EDGE_DIR_OUTGOING ;  // edge direction
	*labels    = NULL ;                     // neighbors labels
	*err_msg   = NULL ;                     // no error
	*ret_type  = GETYPE_NODE ;              // returned entity type
	*distance  = 1 ;                        // direct neighbors
	*rel_types = NULL ;                     // edge types to consider

	if (argc == 0) {
		return true ;
	}

	// if an argument is specified we're expecting a config map
	JSValueConst options_obj = argv[0] ;
	if (!JS_IsObject (options_obj)) {
		*err_msg = "argument must be an object or omitted" ;
		return false ;
	}

	//----------------------------------------------------------------------
	// parse the provided options object
	//----------------------------------------------------------------------

	//----------------------------------------------------------------------
	// parse direction
	//----------------------------------------------------------------------

	JSValue dir_val = JS_GetPropertyStr (js_ctx, options_obj, "direction") ;
	if (!JS_IsUndefined (dir_val)) {
		const char *dir_str = JS_ToCString (js_ctx, dir_val) ;

		if (dir_str != NULL) {
			if (strcmp (dir_str, "outgoing") == 0) {
				*dir = GRAPH_EDGE_DIR_OUTGOING ;
			}

			else if (strcmp (dir_str, "incoming") == 0) {
				*dir = GRAPH_EDGE_DIR_INCOMING ;
			}

			// 'both' is default, but validate if provided
			else if (strcmp (dir_str, "both") != 0) {
				*err_msg = "'direction' must be 'incoming', 'outgoing', or 'both'." ;
			}
			JS_FreeCString (js_ctx, dir_str) ;
		}
	}
	JS_FreeValue (js_ctx, dir_val) ;

	//----------------------------------------------------------------------
	// parse relationship-types
	//----------------------------------------------------------------------

	JSValue types_val = JS_GetPropertyStr (js_ctx, options_obj, "types") ;
	if (!JS_IsUndefined (types_val)) {
		if (js_array_to_c_strings (js_ctx, types_val, rel_types)) {
			// error occurred during conversion
			// (e.g., not an array, or elements aren't strings)
			*err_msg = "'types' must be an array of strings" ;
		}
	}
	JS_FreeValue (js_ctx, types_val) ;

	//----------------------------------------------------------------------
	// parse labels
	//----------------------------------------------------------------------

	JSValue labels_val = JS_GetPropertyStr (js_ctx, options_obj, "labels") ;
	if (!JS_IsUndefined (labels_val)) {
		if (js_array_to_c_strings (js_ctx, labels_val, labels)) {
			// error occurred during conversion
			// (e.g., not an array, or elements aren't strings)
			*err_msg = "'labels' must be an array of strings" ;
		}
	}
	JS_FreeValue (js_ctx, labels_val) ;

	//----------------------------------------------------------------------
	// parse distance
	//----------------------------------------------------------------------

	JSValue dist_val = JS_GetPropertyStr (js_ctx, options_obj, "distance") ;
	if (!JS_IsUndefined (dist_val)) {
		int64_t dist_long;

		// failed to get int64
		if (JS_ToInt64 (js_ctx, &dist_long, dist_val)) {
			*err_msg = "'distance' must be a positive integer." ;
		}

		// check for negative value
		else if (dist_long <= 0) {
			*err_msg = "'distance' must be a positive integer." ;
		}

		// cast to int
		else {
			*distance = (int)dist_long ;
		}
	}
	JS_FreeValue (js_ctx, dist_val) ;

	//----------------------------------------------------------------------
	// parse returnType
	//----------------------------------------------------------------------

	JSValue ret_type_val =
		JS_GetPropertyStr (js_ctx, options_obj, "returnType") ;

	if (!JS_IsUndefined (ret_type_val)) {
		const char *ret_type_str = JS_ToCString (js_ctx, ret_type_val) ;

		if (ret_type_str != NULL) {
			if (strcmp (ret_type_str, "edges") == 0) {
				*ret_type = GETYPE_EDGE ;
			}

			// 'nodes' is default, but validate if provided
			else if (strcmp (ret_type_str, "nodes") != 0) {
				*err_msg = "'returnType' must be 'nodes' or 'edges'." ;
			}
			JS_FreeCString (js_ctx, ret_type_str) ;
		}
	}
	JS_FreeValue (js_ctx, ret_type_val) ;

	//----------------------------------------------------------------------
	// done parsing
	//----------------------------------------------------------------------

	// check for error
	return (*err_msg == NULL) ;
}

// traverse from src node
// caller can specify the set of relationship-types to consider
// in addition to the labels associated with reachable nodes
// this function can traverse in both directions (forward, backwards or both)
// lastly caller can decide if he wish to get the reachable nodes or edges
GraphEntity **traverse
(
	uint *neighbors_count,    // number of discovered neighbors
	const EntityID *sources,  // traversal begins here
	uint n,                   // number of sources
	uint distance,            // direct neighbors [ignored]
	const char **labels,      // neighbors labels
	const char **rel_types,   // edge types to consider
	GRAPH_EDGE_DIR dir,       // edge direction
	GraphEntityType ret_type  // returned entity type
) {
	if (n == 0) {
		return NULL ;
	}
	ASSERT (sources         != NULL) ;
	ASSERT (neighbors_count != NULL) ;

	const Graph         *g  = QueryCtx_GetGraph () ;
	const GraphContext  *gc = QueryCtx_GetGraphCtx () ;
	AlgebraicExpression *ae = NULL ;

	size_t required_dim = Graph_RequiredMatrixDim (g) ;

	//--------------------------------------------------------------------------
	// build filter matrix
	//--------------------------------------------------------------------------

	// a NXM matrix
	Delta_Matrix F ;
	Delta_Matrix_new (&F, GrB_BOOL, n, required_dim, false) ;

	GrB_Matrix FM = Delta_Matrix_M (F) ;
	for (uint i = 0 ; i < n ; i++) {
		GrB_OK (GrB_Matrix_setElement_BOOL (FM, true, i, sources[i])) ;
	}

	ae = AlgebraicExpression_NewOperand (F, false, "n", "n", NULL, "F") ;

	//--------------------------------------------------------------------------
	// relationships
	//--------------------------------------------------------------------------

	AlgebraicExpression *rels = NULL ;
	bool transposed = (dir == GRAPH_EDGE_DIR_INCOMING) ;

	if (rel_types == NULL) {
		// no relationship-types specified, use the ADJ matrix
		// relationship-type agnostic
		Delta_Matrix ADJ = Graph_GetAdjacencyMatrix (g, transposed) ;
		rels = AlgebraicExpression_NewOperand (ADJ, false, "n", "m", NULL,
				"ADJ") ;
	} else {
		int l = array_len (rel_types) ;
		Schema *s = NULL ;
		Delta_Matrix R = NULL ;
		RelationID rel_id = GRAPH_NO_RELATION ;

		if (l == 1) {
			// a single relationship type
			s = GraphContext_GetSchema (gc, rel_types[0], SCHEMA_EDGE) ;
			if (s == NULL) {
				// non existing relationship: F * 0
				R = Graph_GetZeroMatrix (g) ;
			} else {
				// relationship type exists: F * R
				rel_id = Schema_GetID (s) ;
				R = Graph_GetRelationMatrix (g, rel_id, transposed) ;
			}
			rels = AlgebraicExpression_NewOperand (R, false, "n", "m", NULL,
					"R") ;
		} else {
			// multiple relationship-types F * (A + B + C)
			AlgebraicExpression *add =
				AlgebraicExpression_NewOperation (AL_EXP_ADD) ;

			for (int i = 0; i < l; i++) {
				s = GraphContext_GetSchema (gc, rel_types[i], SCHEMA_EDGE) ;
				if (s == NULL) {
					R = Graph_GetZeroMatrix (g) ;
				} else {
					rel_id = Schema_GetID (s) ;
					R = Graph_GetRelationMatrix (g, rel_id, transposed) ;
				}
				AlgebraicExpression_AddChild (add,
						AlgebraicExpression_NewOperand (R, false, "n", "m", "e",
							"R")) ;
			}

			rels = add ;
		}
	}

	// in case we traverse in both directions (incoming & outgoing)
	// compute F * (rels + transpose(resl))
	if (dir == GRAPH_EDGE_DIR_BOTH) {
		AlgebraicExpression *clone = AlgebraicExpression_Clone (rels) ;

		// transpose(rels)
		AlgebraicExpression *t =
			AlgebraicExpression_NewOperation (AL_EXP_TRANSPOSE) ;
		AlgebraicExpression_AddChild (t, clone) ;

		// rels + transpose(rels)
		AlgebraicExpression *add =
			AlgebraicExpression_NewOperation (AL_EXP_ADD) ;
		AlgebraicExpression_AddChild(add, rels) ;
		AlgebraicExpression_AddChild(add, t) ;
		rels = add ;
	}

	// F * rels
	AlgebraicExpression *mul =
		AlgebraicExpression_NewOperation (AL_EXP_MUL) ;
	AlgebraicExpression_AddChild (mul, ae) ;
	AlgebraicExpression_AddChild (mul, rels) ;
	ae = mul ;

	//--------------------------------------------------------------------------
	// collect potential edge relationship types
	//--------------------------------------------------------------------------

	int n_rel_ids = 0 ;           // number of present relationship-types
	RelationID *rel_ids = NULL ;  // relationships ids

	if (ret_type == GETYPE_EDGE) {
		if (rel_types == NULL) {
			// relationship type not specified, consider all types
			rel_ids = rm_malloc (sizeof (RelationID)) ;
			rel_ids[0] = GRAPH_NO_RELATION ;
			n_rel_ids = 1 ;
		} else {
			// collect relationship-type ids
			int l = array_len (rel_types) ;
			rel_ids = rm_malloc (sizeof (RelationID) * l) ;

			for (int i = 0; i < l ; i++) {
				Schema *s =
					GraphContext_GetSchema (gc, rel_types[i], SCHEMA_EDGE) ;
				if (s == NULL) {
					continue ;
				} else {
					rel_ids[n_rel_ids++] = Schema_GetID (s) ;
				}
			}
		}
	}

	//--------------------------------------------------------------------------
	// labels
	//--------------------------------------------------------------------------

	// filter reachable neighbors
	// F * rels * lbls
	for (int i = 0; i < array_len (labels); i++) {
		Delta_Matrix L ;
		Schema *s = GraphContext_GetSchema (gc, labels[i], SCHEMA_NODE) ;
		if (s == NULL) {
			L = Graph_GetZeroMatrix (g) ;
			break ;
		} else {
			LabelID lbl_id = Schema_GetID (s) ;
			L = Graph_GetLabelMatrix (g, lbl_id) ;
		}

		AlgebraicExpression_MultiplyToTheRight (&ae, L) ;
	}

	//--------------------------------------------------------------------------
	// evaluate expression
	//--------------------------------------------------------------------------

	AlgebraicExpression_Optimize (&ae) ;
	AlgebraicExpression_Eval (ae, F) ;

	GxB_Iterator it ;
	GrB_OK (GxB_Iterator_new (&it)) ;
	GrB_OK (GxB_rowIterator_attach (it, FM, NULL)) ;

	//--------------------------------------------------------------------------
	// collect results
	//--------------------------------------------------------------------------

	// determine nuumber of neighbors for each node
	GrB_Vector w ;
	GrB_OK (GrB_Vector_new (&w, GrB_UINT64, n)) ;

	// w[i] holds number of neighbors for node i
	GrB_OK (GrB_Matrix_reduce_Monoid (w, NULL, NULL, GrB_PLUS_MONOID_UINT64, FM,
				NULL)) ;

	GraphEntity **reachables = rm_malloc (sizeof (GraphEntity*) * n) ;

	for (uint i = 0 ; i < n ; i++) {
		uint64_t wi = 0 ;
		GrB_OK (GrB_Vector_extractElement (&wi, w, i)) ;

		if (ret_type == GETYPE_NODE) {
			reachables[i] = (GraphEntity*)rm_malloc (sizeof (Node) * wi) ;
		} else {
			reachables[i] = (GraphEntity*)array_new (Edge, wi) ;
		}
		neighbors_count[i] = wi ;  // set output
	}

	GrB_OK (GrB_free (&w)) ;

	// iterate over reachable nodes / edges
    GrB_Info info = GxB_rowIterator_seekRow (it, 0) ;
    while (info != GxB_EXHAUSTED) {
        // iterate over entries in FM(i,:)
		GrB_Index src_id  = GxB_rowIterator_getRowIndex (it) ;
		GraphEntity *neighbors = reachables[src_id] ;

		uint neighbor_idx = 0 ;
		while (info == GrB_SUCCESS) {
            // get the entry FM(i,j)
			GrB_Index dest_id = GxB_rowIterator_getColIndex (it) ;

			if (ret_type == GETYPE_NODE) {
				// fetch node
				Node *n = (Node*)&neighbors[neighbor_idx++] ;
				Graph_GetNode (g, dest_id, n) ;
			}

			// collecting edges
			else {
				for (int j = 0; j < n_rel_ids; j++) {
					if (transposed) {
						Graph_GetEdgesConnectingNodes (g, dest_id, src_id,
								rel_ids[j], (Edge**)&neighbors) ;
					} else {
						Graph_GetEdgesConnectingNodes (g, src_id, dest_id,
							rel_ids[j], (Edge**)&neighbors) ;
					}
				}
				reachables[src_id] = neighbors ;
			}
            // move to the next entry in FM(i,:)
            info = GxB_rowIterator_nextCol (it) ;
		}
		// move to the next row, FM(i+1,:)
		info = GxB_rowIterator_nextRow (it) ;
	}

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	Delta_Matrix_free (&F) ;
	GxB_Iterator_free (&it) ;
	AlgebraicExpression_Free (ae) ;

	if (rel_ids != NULL) {
		rm_free (rel_ids) ;
	}

	return reachables ;
}

