/*
* Copyright FalkorDB Ltd. 2023 - present
* Licensed under the Server Side Public License v1 (SSPLv1).
*/

#include "RG.h"
#include "utils.h"
#include "classes.h"
#include "node_class.h"
#include "../query_ctx.h"
#include "attributes_class.h"
#include "../schema/schema.h"
#include "../arithmetic/algebraic_expression.h"

extern JSClassID js_node_class_id ;  // JS node class

//------------------------------------------------------------------------------
// forward declaration
//------------------------------------------------------------------------------

static int js_node_get_property (JSContext *js_ctx, JSPropertyDescriptor *desc,
		JSValueConst obj, JSAtom prop) ;

static void js_node_finalizer (JSRuntime *rt, JSValue val) ;

static JSClassExoticMethods js_node_exotic = {
	.get_own_property = js_node_get_property,
} ;

// Define class + prototype
static JSClassDef js_node_class = {
    "Node",
	.exotic = &js_node_exotic,
	.finalizer = js_node_finalizer
} ;

// property accessor for Node attributes
// allows accessing user-defined attributes via n.attributes.xxx
static int js_node_get_property
(
	JSContext *js_ctx,
	JSPropertyDescriptor *desc,
	JSValueConst obj,
	JSAtom prop
) {
	GraphEntity *e = JS_GetOpaque (obj, js_node_class_id) ;
    if (!e) {
        return 0 ;  // property not found
    }

    const char *key = JS_AtomToCString (js_ctx, prop) ;
    if (!key) {
        return -1 ; // exception
    }

	// reject built-in node attributes
	if (strcmp (key, "id")         == 0 ||
		strcmp (key, "attributes") == 0 ||
		strcmp (key, "labels")     == 0) {
		// reject direct access to the 'id', 'attributes' & 'labels'
		// e.g. n.id
		// this will cause quickjs to invoke a dedicated function
		// in case access to a user defined 'id' attribute is required
		// access it via n.attributes.id
		return 0 ;
	}

    // search for attribute
    GraphContext *gc = QueryCtx_GetGraphCtx () ;
    ASSERT (gc != NULL) ;

    // get attribute id
    AttributeID attr_id = GraphContext_GetAttributeID (gc, key) ;
    JS_FreeCString (js_ctx, key) ;

    // unknown attribute
    if (attr_id == ATTRIBUTE_ID_NONE) {
        return 0 ;  // property not found
    }

    // get attribute from node object
    SIValue v ;
    if (!GraphEntity_GetProperty (e, attr_id, &v)) {
		return 0 ; // property not found
	}

	// key found -> convert to JSValue
	if (desc) {
		desc->flags  = JS_PROP_ENUMERABLE ;  // configurable, etc. as needed
		desc->value  = UDF_SIValueToJS (js_ctx, v) ;
		desc->getter = JS_UNDEFINED ;
		desc->setter = JS_UNDEFINED ;
	}
	return 1 ;  // property exists
}

//------------------------------------------------------------------------------
// node accessors
//------------------------------------------------------------------------------

// return the ID of the node
static JSValue js_entity_get_id
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    GraphEntity *entity = JS_GetOpaque2 (js_ctx, this_val, js_node_class_id) ;
    if (!entity) {
        return JS_EXCEPTION ;
	}

    JSValue obj = JS_NewInt64 (js_ctx, ENTITY_GET_ID (entity)) ;

    return obj ;
}

// return the label(s) associated with the node
static JSValue js_entity_get_labels
(
	JSContext *js_ctx,
	JSValueConst this_val
) {
    Node *node = JS_GetOpaque2 (js_ctx, this_val, js_node_class_id) ;
    if (!node) {
        return JS_EXCEPTION ;
	}

	// get node labels
	uint n_lbl ;
	Graph        *g  = QueryCtx_GetGraph () ;
	GraphContext *gc = QueryCtx_GetGraphCtx () ;

	NODE_GET_LABELS (g, node, n_lbl) ;

	// populate js array
	JSValue obj = JS_NewArray (js_ctx) ;
	for (int i = 0; i < n_lbl; i++) {
		// convert label id to string
		Schema *s = GraphContext_GetSchemaByID (gc, labels[i], SCHEMA_NODE) ;
		ASSERT (s != NULL) ;

		const char *lbl = Schema_GetName (s) ;
		JS_SetPropertyUint32 (js_ctx, obj, i, JS_NewString (js_ctx, lbl)) ;
	}

    return obj ;
}

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

// traverse from src node
// caller can specify the set of relationship-types to consider
// in addition to the labels associated with reachable nodes
// this function can traverse in both directions (forward, backwards or both)
// lastly caller can decide if he wish to get the reachable nodes or edges
static JSValue _traverse
(
	JSContext *js_ctx,        // JavaScript context
	EntityID src_id,          // traversal begins from this node
	uint distance,            // direct neighbors [ignored]
	char **labels,            // neighbors labels
	char **rel_types,         // edge types to consider
	GRAPH_EDGE_DIR dir,       // edge direction
	GraphEntityType ret_type  // returned entity type
) {
	const Graph         *g  = QueryCtx_GetGraph () ;
	const GraphContext  *gc = QueryCtx_GetGraphCtx () ;
	AlgebraicExpression *ae = NULL ;

	size_t required_dim = Graph_RequiredMatrixDim (g) ;

	//--------------------------------------------------------------------------
	// filter matrix
	//--------------------------------------------------------------------------

	// a 1XN matrix, practiclly a vector specifying traversal starting position
	Delta_Matrix F ;
	Delta_Matrix_new (&F, GrB_BOOL, 1, required_dim, false) ;

	GrB_Matrix FM = Delta_Matrix_M (F) ;
	GrB_OK (GrB_Matrix_setElement_BOOL (FM, true, 0, src_id)) ;

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

	int i = 0;  // output index
	JSValue neighbors = JS_NewArray (js_ctx) ;  // output

	Edge *edges = NULL ;  // intermediate edge collection
	if (ret_type == GETYPE_EDGE) {
		edges = array_new (Edge, 1) ;
	}

	// iterate over reachable nodes / edges
    GrB_Info info = GxB_rowIterator_seekRow (it, 0) ;
    if (info != GxB_EXHAUSTED) {
        // iterate over entries in FM(0,:)
		while (info == GrB_SUCCESS) {
			GrB_Index dest_id = GxB_rowIterator_getColIndex (it) ;

			if (ret_type == GETYPE_NODE) {
				// fetch node
				Node n ;
				Graph_GetNode (g, dest_id, &n) ;

				// add node to neighbors
				JS_SetPropertyUint32 (js_ctx, neighbors, i++,
						UDF_CreateNode (js_ctx, &n)) ;
			}

			// collecting edges
			else {
				// fetch edges
				for (int j = 0; j < n_rel_ids; j++) {
					if (transposed) {
						Graph_GetEdgesConnectingNodes (g, dest_id, src_id,
								rel_ids[j], &edges) ;
					} else {
						Graph_GetEdgesConnectingNodes (g, src_id, dest_id,
							rel_ids[j], &edges) ;
					}
				}
			}

			// move to the next entry in FM(i,:)
			info = GxB_rowIterator_nextCol (it) ;
		}
	}

	// add edge to neighbors
	for (int j = 0; j < array_len (edges) ; j++) {
		JS_SetPropertyUint32 (js_ctx, neighbors, i++,
				UDF_CreateEdge (js_ctx, edges + j)) ;
	}

	// clean up
	Delta_Matrix_free (&F) ;
	GxB_Iterator_free (&it) ;
	AlgebraicExpression_Free (ae) ;

	if (edges != NULL) {
		array_free (edges) ;
	}

	if (rel_ids != NULL) {
		rm_free (rel_ids) ;
	}

	return neighbors ;
}

// collect node's labels['Person', 'City'],neighbors
//
// example:
// let nodes = n.getNeighbors();
//
// accepts an optional config map:
// {
//   direction:  string   - 'incoming' / 'outgoing' / 'both',
//   types:      string[] - ['KNOWS', 'WORKS_AT'],
//   labels:     string[] - ['Person', 'City'],
//   distance:   number   - traversal depth,
//   returnType: string   - 'nodes' / 'edges'
// }
//
// all fields in map are optional
//
// returns an array of Nodes
static JSValue js_node_get_neighbors
(
	JSContext *js_ctx,
	JSValueConst this_val,
	int argc,
	JSValueConst *argv
) {
	ASSERT (js_ctx != NULL) ;

	// get the native Node pointer from this_val
    Node *node = JS_GetOpaque2 (js_ctx, this_val, js_node_class_id) ;
    if (!node) {
        return JS_EXCEPTION ;
    }

	//--------------------------------------------------------------------------
	// default config
	//--------------------------------------------------------------------------

	uint distance            = 1 ;                        // direct neighbors
	char **labels            = NULL ;                     // neighbors labels
	char **rel_types         = NULL ;                     // edge types to consider
	GRAPH_EDGE_DIR dir       = GRAPH_EDGE_DIR_OUTGOING ;  // edge direction
	GraphEntityType ret_type = GETYPE_NODE ;              // returned entity type

	const char *err_msg = NULL ;

	if (argc > 0) {
		JSValueConst options_obj = argv[0] ;

		// check if argument is an object
		if (!JS_IsObject (options_obj)) {
			return JS_ThrowTypeError (js_ctx,
					"argument must be an object or omitted") ;
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
					dir = GRAPH_EDGE_DIR_OUTGOING ;
				}

				else if (strcmp (dir_str, "incoming") == 0) {
					dir = GRAPH_EDGE_DIR_INCOMING ;
				}

				// 'both' is default, but validate if provided
				else if (strcmp (dir_str, "both") != 0) {
					err_msg = "'direction' must be 'incoming', 'outgoing', or 'both'." ;
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
			if (js_array_to_c_strings (js_ctx, types_val, &rel_types)) {
				// error occurred during conversion
				// (e.g., not an array, or elements aren't strings)
				err_msg = "'types' must be an array of strings" ;
			}
		}
		JS_FreeValue (js_ctx, types_val) ;

		//----------------------------------------------------------------------
		// parse labels
		//----------------------------------------------------------------------

		JSValue labels_val = JS_GetPropertyStr (js_ctx, options_obj, "labels") ;
		if (!JS_IsUndefined (labels_val)) {
			if (js_array_to_c_strings (js_ctx, labels_val, &labels)) {
				// error occurred during conversion
				// (e.g., not an array, or elements aren't strings)
				err_msg = "'labels' must be an array of strings" ;
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
				err_msg = "'distance' must be a positive integer." ;
			}

			// check for negative value
			else if (dist_long <= 0) {
				err_msg = "'distance' must be a positive integer." ;
			}

			// cast to int
			else {
				distance = (int)dist_long ;
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
					ret_type = GETYPE_EDGE ;
				}

				// 'nodes' is default, but validate if provided
				else if (strcmp (ret_type_str, "nodes") != 0) {
					err_msg = "'returnType' must be 'nodes' or 'edges'." ;
				}
				JS_FreeCString (js_ctx, ret_type_str) ;
			}
		}
		JS_FreeValue (js_ctx, ret_type_val) ;

		//----------------------------------------------------------------------
		// done parsing
		//----------------------------------------------------------------------

		// check for error
		if (err_msg != NULL) {
			return JS_ThrowTypeError (js_ctx, "%s", err_msg) ;
		}
	}

	// collect neighbors
	JSValue neighbors = _traverse (js_ctx, ENTITY_GET_ID(node), distance,
			labels, rel_types, dir, ret_type) ;

	//--------------------------------------------------------------------------
	// clean up
	//--------------------------------------------------------------------------

	if (labels != NULL) {
		for (int i = 0; i < array_len (labels) ; i++) {
			free (labels[i]) ;
		}
		array_free (labels) ;
	}

	if (rel_types != NULL) {
		for (int i = 0; i < array_len (rel_types) ; i++) {
			free (rel_types[i]) ;
		}
		array_free (rel_types) ;
	}

	return neighbors ;
}

// create a JavaScript Node object from a FalkorDB Node
// wraps a native FalkorDB Node into a QuickJS JSValue instance
// return JSValue representing the Node in QuickJS
JSValue UDF_CreateNode
(
	JSContext *js_ctx,  // JavaScript context
	const Node *node    // pointer to the native FalkorDB Node
) {
	ASSERT (node   != NULL) ;
	ASSERT (js_ctx != NULL) ;

	// clone node
	Node *_node = rm_malloc (sizeof (Node)) ;
	memcpy (_node, node, sizeof (Node)) ;

    JSValue obj = JS_NewObjectClass (js_ctx, js_node_class_id) ;
    if (JS_IsException (obj)) {
        return obj ;
    }

    JS_SetOpaque (obj, (void*) _node) ;

    return obj ;
}

// destructor for the Node JS object
// frees the underlying native Node when the JS object is garbage collected
static void js_node_finalizer
(
	JSRuntime *rt,
	JSValue val
) {
    // get the opaque pointer
    GraphEntity *node = JS_GetOpaque (val, js_node_class_id) ;

    // check if the pointer exists and free the native object
    if (node) {
		rm_free (node) ;
    }
}

// register the Node class with a JavaScript runtime
// associates the Node class definition with the given QuickJS runtime
void UDF_RegisterNodeClass
(
	JSRuntime *js_runtime  // JavaScript runtime
) {
	ASSERT (js_runtime != NULL) ;

	// register for each runtime
    int res = JS_NewClass (js_runtime, js_node_class_id, &js_node_class) ;
	ASSERT (res == 0) ;
}

// register the Node class with a JavaScript context
// makes the Node class available within the provided QuickJS context
static const JSCFunctionListEntry node_proto_func_list[] = {
	JS_CGETSET_DEF ("id", js_entity_get_id, NULL),
	JS_CGETSET_DEF ("labels", js_entity_get_labels, NULL),
	JS_CGETSET_DEF ("attributes", UDF_EntityGetAttributes, NULL),
	JS_CFUNC_DEF   ("getNeighbors", 1, js_node_get_neighbors),
};

void UDF_RegisterNodeProto
(
	JSContext *js_ctx  // JavaScript context
) {
	ASSERT (js_ctx != NULL) ;

	// prototype object
    JSValue proto = JS_NewObject (js_ctx) ;

    int res =
		JS_SetPropertyFunctionList (js_ctx, proto, node_proto_func_list, 4) ;
	ASSERT (res == 0) ;

    JS_SetClassProto (js_ctx, js_node_class_id, proto) ;
}

