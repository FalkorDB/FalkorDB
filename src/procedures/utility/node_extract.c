/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "./internal.h"
#include "GraphBLAS.h"
#include "graph/delta_matrix/delta_matrix.h"
#include "graph/entities/attribute_set.h"
#include "value.h"

// structure that holds all the context nessesary for the GraphBLAS functions
typedef struct
{
	const Graph *g;  // graph
	AttributeID w;   // attribute used as weight
	SIType type;     // desired type of attribute
	void *defaultA;  // default value of the attribute
} selectContext;

// outputs true if the current nodeID has the given attribute
void _selectID_withAttribute
(
	bool *z,                   // [output] keep edge?
	const void *x,             // unused
	GrB_Index i,               // nodeID
	GrB_Index j,               // unused
	const selectContext *ctx   // theta
) {
	ASSERT(sizeof(ctype) == sizeof(v.longval));
	Node n;
	ASSERT(SCALAR_ENTRY(*x));
	bool found = Graph_GetNode(ctx->g, (NodeID) i, &n);
	ASSERT(found == true);

	SIValue v ;
	GraphEntity_GetProperty ((GraphEntity *) &n, ctx->w, &v) ;
	*z = (SI_TYPE(v) & ctx->type) != 0;
}

// get the value of a certain attribute given its c-type
#define FDB_GETATT_FUNC(ctype)                                                 \
void _getAtt_##ctype                                                           \
(                                                                              \
	ctype *z,                  /* [output] edge weight */                      \
	const void *x,             /* unused */                                    \
	GrB_Index i,               /* nodeID */                                    \
	GrB_Index j,               /* unused */                                    \
	const selectContext *ctx   /* theta */                                     \
) {                                                                            \
	ASSERT(sizeof(ctype) == sizeof(v.longval));                                \
	Node e;                                                                    \
	ASSERT(SCALAR_ENTRY(*x));                                                  \
	bool found = Graph_GetNode(ctx->g, (NodeID) i, &e);                        \
	ASSERT(found == true);                                                     \
                                                                               \
	SIValue v ;                                                                \
	GraphEntity_GetProperty ((GraphEntity *) &e, ctx->w, &v) ;                 \
	ctype *val = (ctype *) &v.longval;                                         \
                                                                               \
	if(SI_TYPE(v) & ctx->type) {                                               \
		*z = *val;                                                             \
	} else {                                                                   \
		*z = *((ctype *) ctx->defaultA);                                       \
	}                                                                          \
}

// funtion definitions for _getAtt_* functions
FDB_GETATT_FUNC(int64_t);
FDB_GETATT_FUNC(double);

void get_nodes_with_lbls
(
	GrB_Vector *rows,      // [output] filtered rows
	const Graph *g,        // graph
	const LabelID *lbls,   // [optional] labels to consider
	unsigned short n_lbls  // number of labels
) {
	ASSERT(rows != NULL);

	if(n_lbls > 0) {
		Delta_Matrix lbls = Graph_GetNodeLabelMatrix(g);
		GrB_Matrix   L    = NULL;
		GrB_Vector   x    = NULL;
		GrB_OK (GrB_Vector_new(&x, GrB_BOOL, Graph_RequiredMatrixDim(g)));
		GrB_OK (GrB_Vector_new(rows, GrB_BOOL, Graph_RequiredMatrixDim(g)));

		// TODO: use some sort of Delta_mxv instead of exporting
		GrB_OK (Delta_Matrix_export(&L, lbls, GrB_BOOL));
		
		GrB_OK (GrB_mxv(*rows, NULL, NULL, GxB_ANY_PAIR_BOOL, L, x, NULL));
		GrB_OK(GrB_Vector_resize(*rows, Graph_UncompactedNodeCount(g)));
	} else if(rows != NULL) {
		GrB_OK (GrB_Vector_new(rows, GrB_BOOL, Graph_UncompactedNodeCount(g)));
		// no labels, N = present nodes
		GrB_OK (GrB_Vector_assign_BOOL(
			*rows, NULL, NULL, true, GrB_ALL, 0, NULL));

		// remove deleted nodes from N
		if(Graph_DeletedNodeCount(g) > 0) {
			NodeID *deleted_n = NULL;
			uint64_t deleted_n_count = 0;
			Graph_DeletedNodes(g, &deleted_n, &deleted_n_count);
			
			for(uint64_t i = 0; i < deleted_n_count; i++) {
				// remove deleted nodes from N
				GrB_OK (GrB_Vector_removeElement(*rows, deleted_n[i]));
			}
			rm_free(deleted_n);
		}
	}
}

// Get attributes of the given nodes
void get_node_attribute
(
	GrB_Vector rows,       // [input / output] the nodes for which to get values
	const Graph *g,        // graph
	AttributeID attr,      // attribute to get
	SIValue default_val,   // the default value if attribute does not exist
	SIType allowed_types    // allowed types (other attributes as DNE)
) {
	ASSERT (g    != NULL);
	ASSERT (rows != NULL);
	ASSERT (type != NULL);
	ASSERT (attr != ATTRIBUTE_ID_ALL);
	ASSERT (attr != ATTRIBUTE_ID_NONE);

	bool del_nodes = SIValue_IsNull(default_val); // delete nodes of wrong type?
	ASSERT (del_nodes || (allowed_type & SI_TYPE(defaul_val)));

	selectContext ctx = {.g = g, .w = attr, .type = allowed_types,
	                     .defaultA = &default_val.longval };

	GrB_Index nrows = 0;
	GrB_Vector       x        = NULL;
	GrB_Type         ctx_type = NULL;
	GrB_IndexUnaryOp selectOp = NULL; // select values within allowed types
	GrB_IndexUnaryOp getValue = NULL;
	GrB_Scalar       ctx_s    = NULL;
	GxB_Container    cont     = NULL; // to unload rows
	GrB_Scalar       iso_val  = NULL; // isoval of new rows vector
	GrB_Type         type     = NULL;

	GrB_OK (GxB_Container_new(&cont));
	GrB_OK (GrB_Type_new(&ctx_type, sizeof(selectContext)));
	GrB_OK (GrB_Scalar_new(&ctx_s, ctx_type));
	GrB_OK (GrB_Scalar_setElement_UDT(ctx_s, (void *) &ctx));
	GrB_OK (GrB_Vector_size(&nrows, rows));

	if (del_nodes){
		GrB_OK (GrB_IndexUnaryOp_new(
			&selectOp, (GxB_index_unary_function) _selectID_withAttribute,
			GrB_BOOL, GrB_BOOL, ctx_type));
		GrB_OK (GrB_Vector_select_Scalar(
			rows, NULL, NULL, selectOp, rows, ctx_s, NULL));
		GrB_OK (GrB_free(&selectOp));
	}

	// prepare to switch GrB_Type of rows
	GrB_OK (GxB_unload_Vector_into_Container(rows, cont, NULL));
	// row should be an isovalued vector
	ASSERT (cont->iso);
	// free container value
	GrB_OK (GrB_free(&cont->x));

	switch (allowed_types) {
		case T_BOOL:
		case T_INT64:
		case T_BOOL | T_INT64:
			type = GrB_INT64;
			GrB_OK (GrB_IndexUnaryOp_new(
				&getValue, (GxB_index_unary_function) _getAtt_int64_t,
				type, type, ctx_type));

		break;
		case T_DOUBLE:
			type = GrB_FP64;
			GrB_OK (GrB_IndexUnaryOp_new(
				&getValue, (GxB_index_unary_function) _getAtt_double,
				type, type, ctx_type));
		break;
		
		default:
			// non-integer typecasting and certain types not currently supported
			ASSERT(false);
			break;
	}
	GrB_OK (GrB_Vector_new(&x, type, nrows));
	GrB_OK (GrB_Scalar_new(&iso_val, type));

	// doesn't really matter if this casts correctly,
	// the value will change shortly.
	GrB_OK (GrB_Scalar_setElement_INT64(iso_val, 0));

	cont->x = (GrB_Vector) iso_val;
	GrB_OK (GxB_load_Vector_from_Container(rows, cont, NULL));

	GrB_OK (GrB_Vector_apply_IndexOp_Scalar(
		rows, NULL, NULL, getValue, rows, ctx_s, NULL));

	GrB_OK (GrB_free(&getValue));
	GrB_OK (GrB_free(&iso_val));
	GrB_OK (GrB_free(&ctx_s));
	GrB_OK (GrB_free(&cont));
}
