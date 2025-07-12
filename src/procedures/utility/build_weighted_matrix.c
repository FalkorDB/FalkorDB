/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "./internal.h"

// the ith relationID is i if no relation is given, and rels[i] if it is.
#define GETRELATIONID(i) ((rels) ? rels[i] : i)

// frees the build weighted matrix workspace
#define BWM_FREE                                                 \
{                                                                \
	if (weight != ATTRIBUTE_ID_NONE) GrB_BinaryOp_free(&minID);  \
	GrB_Scalar_free(&theta);                                     \
	GrB_Type_free(&contx_type);                                  \
	GrB_UnaryOp_free(&toMatrix);                                 \
	GrB_BinaryOp_free(&weightOp);                                \
	GrB_BinaryOp_free(&toMatrixMin);                             \
	GxB_IndexBinaryOp_free(&minID_indexOP);                      \
}

// structure that holds all the context nessesary for the GraphBLAS functions
// can select the right edge
typedef struct
{
	const Graph *g;  // Graph
	AttributeID w;   // Attribute used as weight
	int comp;        // -1 if max, 1 if min
} compareContext;

// binary index op, does not use the index but needs the context. 
// will compare two edges and give the one with the minimum or maximum weight.
static void _compare_EdgeID_value 
(
	uint64_t *z,
	const uint64_t *x,
	GrB_Index ix,
	GrB_Index jx,
	const uint64_t *y,
	GrB_Index iy,
	GrB_Index jy,
	const compareContext *ctx
) {
	ASSERT(SCALAR_ENTRY(*x));
	ASSERT(SCALAR_ENTRY(*y));
	*z = *x;
	Edge _x, _y;
	Graph_GetEdge(ctx->g, (EdgeID) (*x), &_x);
	Graph_GetEdge(ctx->g, (EdgeID) (*y), &_y);
	SIValue *xv = GraphEntity_GetProperty((GraphEntity *) &_x, ctx->w);
	SIValue *yv = GraphEntity_GetProperty((GraphEntity *) &_y, ctx->w);

	if((SI_TYPE(*xv) & SI_NUMERIC) == 0 || 
		((SI_TYPE(*yv) & SI_NUMERIC) && 
		SIValue_Compare(*xv, *yv, NULL) == ctx->comp)
	) {
		*z = *y;
	}
}

// collapse the entries in a tensor down to a single value by finding the lowest 
// or highest weight edge
static void _reduceToMatrix
(
	uint64_t *z,
	const uint64_t *x,
	const compareContext *ctx
) {
	if(SCALAR_ENTRY(*x)) {
		*z = *x;
	} else { // find the minimum weighted edge in the vector
		GrB_Vector _v = AS_VECTOR(*x);
		
		// stack allocate the iterator
		struct GB_Iterator_opaque _i;
		GxB_Iterator i = &_i;

		GrB_Info info = GxB_Vector_Iterator_attach(i, _v, NULL);
		ASSERT(info == GrB_SUCCESS)

		info = GxB_Vector_Iterator_seek(i, 0);
		ASSERT(info == GrB_SUCCESS)

		Edge currE;
		EdgeID minID   = (EdgeID) GxB_Vector_Iterator_getIndex(i);
		SIValue *currV = NULL;

		// -infinity if max or +infinity if min
		SIValue minV = SI_DoubleVal(ctx->comp * INFINITY);

		Graph_GetEdge(ctx->g, minID, &currE);
		currV = GraphEntity_GetProperty((GraphEntity *) &currE, ctx->w);
		info = GxB_Vector_Iterator_next(i);

		// treat edges without the attribute or with a non-numeric attribute
		// as infinite length
		if (SI_TYPE(*currV) & SI_NUMERIC) {
			minV = *currV;
		}

		while (info != GxB_EXHAUSTED) {
			EdgeID CurrID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
			Graph_GetEdge(ctx->g, CurrID, &currE);
			currV = GraphEntity_GetProperty((GraphEntity *) &currE, ctx->w);

			if((SI_TYPE(*currV) & SI_NUMERIC) && 
				SIValue_Compare(minV, *currV, NULL) == ctx->comp) {
				minV  = *currV;
				minID = CurrID;
			}

			info = GxB_Vector_Iterator_next(i);
		}
		
		*z = (uint64_t) minID;
	}
}

// collapse the entries in a two tensors down to a single value by finding the 
// lowest or highest weight edge
static void _pickBinary
(
	uint64_t *z,
	const uint64_t *x,
	GrB_Index ix,
	GrB_Index jx,
	const uint64_t *y,
	GrB_Index iy,
	GrB_Index jy,
	const compareContext *ctx
) {
	GrB_Vector _v = AS_VECTOR(*x);

	// -infinity if max or +infinity if min
	SIValue minV = SI_DoubleVal(ctx->comp * INFINITY);
	EdgeID minID;

	if(SCALAR_ENTRY(*x) || _v == NULL) {
		minID = *x;
		Edge currE;
		Graph_GetEdge(ctx->g, minID, &currE);
		SIValue *currV = GraphEntity_GetProperty((GraphEntity *) &currE, ctx->w);

		// treat edges without the attribute or with a non-numeric attribute
		// as infinite length
		if (SI_TYPE(*currV) & SI_NUMERIC) {
			minV = *currV;
		}
	} else { // find the minimum weighted edge in the vector
		// stack allocate the iterator
		struct GB_Iterator_opaque _i;
		GxB_Iterator i = &_i;

		GrB_Info info = GxB_Vector_Iterator_attach(i, _v, NULL);
		ASSERT(info == GrB_SUCCESS)

		info = GxB_Vector_Iterator_seek(i, 0);
		ASSERT(info == GrB_SUCCESS)

		Edge currE;
		SIValue *currV = NULL;
		minID = (EdgeID) GxB_Vector_Iterator_getIndex(i);

		Graph_GetEdge(ctx->g, minID, &currE);
		currV = GraphEntity_GetProperty((GraphEntity *) &currE, ctx->w);
		info = GxB_Vector_Iterator_next(i);

		// treat edges without the attribute or with a non-numeric attribute
		// as infinite length
		if (SI_TYPE(*currV) & SI_NUMERIC) {
			minV = *currV;
		}

		while (info != GxB_EXHAUSTED) {
			EdgeID CurrID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
			Graph_GetEdge(ctx->g, CurrID, &currE);
			currV = GraphEntity_GetProperty((GraphEntity *) &currE, ctx->w);

			if((SI_TYPE(*currV) & SI_NUMERIC) && 
				SIValue_Compare(minV, *currV, NULL) == ctx->comp) {
				minV  = *currV;
				minID = CurrID;
			}

			info = GxB_Vector_Iterator_next(i);
		}
	}
	
	_v = AS_VECTOR(*y);
	
	if(SCALAR_ENTRY(*y) || _v == NULL) {
		Edge currE;
		Graph_GetEdge(ctx->g, (EdgeID) *y, &currE);
		SIValue *currV = GraphEntity_GetProperty((GraphEntity *) &currE, ctx->w);

		// treat edges without the attribute or with a non-numeric attribute
		// as infinite length
		if (SI_TYPE(*currV) & SI_NUMERIC &&
			SIValue_Compare(minV, *currV, NULL) == ctx->comp) {
			minV = *currV;
			minID = *y;
		}
	} else { // find the minimum weighted edge in the vector
		// stack allocate the iterator
		struct GB_Iterator_opaque _i;
		GxB_Iterator i = &_i;

		GrB_Info info = GxB_Vector_Iterator_attach(i, _v, NULL);
		ASSERT(info == GrB_SUCCESS)

		info = GxB_Vector_Iterator_seek(i, 0);
		ASSERT(info == GrB_SUCCESS)

		Edge currE;
		SIValue *currV = NULL;
		EdgeID currID = (EdgeID) GxB_Vector_Iterator_getIndex(i);

		Graph_GetEdge(ctx->g, currID, &currE);
		currV = GraphEntity_GetProperty((GraphEntity *) &currE, ctx->w);
		info = GxB_Vector_Iterator_next(i);

		// treat edges without the attribute or with a non-numeric attribute
		// as infinite length
		if (SI_TYPE(*currV) & SI_NUMERIC &&
			SIValue_Compare(minV, *currV, NULL) == ctx->comp) {
			minV = *currV;
			minID= currID;
		}

		while (info != GxB_EXHAUSTED) {
			currID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
			Graph_GetEdge(ctx->g, currID, &currE);
			currV = GraphEntity_GetProperty((GraphEntity *) &currE, ctx->w);

			if((SI_TYPE(*currV) & SI_NUMERIC) && 
				SIValue_Compare(minV, *currV, NULL) == ctx->comp) {
				minV  = *currV;
				minID = currID;
			}

			info = GxB_Vector_Iterator_next(i);
		}
	}

	*z = minID;
}

// returns the double value of the given attribute given an edgeId
void _getAttFromID
(
	double *z,                 // [output] edge weight
	const uint64_t *x,         // edge id
	const compareContext *ctx  // theta
) {
	ASSERT(SCALAR_ENTRY(*x));
	Edge e;
	bool found = Graph_GetEdge(ctx->g, (EdgeID) (*x), &e);
	ASSERT(found == true);

	SIValue *v = GraphEntity_GetProperty((GraphEntity *) &e, ctx->w);

	if(SI_TYPE(*v) & SI_NUMERIC) {
		int info = SIValue_ToDouble(v, z);
		ASSERT(info == 1);
	} else {
		// -infinity if max or +infinity if min
		*z = ctx->comp * INFINITY;
	}
}

// reduces Tensor entries to the first ID
static void _reduceToMatrixAny
(
	uint64_t *z,       // [output] single Edge ID
	const uint64_t *x  // possibly a vector entry
) {
	if(SCALAR_ENTRY(*x)) {
		*z = *x;
	} else {
		GrB_Vector _v = AS_VECTOR(*x);
		
		// stack allocate iterator
		struct GB_Iterator_opaque _i;
		GxB_Iterator i = &_i;

		GrB_Info info = GxB_Vector_Iterator_attach(i, _v, NULL);
		ASSERT(info == GrB_SUCCESS);

		// find the first edge in the vector
		info = GxB_Vector_Iterator_seek(i, 0);
		ASSERT(info == GrB_SUCCESS);

		// get the first edge ID in the vector
		EdgeID minID = (EdgeID) GxB_Vector_Iterator_getIndex(i);

		*z = (uint64_t) minID;
	}
}

// compose multiple label & relation matrices into a single matrix
// L = L0 U L1 U ... Lm
// A = L * (R0 + R1 + ... Rn) * L 
//
// if a weight attribute is specified, this function will pick which edge to 
// return given a BWM_reduce strategy
// for example, BWM_MIN returns the edge with minimum weight
// 
// A_w  = [attribute values of A]
// rows = nodes with specified labels
// in case no labels are specified rows is a dense 1 vector: [1, 1, ...1]
GrB_Info get_sub_weight_matrix_OLD
(
	GrB_Matrix *A,             // [output] matrix (EdgeIDs)
	GrB_Matrix *A_w,           // [output] matrix (weights)
	GrB_Vector *rows,          // [output] filtered rows
	const Graph *g,            // graph
	const LabelID *lbls,       // [optional] labels to consider
	unsigned short n_lbls,     // number of labels
	const RelationID *rels,    // [optional] relationships to consider
	unsigned short n_rels,     // number of relationships
	const AttributeID weight,  // weight attribute to consider
	BWM_reduce strategy,       // use either maximum or minimum weight
	bool symmetric             // build a symmetric matrix
) {
	ASSERT(g != NULL);
	ASSERT(A != NULL);
	ASSERT((lbls != NULL && n_lbls > 0) || (lbls == NULL && n_lbls == 0));
	ASSERT((rels != NULL && n_rels > 0) || (rels == NULL && n_rels == 0));

	// context for GrB operations
	compareContext ctx = {
		.g = g,                              // the input graph
		.w = weight,                         // the weight attribute to consider 
		.comp = (strategy == BWM_MAX)? -1: 1 // -1 if max, 1 if min
	};

	GrB_BinaryOp      minID         = NULL;  // gets two edge IDs and picks one
	GrB_Scalar        theta         = NULL;  // Scalar containing the context
	GrB_UnaryOp       toMatrix      = NULL;  // get any ID from vector entry
	GrB_BinaryOp      weightOp      = NULL;  // get weight from edgeID
	GrB_Type          contx_type    = NULL;  // GB equivalent of compareContext
	GrB_BinaryOp      toMatrixMin   = NULL;  // get min weight ID from vectors
	GxB_IndexBinaryOp minID_indexOP = NULL;  // minID's underlying index op 
	size_t            n             = Graph_UncompactedNodeCount(g);

	GrB_Type_new(&contx_type, sizeof(compareContext));

	if(weight == ATTRIBUTE_ID_NONE) {
		// weight attribute wasn't specified, use a dummy weight with value: 0
		minID = GrB_SECOND_UINT64;
		GrB_UnaryOp_new (
			&toMatrix, (GxB_unary_function) _reduceToMatrixAny, GrB_UINT64, 
			GrB_UINT64
		) ;
	} else {
		GrB_Scalar_new(&theta, contx_type);
		GrB_Scalar_setElement_UDT(theta, (void *) &ctx);
		GrB_BinaryOp_new(
			&weightOp, (GxB_binary_function) _getAttFromID, 
			GrB_FP64, GrB_UINT64, contx_type
		) ;

		// reduce a matrix to its minimum (or maximum) valued edge
		GrB_BinaryOp_new(
			&toMatrixMin, (GxB_binary_function) _reduceToMatrix, 
			GrB_UINT64, GrB_UINT64, contx_type
		) ;

		// pick the minimum (or maximum) valued edge from the two matricies
		GxB_IndexBinaryOp_new(&minID_indexOP,
				(GxB_index_binary_function) _compare_EdgeID_value, GrB_UINT64,
				GrB_UINT64, GrB_UINT64, contx_type, NULL, NULL);

		GxB_BinaryOp_new_IndexOp(&minID, minID_indexOP, theta);
	}

	GrB_Info info        = GrB_DEFAULT;
	GrB_Matrix M         = NULL;  // temporary matrix
	GrB_Matrix _A        = NULL;  // output matrix containing EdgeIDs
	GrB_Vector _N        = NULL;  // output filtered rows
	Delta_Matrix D       = NULL;  // graph delta matrix
	GrB_Index nrows      = 0;     // number of rows in matrix
	GrB_Index ncols      = 0;     // number of columns in matrix
	GrB_Type A_type      = NULL;  // type of the matrix
	GrB_Descriptor  desc =  NULL;

	GrB_Index rows_nvals;

	GrB_Descriptor_new(&desc);
	GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST);
	GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST);

	// if no relationships are specified, use all relationships
	// can't use adj matrix since I need access to the edgeIds of all edges
	if (rels == NULL) {
		n_rels = Graph_RelationTypeCount(g);
	}

	if (n_rels == 0) {
		// graph does not have any relations, return empty matrix
		info = GrB_Matrix_new(A, GrB_UINT64, 0, 0);
		ASSERT(info == GrB_SUCCESS);
		
		if (A_w) {
			info = GrB_Matrix_new(A_w, GrB_FP64, 0, 0);
			ASSERT(info == GrB_SUCCESS);
		}

		if (rows) {
			info = GrB_Vector_new(rows, GrB_BOOL, 0);
			ASSERT(info == GrB_SUCCESS);
		}

		BWM_FREE;
		return GrB_SUCCESS;
	}

	//--------------------------------------------------------------------------
	// compute R
	//--------------------------------------------------------------------------

	ASSERT(n_rels > 0);

	RelationID rel_id = GETRELATIONID(0);
	D = Graph_GetRelationMatrix(g, rel_id, false);

	info = Delta_Matrix_export(&_A, D);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_nrows(&nrows, _A);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols, _A);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_Matrix_type(&A_type, _A);
	ASSERT(info == GrB_SUCCESS);

	// expecting a square matrix
	ASSERT(nrows == ncols);

	// if _A has tensor entries, reduce it to a matrix
	if (Graph_RelationshipContainsMultiEdge(g, rel_id)) {
		if (weight == ATTRIBUTE_ID_NONE) {
			info = GrB_Matrix_apply(_A, NULL, NULL, toMatrix, _A, NULL);
		} else {
			info = GrB_Matrix_apply_BinaryOp2nd_UDT(_A, NULL, NULL, toMatrixMin,
					_A, (void *) (&ctx), NULL);
		}
		ASSERT(info == GrB_SUCCESS);
	}

	// in case there are multiple relation types, include them in _A
	for (unsigned short i = 1; i < n_rels; i++) {
		rel_id = GETRELATIONID(i);
		D = Graph_GetRelationMatrix(g, rel_id, false);
		info = Delta_Matrix_export(&M, D);
		ASSERT(info == GrB_SUCCESS);

		// reduce tensors to matricies and add the resulting matrix to _A
		if (Graph_RelationshipContainsMultiEdge(g, rel_id)) {
			if (weight == ATTRIBUTE_ID_NONE) {
				// apply toMatrix to get any edge and accum to also choose any 
				info = GrB_Matrix_apply(M, NULL, NULL, toMatrix, M, NULL);
			} else {
				// apply to get the min (max) valued edge from the tensor
				info = GrB_Matrix_apply_BinaryOp2nd_UDT(M, NULL, NULL,
						toMatrixMin, M, (void *) (&ctx), NULL);
				ASSERT(info == GrB_SUCCESS);

			}
		}

		// add and pick the min (max) valued edge if there is overlap
		info = GrB_Matrix_eWiseAdd_BinaryOp(_A, NULL, NULL, minID, _A, M, NULL);
		ASSERT(info == GrB_SUCCESS);
		GrB_free(&M);
	}

	//--------------------------------------------------------------------------
	// compute L
	//--------------------------------------------------------------------------

	// create vector N denoting all nodes participating in the algorithm
	if (rows != NULL) {
		info = GrB_Vector_new(&_N, GrB_BOOL, nrows);
		ASSERT(info == GrB_SUCCESS);
	}

	// enforce labels
	if (n_lbls > 0) {
		Delta_Matrix DL = Graph_GetLabelMatrix(g, lbls[0]);

		GrB_Matrix L;
		info = Delta_Matrix_export(&L, DL);
		ASSERT(info == GrB_SUCCESS);

		// L = L U M
		for (unsigned short i = 1; i < n_lbls; i++) {
			DL = Graph_GetLabelMatrix(g, lbls[i]);

			info = Delta_Matrix_export(&M, DL);
			ASSERT(info == GrB_SUCCESS);

			info = GrB_Matrix_eWiseAdd_Semiring(L, NULL, NULL,
					GxB_ANY_PAIR_BOOL, L, M, NULL);
			ASSERT(info == GrB_SUCCESS);

			GrB_Matrix_free(&M);
		}

		// set N to L's main diagonal denoting all participating nodes 
		if (_N != NULL) {
			info = GxB_Vector_diag(_N, L, 0, NULL);
			ASSERT(info == GrB_SUCCESS);
		}

		info = GrB_Vector_nvals(&rows_nvals, _N);
		// A = L * A * L
		GrB_Matrix temp = NULL;
		GrB_Matrix_new(&temp, GrB_UINT64, rows_nvals, rows_nvals);
		info = GxB_Matrix_extract_Vector(
			temp, NULL, NULL, _A, _N, _N, desc);
		ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_free(&_A);
		_A = temp;
		temp = NULL;

		// free L matrix
		info = GrB_Matrix_free(&L);
		ASSERT(info == GrB_SUCCESS);
	} else if (rows != NULL) {
		// no labels, N = [1,....1]
		info = GrB_Vector_assign_BOOL(
			_N, NULL, NULL, true, GrB_ALL, nrows, NULL);
		ASSERT(info == GrB_SUCCESS);
	}

	if (symmetric) {
		// make A symmetric A = A + At
		info = GrB_Matrix_eWiseAdd_BinaryOp(_A, NULL, NULL, minID, _A, _A,
				GrB_DESC_T1);
		ASSERT(info == GrB_SUCCESS);
	}

	if (rows != NULL) {
		info = GrB_Vector_resize(_N, n);
		ASSERT(info == GrB_SUCCESS);
	}

	//--------------------------------------------------------------------------
	// compute weight matrix
	//--------------------------------------------------------------------------

	if (A_w) {
		info = GrB_Matrix_new(A_w, GrB_FP64, rows_nvals, rows_nvals);
		ASSERT(info == GrB_SUCCESS);

		if (weight == ATTRIBUTE_ID_NONE) {
			// if no weight specified, weights are zero
			info = GrB_Matrix_assign_FP64(
				*A_w, _A, NULL, 0.0, GrB_ALL, n, GrB_ALL, n, GrB_DESC_S
			);
		} else {
			// get the weight value from the edge ids
			info = GrB_Matrix_apply_BinaryOp2nd_UDT(
				*A_w, NULL, NULL, weightOp, _A, (void *) (&ctx), NULL
			);
		}
		ASSERT(info == GrB_SUCCESS);
	}

	if(n_lbls == 0){
		// get rid of extra unused rows and columns
		info = GrB_Matrix_resize(_A, n, n);
		ASSERT(info == GrB_SUCCESS);
	}

	if(rows != NULL) {
		info = GrB_Vector_resize(_N, n);
		ASSERT(info == GrB_SUCCESS);
	}

	// set outputs
	*A = _A;
	if (rows) *rows = _N;
	_A = NULL;
	_N = NULL;

	BWM_FREE;
	return info;
}


GrB_Info _compile_matricies_w
(
	GrB_Matrix *A,             // [output] matrix
	const Delta_Matrix *mats,  // matricies to consider
	unsigned short n_mats,     // number of matricies
	GrB_BinaryOp op            // Input matrix zombie value
) {
	ASSERT(n_mats > 0);

	GrB_Info      info;
	Delta_Matrix C = NULL;
	GrB_Index    nrows;
	GrB_Index    ncols;

	Delta_Matrix_nrows(&nrows, mats[0]);
	Delta_Matrix_ncols(&ncols, mats[0]);
	
	if (n_mats == 1){
		// export relation matrix to A
		info = Delta_Matrix_export(A, mats[0]);
		ASSERT(info == GrB_SUCCESS);
	} else {
		// given the semiring I am using, C will be an invalid delta matrix.
		// but it is quickly exported so it should not be an issue.
		info = Delta_Matrix_new(&C, GrB_UINT64, nrows, ncols, false);
		ASSERT(info == GrB_SUCCESS);

		info = Delta_eWiseAdd_Biop(C, op, mats[0], mats[1]);
		ASSERT(info == GrB_SUCCESS);

		// in case there are multiple relation types, include them in A
		for(unsigned short i = 2; i < n_mats; i++) {
			info = Delta_eWiseAdd_Biop(C, op, C, mats[1]);
			ASSERT(info == GrB_SUCCESS);
		}
		
		Delta_Matrix_wait(C, true);
		*A = DELTA_MATRIX_M(C);
		DELTA_MATRIX_M(C) = NULL;
		// info = Delta_Matrix_export(A, C);
		ASSERT(info == GrB_SUCCESS);
	}

	Delta_Matrix_free(&C);
	return info;
}

// compose multiple label & relation matrices into a single matrix
// L = L0 U L1 U ... Lm
// A = L * (R0 + R1 + ... Rn) * L 
//
// if a weight attribute is specified, this function will pick which edge to 
// return given a BWM_reduce strategy
// for example, BWM_MIN returns the edge with minimum weight
// 
// A_w  = [attribute values of A]
// rows = nodes with specified labels
// in case no labels are specified rows is a dense 1 vector: [1, 1, ...1]
GrB_Info get_sub_weight_matrix
(
	GrB_Matrix *A,             // [output] matrix (EdgeIDs)
	GrB_Matrix *A_w,           // [output] matrix (weights)
	GrB_Vector *rows,          // [output] filtered rows
	const Graph *g,            // graph
	const LabelID *lbls,       // [optional] labels to consider
	unsigned short n_lbls,     // number of labels
	const RelationID *rels,    // [optional] relationships to consider
	unsigned short n_rels,     // number of relationships
	const AttributeID weight,  // weight attribute to consider
	BWM_reduce strategy,       // use either maximum or minimum weight
	bool symmetric             // build a symmetric matrix
) {
	ASSERT(g != NULL);
	ASSERT(A != NULL);
	ASSERT((lbls != NULL && n_lbls > 0) || (lbls == NULL && n_lbls == 0));
	ASSERT((rels != NULL && n_rels > 0) || (rels == NULL && n_rels == 0));

	// context for GrB operations
	compareContext ctx = {
		.g = g,                              // the input graph
		.w = weight,                         // the weight attribute to consider 
		.comp = (strategy == BWM_MAX)? -1: 1 // -1 if max, 1 if min
	};

	GrB_BinaryOp      minID         = NULL;  // gets two edge IDs and picks one
	GrB_Scalar        theta         = NULL;  // Scalar containing the context
	GrB_UnaryOp       toMatrix      = NULL;  // get any ID from vector entry
	GrB_BinaryOp      weightOp      = NULL;  // get weight from edgeID
	GrB_Type          contx_type    = NULL;  // GB equivalent of compareContext
	GrB_BinaryOp      toMatrixMin   = NULL;  // get min weight ID from vectors
	GxB_IndexBinaryOp minID_indexOP = NULL;  // minID's underlying index op 
	size_t            n             = Graph_UncompactedNodeCount(g);

	GrB_Type_new(&contx_type, sizeof(compareContext));

	if(weight == ATTRIBUTE_ID_NONE) {
		// weight attribute wasn't specified, use a dummy weight with value: 0
		minID = GrB_FIRST_UINT64;
		GrB_UnaryOp_new (
			&toMatrix, (GxB_unary_function) _reduceToMatrixAny, GrB_UINT64, 
			GrB_UINT64
		) ;
	} else {
		GrB_Scalar_new(&theta, contx_type);
		GrB_Scalar_setElement_UDT(theta, (void *) &ctx);
		GrB_BinaryOp_new(
			&weightOp, (GxB_binary_function) _getAttFromID, 
			GrB_FP64, GrB_UINT64, contx_type
		) ;

		// reduce a matrix to its minimum (or maximum) valued edge
		GrB_BinaryOp_new(
			&toMatrixMin, (GxB_binary_function) _reduceToMatrix, 
			GrB_UINT64, GrB_UINT64, contx_type
		) ;

		// pick the minimum (or maximum) valued edge from the two matricies
		GxB_IndexBinaryOp_new(&minID_indexOP,
				(GxB_index_binary_function) _pickBinary, GrB_UINT64,
				GrB_UINT64, GrB_UINT64, contx_type, NULL, NULL);

		GxB_BinaryOp_new_IndexOp(&minID, minID_indexOP, theta);
	}

	GrB_Info info        = GrB_DEFAULT;
	GrB_Matrix M         = NULL;  // temporary matrix
	GrB_Matrix _A        = NULL;  // output matrix containing EdgeIDs
	GrB_Vector _N        = NULL;  // output filtered rows
	Delta_Matrix D       = NULL;  // graph delta matrix
	GrB_Index nrows      = 0;     // number of rows in matrix
	GrB_Index ncols      = 0;     // number of columns in matrix
	GrB_Type A_type      = NULL;  // type of the matrix
	GrB_Descriptor  desc =  NULL;

	GrB_Index rows_nvals;

	GrB_Descriptor_new(&desc);
	GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST);
	GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST);

	// if no relationships are specified, use all relationships
	// can't use adj matrix since I need access to the edgeIds of all edges
	if (rels == NULL) {
		n_rels = Graph_RelationTypeCount(g);
	}

	if (n_rels == 0) {
		// graph does not have any relations, return empty matrix
		info = GrB_Matrix_new(A, GrB_UINT64, 0, 0);
		ASSERT(info == GrB_SUCCESS);
		
		if (A_w) {
			info = GrB_Matrix_new(A_w, GrB_FP64, 0, 0);
			ASSERT(info == GrB_SUCCESS);
		}

		if (rows) {
			info = GrB_Vector_new(rows, GrB_BOOL, 0);
			ASSERT(info == GrB_SUCCESS);
		}

		BWM_FREE;
		return GrB_SUCCESS;
	}

	//--------------------------------------------------------------------------
	// compute R
	//--------------------------------------------------------------------------

	ASSERT(n_rels > 0);

	RelationID rel_id = GETRELATIONID(0);
	D = Graph_GetRelationMatrix(g, rel_id, false);
	
	Delta_Matrix  *rel_ms  =  rm_calloc(n_rels, sizeof(Delta_Matrix)) ;

	for(int i = 0; i < n_rels; ++i) {
		RelationID id = GETRELATIONID(i);
		rel_ms[i] = Graph_GetRelationMatrix(g, id, false);
	}

	info = _compile_matricies_w(&_A, rel_ms, n_rels, minID);
	ASSERT(info == GrB_SUCCESS);
	rm_free(rel_ms);

	//--------------------------------------------------------------------------
	// compute L
	//--------------------------------------------------------------------------

	// create vector N denoting all nodes participating in the algorithm
	if (rows != NULL) {
		info = GrB_Vector_new(&_N, GrB_BOOL, nrows);
		ASSERT(info == GrB_SUCCESS);
	}

	// enforce labels

	_get_rows_with_labels(_N, g, lbls, n_lbls);

	info = GrB_Vector_nvals(&rows_nvals, _N);

	if (n_rels > 0){
		// A = L * A * L
		GrB_Matrix temp = NULL;
		GrB_Matrix_new(&temp, GrB_UINT64, rows_nvals, rows_nvals);
		info = GxB_Matrix_extract_Vector(
			temp, NULL, NULL, _A, _N, _N, desc);
		ASSERT(info == GrB_SUCCESS);
		info = GrB_Matrix_free(&_A);
		_A = temp;
		temp = NULL;
	}

	bool tensor_flag = true; //TODO
	// if _A has tensor entries, reduce it to a matrix
	if (tensor_flag) {
		if (weight == ATTRIBUTE_ID_NONE) {
			info = GrB_Matrix_apply(_A, NULL, NULL, toMatrix, _A, NULL);
		} else {
			info = GrB_Matrix_apply_BinaryOp2nd_UDT(_A, NULL, NULL, toMatrixMin,
					_A, (void *) (&ctx), NULL);
		}
		ASSERT(info == GrB_SUCCESS);
	}

	if (symmetric) {
		// make A symmetric A = A + At
		info = GrB_Matrix_eWiseAdd_BinaryOp(_A, NULL, NULL, minID, _A, _A,
				GrB_DESC_T1);
		ASSERT(info == GrB_SUCCESS);
	}

	if (rows != NULL) {
		info = GrB_Vector_resize(_N, n);
		ASSERT(info == GrB_SUCCESS);
	}

	//--------------------------------------------------------------------------
	// compute weight matrix
	//--------------------------------------------------------------------------

	if (A_w) {
		info = GrB_Matrix_new(A_w, GrB_FP64, rows_nvals, rows_nvals);
		ASSERT(info == GrB_SUCCESS);

		if (weight == ATTRIBUTE_ID_NONE) {
			// if no weight specified, weights are zero
			info = GrB_Matrix_assign_FP64(
				*A_w, _A, NULL, 0.0, GrB_ALL, n, GrB_ALL, n, GrB_DESC_S
			);
		} else {
			// get the weight value from the edge ids
			info = GrB_Matrix_apply_BinaryOp2nd_UDT(
				*A_w, NULL, NULL, weightOp, _A, (void *) (&ctx), NULL
			);
		}
		ASSERT(info == GrB_SUCCESS);
	}

	if(n_lbls == 0){
		// get rid of extra unused rows and columns
		info = GrB_Matrix_resize(_A, n, n);
		ASSERT(info == GrB_SUCCESS);
	}

	if(rows != NULL) {
		info = GrB_Vector_resize(_N, n);
		ASSERT(info == GrB_SUCCESS);
	}

	// set outputs
	*A = _A;
	if (rows) *rows = _N;
	_A = NULL;
	_N = NULL;

	BWM_FREE;
	return info;
}

