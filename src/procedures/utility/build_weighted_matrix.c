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
	const Graph *g;  // graph
	AttributeID w;   // attribute used as weight
	int comp;        // -1 if max, 1 if min
} compareContext;

// collapse the entries in a tensor down to a single value by finding the lowest 
// or highest weight edge
static void _reduceToMatrix
(
	uint64_t *z,               // [output] scalar entry with min/max weight
	const uint64_t *x,         // [input]  possible vectoor entry
	const compareContext *ctx  // context
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

		EdgeID minID   = (EdgeID) GxB_Vector_Iterator_getIndex(i);
		SIValue *currV = NULL;

		// -infinity if max or +infinity if min
		SIValue minV = SI_DoubleVal(ctx->comp * INFINITY);
		SIValue tempV;

		Edge currE;
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

// returns the double value of the given attribute given an edgeId
void _getAttFromID
(
	double *z,                 // [output] edge weight
	const uint64_t *x,         // edge id
	const compareContext *ctx  // theta
) {
	Edge e;
	ASSERT(SCALAR_ENTRY(*x));
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

// collapse the entries in a two tensors down to a single value by finding the 
// lowest or highest weight edge
static void _pickBinary
(
	uint64_t *z,               // [output] single edgeId (min or max)
	const uint64_t *x,         // [input]  scalar or vector entry
	GrB_Index ix,              // unused
	GrB_Index jx,              // unused
	const uint64_t *y,         // [input]  scalar or vector entry
	GrB_Index iy,              // unused
	GrB_Index jy,              // unused
	const compareContext *ctx  // context
) {
	GrB_Vector _v = AS_VECTOR(*x);

	// -infinity if max or +infinity if min
	SIValue minV = SI_DoubleVal(ctx->comp * INFINITY);
	EdgeID minID;

	if(SCALAR_ENTRY(*x)) {
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
	
	if(SCALAR_ENTRY(*y)) {
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

// compose multiple label & relation matrices into a single matrix
// L = L0 U L1 U ... Lm
// A = L * (R0 + R1 + ... Rn) * L 
//
// if a weight attribute is specified, this function will pick which edge to 
// return given a BWM_reduce_strategy
// for example, BWM_MIN returns the edge with minimum weight
// 
// A_w  = [attribute values of A]
// rows = nodes with specified labels
// in case no labels are specified rows is a dense 1 vector: [1, 1, ...1]
GrB_Info Build_Weighted_Matrix
(
	GrB_Matrix *A,                 // [output] matrix (EdgeIDs)
	GrB_Matrix *A_w,               // [output] matrix (weights)
	GrB_Vector *rows,              // [output] filtered rows
	const Graph *g,                // graph
	const LabelID *lbls,           // [optional] labels to consider
	unsigned short n_lbls,         // number of labels
	const RelationID *rels,        // [optional] relationships to consider
	unsigned short n_rels,         // number of relationships
	const AttributeID weight,      // weight attribute to consider
	BWM_reduce_strategy strategy,  // use either maximum or minimum weight
	bool symmetric,                // build a symmetric matrix
	bool compact                   // remove unused row & columns
) {
	ASSERT(g != NULL);
	ASSERT(A != NULL);
	ASSERT((lbls != NULL && n_lbls > 0) || (lbls == NULL && n_lbls == 0));
	ASSERT((rels != NULL && n_rels > 0) || (rels == NULL && n_rels == 0));

	// context for GrB operations
	compareContext ctx = {
		.g = g,                               // the input graph
		.w = weight,                          // weight attribute to consider
		.comp = (strategy == BWM_MAX)? -1: 1  // -1 if max, 1 if min
	};

	GrB_BinaryOp      minID         = NULL;  // gets two edge IDs and picks one
	GrB_Scalar        theta         = NULL;  // scalar containing the context
	GrB_UnaryOp       toMatrix      = NULL;  // get any ID from vector entry
	GrB_BinaryOp      weightOp      = NULL;  // get weight from edgeID
	GrB_Type          contx_type    = NULL;  // GB equivalent of compareContext
	GrB_BinaryOp      toMatrixMin   = NULL;  // get min weight ID from vectors
	GxB_IndexBinaryOp minID_indexOP = NULL;  // minID's underlying index op 
	size_t n;

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
				(GxB_index_binary_function) _pickBinary, GrB_UINT64,
				GrB_UINT64, GrB_UINT64, contx_type, NULL, NULL);

		GxB_BinaryOp_new_IndexOp(&minID, minID_indexOP, theta);
	}

	GrB_Info info;
	GrB_Matrix M    = NULL;  // temporary matrix
	GrB_Matrix _A   = NULL;  // output matrix containing EdgeIDs
	GrB_Vector _N   = NULL;  // output filtered rows
	Delta_Matrix D  = NULL;  // graph delta matrix
	GrB_Index nrows = 0;     // number of rows in matrix
	GrB_Index ncols = 0;     // number of columns in matrix
	GrB_Type A_type = NULL;  // type of the matrix

	// if no relationships are specified, use all relationships
	// can't use adj matrix since we need access to the edgeIds of all edges
	if (rels == NULL) {
		n_rels = Graph_RelationTypeCount(g);
	}

	if (n_rels == 0) {
		n = compact? Graph_UncompactedNodeCount(g) : Graph_RequiredMatrixDim(g);
		// graph does not have any relations, return empty matrix
		info = GrB_Matrix_new(A, GrB_UINT64, n, n);
		ASSERT(info == GrB_SUCCESS);
		
		if (A_w) {
			info = GrB_Matrix_new(A_w, GrB_FP64, n, n);
			ASSERT(info == GrB_SUCCESS);
		}

		if (rows) {
			info = GrB_Vector_new(rows, GrB_BOOL, n);
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
	ASSERT(D != NULL);

	info = Delta_Matrix_export(&_A, D, GrB_UINT64);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_nrows(&nrows, _A);
	ASSERT(info == GrB_SUCCESS);

	info = GrB_Matrix_ncols(&ncols, _A);
	ASSERT(info == GrB_SUCCESS);

	info = GxB_Matrix_type(&A_type, _A);
	ASSERT(info == GrB_SUCCESS);

	// expecting a square matrix
	ASSERT(nrows == ncols);

	bool multiEdgeFlag = Graph_RelationshipContainsMultiEdge(g, rel_id);

	for (unsigned short i = 1; i < n_rels; i++) {
		rel_id = GETRELATIONID(i);
		multiEdgeFlag = multiEdgeFlag || 
			Graph_RelationshipContainsMultiEdge(g, rel_id);

		D = Graph_GetRelationMatrix(g, rel_id, false);
		ASSERT(D != NULL);

		info = Delta_Matrix_export(&M, D, GrB_UINT64);
		ASSERT(info == GrB_SUCCESS);

		// add and pick the min (max) valued edge if there is overlap
		info = GrB_Matrix_eWiseAdd_BinaryOp(_A, NULL, NULL, minID, _A, M, NULL);
		ASSERT(info == GrB_SUCCESS);
		GrB_free(&M);
	}

	// if _A has tensor entries, reduce it to a matrix
	if (multiEdgeFlag) {
		if (weight == ATTRIBUTE_ID_NONE) {
			info = GrB_Matrix_apply(_A, NULL, NULL, toMatrix, _A, NULL);
		} else {
			info = GrB_Matrix_apply_BinaryOp2nd_UDT(_A, NULL, NULL, toMatrixMin,
					_A, (void *) (&ctx), NULL);
		}
		ASSERT(info == GrB_SUCCESS);
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
		ASSERT(DL != NULL);

		GrB_Matrix L;
		info = Delta_Matrix_export(&L, DL, GrB_BOOL);
		ASSERT(info == GrB_SUCCESS);

		// L = L U M
		for (unsigned short i = 1; i < n_lbls; i++) {
			DL = Graph_GetLabelMatrix(g, lbls[i]);
			ASSERT(DL != NULL);

			info = Delta_Matrix_export(&M, DL, GrB_BOOL);
			ASSERT(info == GrB_SUCCESS);

			info = GrB_Matrix_eWiseAdd_Semiring(L, NULL, NULL,
					GxB_ANY_PAIR_BOOL, L, M, NULL);
			ASSERT(info == GrB_SUCCESS);

			GrB_Matrix_free(&M);
		}

		// A = L * A * L
		info = GrB_mxm(_A, NULL, NULL, GxB_ANY_SECOND_UINT64, L, _A, NULL);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_mxm(_A, NULL, NULL, GxB_ANY_FIRST_UINT64, _A, L, NULL);
		ASSERT(info == GrB_SUCCESS);

		// set N to L's main diagonal denoting all participating nodes 
		if (_N != NULL) {
			info = GxB_Vector_diag(_N, L, 0, NULL);
			ASSERT(info == GrB_SUCCESS);
		}

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

	n = nrows;
	if (compact) {
		// determine the number of nodes in the graph
		// this includes deleted nodes
		n = Graph_UncompactedNodeCount(g);

		// get rid of extra unused rows and columns
		GrB_Info info = GrB_Matrix_resize(_A, n, n);
		ASSERT(info == GrB_SUCCESS);

		if (rows != NULL) {
			info = GrB_Vector_resize(_N, n);
			ASSERT(info == GrB_SUCCESS);
		}
	}

	//--------------------------------------------------------------------------
	// compute weight matrix
	//--------------------------------------------------------------------------

	if (A_w) {
		info = GrB_Matrix_new(A_w, GrB_FP64, n, n);
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

	// set outputs
	*A = _A;
	if (rows) *rows = _N;

	_A = NULL;
	_N = NULL;

	BWM_FREE;
	return info;
}

