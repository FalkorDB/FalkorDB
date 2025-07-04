/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "./internal.h"

// the ith relationID is i if no relation is given, and rels[i] if it is.
#define GETRELATIONID(i) ((rels)? rels[i] : i)
#define SI_VAL_IS_NUM(X) \
	(((X) != ATTRIBUTE_NOTFOUND) && ((SI_NUMERIC & SI_TYPE(*(X))) != 0))

// Frees the build weighted matrix workspace
#define BWM_FREE                                                 \
{                                                                \
 	GrB_UnaryOp_free(&toMatrix);                                 \
 	if (weight != ATTRIBUTE_ID_NONE) GrB_BinaryOp_free(&minID);  \
+	GxB_IndexBinaryOp_free(&minID_indexOP);                      \
 	GrB_BinaryOp_free(&toMatrixMin);                             \
 	GrB_BinaryOp_free(&weightOp);                                \
 	GrB_Scalar_free(&theta);                                     \
 	GrB_IndexUnaryOp_free(&hasAtt);                              \
+	GrB_Type_free(&contx_type);                                  \
 }

typedef struct
{
	const Graph *g;  // Graph
	AttributeID w;   // Attribute used as weight
	int comp;        // -1 if max, 1 if min
} compareContext;

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
)
{
	*z = *x;
	Edge _x, _y;
	Graph_GetEdge(ctx->g, (EdgeID) (*x), &_x);
	Graph_GetEdge(ctx->g, (EdgeID) (*y), &_y);
	SIValue *xv = GraphEntity_GetProperty((GraphEntity *) &_x, ctx->w);
	SIValue *yv = GraphEntity_GetProperty((GraphEntity *) &_y, ctx->w);

	if(!SI_VAL_IS_NUM(xv) || 
		(SI_VAL_IS_NUM(yv) && SIValue_Compare(*xv, *yv, NULL) == ctx->comp)) {
		*z = *y;
	}
}

static void _reduceToMatrix
(
	uint64_t *z,
	const uint64_t *x,
	const compareContext *ctx
)
{
	if(SCALAR_ENTRY(*x)) {
		*z = *x;
	} else { // Find the minimum weighted edge in the vector
		GrB_Vector _v = AS_VECTOR(*x);
		GxB_Iterator i = NULL;

		GrB_Info info = GxB_Iterator_new(&i);
		ASSERT(info == GrB_SUCCESS)
		info = GxB_Vector_Iterator_attach(i, _v, NULL);
		ASSERT(info == GrB_SUCCESS)
		info = GxB_Vector_Iterator_seek(i, 0);
		ASSERT(info == GrB_SUCCESS)

		Edge currE;
		EdgeID minID    = (EdgeID) GxB_Vector_Iterator_getIndex(i);
		SIValue *currV  = NULL;

		// -infinity if max or +infinity if min
		SIValue minV    = SI_DoubleVal(ctx->comp * INFINITY);
		SIValue tempV;

		Graph_GetEdge(ctx->g, minID, &currE);
		currV = GraphEntity_GetProperty((GraphEntity *) &currE, ctx->w);
		info = GxB_Vector_Iterator_next(i);

		// Treat edges without the attribute or with a non-numeric attribute
		// as infinite length
		if(SI_VAL_IS_NUM(currV)) {
			minV = *currV;
		}

		while(info != GxB_EXHAUSTED){
			EdgeID CurrID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
			Graph_GetEdge(ctx->g, CurrID, &currE);
			currV = GraphEntity_GetProperty((GraphEntity *) &currE, ctx->w);
			if(SI_VAL_IS_NUM(currV) && 
				SIValue_Compare(minV, *currV, NULL) == ctx->comp){
				minV = *currV;
				minID = CurrID;
			}
			info = GxB_Vector_Iterator_next(i);
		}
		
		*z = (uint64_t) minID;
		GxB_Iterator_free(&i);
	}
}

// returns true if the edge has the given attribute and it is numerical.
static void _edgeHasAttribute
(
	bool *z,
	const uint64_t *x,
	GrB_Index i,
	GrB_Index j,
	const compareContext *ctx
)
{
	Edge _x;
	Graph_GetEdge(ctx->g, (EdgeID) (*x), &_x);
	SIValue *v = GraphEntity_GetProperty((GraphEntity *) &_x, ctx->w);
	*z = SI_VAL_IS_NUM(v);
}

// returns the double value of the given attribute given an edgeId
// precondition: matrix must not contain edges which don't have the attribut
void _getAttFromID
(
	double *z, 
	const uint64_t *x, 
	const compareContext *ctx
)
{
	Edge _x;
	Graph_GetEdge(ctx->g, (EdgeID) (*x), &_x);
	SIValue *v = GraphEntity_GetProperty((GraphEntity *) &_x, ctx->w);

	if(SI_VAL_IS_NUM(v)) {
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
	uint64_t *z,         // single Edge ID
	const uint64_t *x    // possibly a vector entry
)
{
	if(SCALAR_ENTRY(*x)){
		*z = *x;
	} else {
		GrB_Vector _v = AS_VECTOR(*x);
		GxB_Iterator i = NULL;
		GrB_Info info = GxB_Iterator_new(&i);
		ASSERT(info == GrB_SUCCESS);
		info = GxB_Vector_Iterator_attach(i, _v, NULL);
		ASSERT(info == GrB_SUCCESS);

		// find the first edge in the vector
		info = GxB_Vector_Iterator_seek(i, 0);
		ASSERT(info == GrB_SUCCESS);

		// get the first edge ID in the vector
		EdgeID minID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
		info = GxB_Iterator_free(&i);
		ASSERT(info == GrB_SUCCESS);
		*z = (uint64_t) minID;
	}
}

// compose multiple label & relation matrices into a single matrix
// L = L0 U L1 U ... Lm
// A = L * (R0 + R1 + ... Rn) * L 
//
// if a weight attribute is specified, this function will pick which Edge to 
// return given a BWM_reduce strategy. For example, BWM_MIN returns the edge 
// with minimum weight.
// 
// A_w = [Attribute values of A]
// rows = L's main diagonal
// in case no labels are specified rows is a dense 1 vector: [1,1,...1]
GrB_Info Build_Weighted_Matrix
(
	GrB_Matrix *A,             // [output] matrix (EdgeIDs)
	GrB_Matrix *A_w,           // [output] matrix (weights)
	GrB_Vector *rows,          // [output] filtered rows
	const Graph *g,            // graph
	const LabelID *lbls,       // [optional] labels to consider
	unsigned short n_lbls,     // number of labels
	const RelationID *rels,    // [optional] relationships to consider
	unsigned short n_rels,     // number of relationships
	const AttributeID weight,  // attribute to return
	BWM_reduce strategy,       // decides how singular returned edge id is picked
	bool symmetric,            // build a symmetric matrix
	bool compact               // remove unused row & columns
) {
	compareContext ctx               = {.g = g, .w = weight, 
	                                    .comp = (strategy == BWM_MAX)? -1: 1};
	GrB_Type contx_type              = NULL;
	GxB_IndexBinaryOp minID_indexOP  = NULL;
	GrB_IndexUnaryOp hasAtt          = NULL;
	GrB_BinaryOp minID               = NULL;
	GrB_BinaryOp toMatrixMin         = NULL;
	GrB_BinaryOp weightOp            = NULL;
	GrB_UnaryOp toMatrix             = NULL;
	GrB_Scalar theta                 = NULL;

	ASSERT(g != NULL);
	ASSERT(A != NULL);
	ASSERT((lbls != NULL && n_lbls > 0) || (lbls == NULL && n_lbls == 0));
	ASSERT((rels != NULL && n_rels > 0) || (rels == NULL && n_rels == 0));

	GrB_Type_new(&contx_type, sizeof(compareContext));

	if(weight == ATTRIBUTE_ID_NONE) {
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

		// pick the minimum (or maximum) valued edge from the two matricies.
		GxB_IndexBinaryOp_new(
			&minID_indexOP, (GxB_index_binary_function) _compare_EdgeID_value,
			GrB_UINT64, GrB_UINT64, GrB_UINT64, contx_type, NULL, NULL) ;
		GxB_BinaryOp_new_IndexOp (&minID, minID_indexOP, theta) ;
		
		GrB_IndexUnaryOp_new(
			&hasAtt, (GxB_index_unary_function) _edgeHasAttribute, 
			GrB_BOOL, GrB_UINT64, contx_type
		) ;
	}

	GrB_Info info    =  GrB_DEFAULT;
	Delta_Matrix D   =  NULL;         // graph delta matrix
	GrB_Index nrows  =  0;            // number of rows in matrix
	GrB_Index ncols  =  0;            // number of columns in matrix
	GrB_Type A_type  =  NULL;         // type of the matrix
	GrB_Matrix _A    =  NULL;         // output matrix containing EdgeIDs
	GrB_Matrix _A_w  =  NULL;         // output matrix containing Weights
	GrB_Vector _N    =  NULL;         // output filtered rows
	GrB_Matrix M     =  NULL;         // temporary matrix

	// if no relationships are specified, use all relationships
	// can't use adj matrix since I need access to the edgeIds of all edges
	if(rels == NULL) {
		n_rels = Graph_RelationTypeCount(g);
	}

	if(n_rels == 0) { 
		// Graph does not have any relations. Return empty matrix.
		info = GrB_Matrix_new(A, GrB_UINT64, 0, 0);
		ASSERT(info == GrB_SUCCESS);
		if(A_w) {
			info = GrB_Matrix_new(A_w, GrB_FP64, 0, 0);
			ASSERT(info == GrB_SUCCESS);
		}
		if(rows) {
			info = GrB_Vector_new(rows, GrB_BOOL, 0);
			ASSERT(info == GrB_SUCCESS);
		}
		BWM_FREE;
		return GrB_SUCCESS;
	}
	RelationID id = GETRELATIONID(0);
	D = Graph_GetRelationMatrix(g, id, false);

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

	if(Graph_RelationshipContainsMultiEdge(g, id)) {
		if(weight == ATTRIBUTE_ID_NONE) {
			info = GrB_Matrix_apply(
				_A, NULL, NULL, toMatrix, _A, NULL
			) ;
		} else {
			info = GrB_Matrix_apply_BinaryOp2nd_UDT(
				_A, NULL, NULL, toMatrixMin, _A, (void *) (&ctx), NULL
			) ;
		}
		ASSERT(info == GrB_SUCCESS);
	}

	// in case there are multiple relation types, include them in A
	for(unsigned short i = 1; i < n_rels; i++) {
		id = GETRELATIONID(i);
		D = Graph_GetRelationMatrix(g, id, false);
		info = Delta_Matrix_export(&M, D, GrB_UINT64);
		ASSERT(info == GrB_SUCCESS);

		if(Graph_RelationshipContainsMultiEdge(g, id)) {
			if(weight == ATTRIBUTE_ID_NONE) {
				// apply toMatrix to get any edge and accum to also choose any 
				info = GrB_Matrix_apply(
					_A, NULL, minID, toMatrix, M, NULL);
			} else {
				// Cannot use indexbinaryOp in accum so I use eWiseAdd
				// info = GrB_Matrix_apply_BinaryOp2nd_UINT64(
				// 	_A, NULL, minID, toMatrixMin, M, (uint64_t) (&ctx), NULL
				// ) ;

				// apply to get the min (max) valued edge from the tensor
				info = GrB_Matrix_apply_BinaryOp2nd_UDT(
					M, NULL, NULL, toMatrixMin, M, (void *) (&ctx), NULL
				) ;
				ASSERT(info == GrB_SUCCESS);

				// add and pick the min (max) valued edge if there is overlap
				info = GrB_Matrix_eWiseAdd_BinaryOp(
					_A, NULL, NULL, minID, _A, M, NULL
				) ;
				ASSERT(info == GrB_SUCCESS);
			}
		} else {
			info = GrB_Matrix_eWiseAdd_BinaryOp(
				_A, NULL, NULL, minID, _A, M, NULL) ;
		}
		ASSERT(info == GrB_SUCCESS);
		GrB_free(&M);
	}

	// create vector N denoting all nodes participating in the algorithm
	if(rows != NULL) {
		info = GrB_Vector_new(&_N, GrB_BOOL, nrows);
		ASSERT(info == GrB_SUCCESS);
	}

	// enforce labels
	if(n_lbls > 0) {
		Delta_Matrix DL = Graph_GetLabelMatrix(g, lbls[0]);

		GrB_Matrix L;
		info = Delta_Matrix_export(&L, DL, GrB_BOOL);
		ASSERT(info == GrB_SUCCESS);

		// L = L U M
		for(unsigned short i = 1; i < n_lbls; i++) {
			DL = Graph_GetLabelMatrix(g, lbls[i]);

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
		if(rows != NULL) {
			info = GxB_Vector_diag(_N, L, 0, NULL);
			ASSERT(info == GrB_SUCCESS);
		}

		// free L matrix
		info = GrB_Matrix_free(&L);
		ASSERT(info == GrB_SUCCESS);
	} else if(rows != NULL) {
		// N = [1,....1]
		GrB_Scalar scalar;
		info = GrB_Scalar_new(&scalar, GrB_BOOL);
		ASSERT(info == GrB_SUCCESS);

		info = GxB_Scalar_setElement_BOOL(scalar, true);
		ASSERT(info == GrB_SUCCESS);

		info = GrB_Vector_assign_Scalar(_N, NULL, NULL, scalar, GrB_ALL, nrows,
				NULL);
		ASSERT(info == GrB_SUCCESS);
    
		info = GrB_free(&scalar);
		ASSERT(info == GrB_SUCCESS);
	}

	if(symmetric) {
		// make A symmetric A = A + At
		info = GrB_Matrix_eWiseAdd_BinaryOp(_A, NULL, NULL, minID,
				_A, _A, GrB_DESC_T1);
		ASSERT(info == GrB_SUCCESS);
	}

	size_t n = nrows;
	if(compact) {
		// determine the number of nodes in the graph
		// this includes deleted nodes
		n = Graph_UncompactedNodeCount(g);

		// get rid of extra unused rows and columns
		GrB_Info info = GrB_Matrix_resize(_A, n, n);
		ASSERT(info == GrB_SUCCESS);

		if(rows != NULL) {
			info = GrB_Vector_resize(_N, n);
			ASSERT(info == GrB_SUCCESS);
		}
	}

	if(A_w) {
		info = GrB_Matrix_new(A_w, GrB_FP64, n, n);
		ASSERT(info == GrB_SUCCESS);
		if(weight == ATTRIBUTE_ID_NONE) {
			// if no weight specified, weights are zero.
			info = GrB_Matrix_assign_FP64(
				*A_w, _A, NULL, 0.0, GrB_ALL, n, GrB_ALL, n, GrB_DESC_S
			) ;
		} else {
			// get the weight value from the edge ids.
			info = GrB_Matrix_apply_BinaryOp2nd_UDT(
				*A_w, NULL, NULL, weightOp, _A, (void *) (&ctx), NULL
			);
		}
		ASSERT(info == GrB_SUCCESS);
	}

	// set outputs
	*A = _A;
	if(rows) *rows = _N;
	_A = NULL;
	_N = NULL;
	BWM_FREE;
	return info;
}

