/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "./internal.h"

#define GETRELATIONID(i) ((rels)? rels[i] : i)

#define BWM_FREE 													\
{																	\
	GrB_UnaryOp_free(&toMatrix);									\
	if (weight != ATTRIBUTE_ID_NONE) GrB_BinaryOp_free(&minID);		\
	GrB_BinaryOp_free(&toMatrixMin);								\
	GrB_BinaryOp_free(&weightOp);									\
	GrB_Scalar_free(&theta);										\
	GrB_IndexUnaryOp_free(&hasAtt);									\
}
typedef struct
{
	const Graph *g;		// Graph
	AttributeID w;	// Weight Edge
	RelationID rel;
} compareContext;

void _selectAttribute
(
	bool *z,
	const uint64_t *x,
	GrB_Index i,
	GrB_Index j,
	const uint64_t *y
)
{
	compareContext *contx = (compareContext *) (*y);
	Edge _x;
	Graph_GetEdge(contx->g, (EdgeID) (*x), &_x);
	SIValue *v = GraphEntity_GetProperty((GraphEntity *) &_x, contx->w);
	*z = (v != ATTRIBUTE_NOTFOUND);
}
#if 0
void _getAttFromID
(
	double *z, 
	const uint64_t *x, 
	const uint64_t *y
)
{
	compareContext *contx = (compareContext *) (*y);
	Edge _x, _y;
	Graph_GetEdge(contx->g, (EdgeID) (*x), &_x);
	SIValue *v = AttributeSet_Get(*_x.attributes, contx->w);
	SIValue_ToDouble(v, z);
}
#else
void _getAttFromID
(
	int64_t *z, 
	const uint64_t *x, 
	const uint64_t *y
)
{
	compareContext *contx = (compareContext *) (*y);
	Edge _x, _y;
	Graph_GetEdge(contx->g, (EdgeID) (*x), &_x);
	SIValue *v = AttributeSet_Get(*_x.attributes, contx->w);
	ASSERT(v != ATTRIBUTE_NOTFOUND && ((SI_NUMERIC & SI_TYPE(*v)) != 0));
	*z = (int64_t) SI_GET_NUMERIC(*v);
}
#endif
void _compareAndReturnEdgeID 
(
	uint64_t *z, 
	const uint64_t *x, 
	GrB_Index ix,
	GrB_Index jx,
	const uint64_t *y,
	GrB_Index iy,
	GrB_Index jy,
	const uint64_t *theta
)
{
	*z = *x;
	compareContext *contx = (compareContext *) (*theta);
	Edge _x, _y;
	Graph_GetEdge(contx->g, (EdgeID) (*x), &_x);
	Graph_GetEdge(contx->g, (EdgeID) (*y), &_y);
	SIValue *xv = AttributeSet_Get(*_x.attributes, contx->w);
	SIValue *yv = AttributeSet_Get(*_y.attributes, contx->w);
	if(xv == ATTRIBUTE_NOTFOUND ||
		(yv != ATTRIBUTE_NOTFOUND && SIValue_Compare(*xv, *yv, NULL) == 1))
	{
		*z = *y;
	}
}

// TODO: allow user to customize the comparison function.
void _reduceToMatrixMin
(
	uint64_t *z, 
	const uint64_t *x, 
	const uint64_t *y
)
{
	if(SCALAR_ENTRY(*x))
	{
		*z = *x;
	}
	else
	{
		GrB_Vector _v = AS_VECTOR(*x);
		compareContext *contx = ((compareContext *) (*y));
		GxB_Iterator i = NULL;
		GrB_Info info = GxB_Iterator_new(&i);
		ASSERT(info == GrB_SUCCESS)
		info = GxB_Vector_Iterator_attach(i, _v, NULL);
		ASSERT(info == GrB_SUCCESS)
		info = GxB_Vector_Iterator_seek(i, 0);
		ASSERT(info == GrB_SUCCESS)
		Edge currE;
		EdgeID minID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
		Graph_GetEdge(contx->g, minID, &currE);
		SIValue *currV = NULL, minV = SI_DoubleVal(INFINITY), tempV;
		currV = AttributeSet_Get(*currE.attributes, contx->w);
		info = GxB_Vector_Iterator_next(i);
		if(currV != ATTRIBUTE_NOTFOUND)
		{
			minV = *currV;
		}
		while(info != GxB_EXHAUSTED)
		{
			EdgeID CurrID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
			Graph_GetEdge(contx->g, CurrID, &currE);
			currV = AttributeSet_Get(*currE.attributes, contx->w);
			ASSERT(currV == ATTRIBUTE_NOTFOUND 
				|| ((SI_NUMERIC & SI_TYPE(*currV)) != 0));
			if(currV != ATTRIBUTE_NOTFOUND && 
				SIValue_Compare(minV, *currV, NULL) == 1)
			{
				minV = *currV;
				minID = CurrID;
			}
			info = GxB_Vector_Iterator_next(i);
		} 
		*z = (uint64_t) minID;
		GxB_Iterator_free(&i);
	}
}

void _reduceToMatrixAny
(
	uint64_t *z, 
	const uint64_t *x
)
{
	if(SCALAR_ENTRY(*x))
	{
		*z = *x;
	}
	else
	{
		GrB_Vector _v = AS_VECTOR(*x);
		GxB_Iterator i = NULL;
		GrB_Info info = GxB_Iterator_new(&i);
		ASSERT(info == GrB_SUCCESS);
		info = GxB_Vector_Iterator_attach(i, _v, NULL);
		ASSERT(info == GrB_SUCCESS);
		info = GxB_Vector_Iterator_seek(i, 0);
		ASSERT(info == GrB_SUCCESS);
		EdgeID minID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
		info = GxB_Iterator_free(&i);
		ASSERT(info == GrB_SUCCESS);
		*z = minID;
	}
}
// compose multiple label & relation matrices into a single matrix
// L = L0 U L1 U ... Lm
// A = L * (R0 + R1 + ... Rn) * L
//
// rows = L's main diagonal
// in case no labels are specified rows is a dense 1 vector: [1,1,...1]
GrB_Info Build_Weighted_Matrix
(
	GrB_Matrix *A,           // [output] matrix (EdgeIDs)
	GrB_Matrix *A_w,         // [output] matrix (weights)
	GrB_Vector *rows,        // [output] filtered rows
	const Graph *g,          // graph
	const LabelID *lbls,     // [optional] labels to consider
	unsigned short n_lbls,   // number of labels
	const RelationID *rels,  // [optional] relationships to consider
	unsigned short n_rels,   // number of relationships
	const AttributeID weight,// attribute to return
	bool symmetric,          // build a symmetric matrix
	bool compact            // remove unused row & columns
) {
	compareContext contx = {.g = g, .w = weight};
	GxB_IndexBinaryOp minID_indexOP = NULL;
	GrB_IndexUnaryOp hasAtt = NULL;
	GrB_BinaryOp minID = NULL, toMatrixMin = NULL, weightOp = NULL;
	GrB_UnaryOp toMatrix = NULL;
	GrB_Scalar theta = NULL;
	ASSERT(g != NULL);
	ASSERT(A != NULL);
	ASSERT((lbls != NULL && n_lbls > 0) || (lbls == NULL && n_lbls == 0));
	ASSERT((rels != NULL && n_rels > 0) || (rels == NULL && n_rels == 0));

	if(weight == ATTRIBUTE_ID_NONE)
	{
		minID = GrB_SECOND_UINT64;
		GrB_UnaryOp_new (
			&toMatrix, (GxB_unary_function) _reduceToMatrixAny, GrB_UINT64, 
			GrB_UINT64
		) ;
	}
	else
	{
		GrB_Scalar_new(&theta, GrB_UINT64);
		GrB_Scalar_setElement_UINT64(theta, (uint64_t) (&contx));
		GrB_BinaryOp_new(
			&weightOp, (GxB_binary_function) _getAttFromID, 
			GrB_INT64, GrB_UINT64, GrB_UINT64
		) ;
		GrB_BinaryOp_new(
			&toMatrixMin, (GxB_binary_function) _reduceToMatrixMin, 
			GrB_UINT64, GrB_UINT64, GrB_UINT64
		) ;
		GxB_IndexBinaryOp_new(
			&minID_indexOP, (GxB_index_binary_function) _compareAndReturnEdgeID,
			GrB_UINT64, GrB_UINT64, GrB_UINT64, GrB_UINT64, NULL, NULL) ;
		GxB_BinaryOp_new_IndexOp (&minID, minID_indexOP, theta) ;
		GrB_IndexUnaryOp_new(
			&hasAtt, (GxB_index_unary_function) _selectAttribute, 
			GrB_BOOL, GrB_UINT64, GrB_UINT64
		) ;
	}

	GrB_Info info;
	Delta_Matrix D = NULL;   	// graph delta matrix
	GrB_Index nrows;  			// number of rows in matrix
	GrB_Index ncols;  			// number of columns in matrix
	GrB_Type A_type = NULL;  	// type of the matrix
	GrB_Matrix _A = NULL;    	// output matrix
	GrB_Matrix _A_w = NULL;    	// output matrix
	GrB_Vector _N = NULL;    	// output filtered rows
	GrB_Matrix M = NULL;		// temporary matrix

	// TODO do I need a read lock?
	// Graph_AcquireReadLock(g);
	
	// if no relationships are specified, use all relationships
	if(rels == NULL)
	{
		n_rels = Graph_RelationTypeCount(g);
	}
	if(n_rels == 0) { 
		// empty graph, return empty matrix
		info = GrB_Matrix_new(A, GrB_UINT64, 0, 0);
		ASSERT(info == GrB_SUCCESS);
		if(A_w) {
			info = GrB_Matrix_new(A_w, GrB_INT64, 0, 0);
			ASSERT(info == GrB_SUCCESS);
		}
		if(rows) {
			info = GrB_Vector_new(rows, GrB_BOOL, 0);
			ASSERT(info == GrB_SUCCESS);
		}
		BWM_FREE;
		// TODO do I need a read lock?
		// Graph_ReleaseLock(g);
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

	if(Graph_RelationshipContainsMultiEdge(g, id))
	{
		if(weight == ATTRIBUTE_ID_NONE) 
		{
			info = GrB_Matrix_apply(
				_A, NULL, NULL, toMatrix, _A, NULL
			) ;
		} 
		else 
		{
			info = GrB_Matrix_apply_BinaryOp2nd_UINT64(
				_A, NULL, NULL, toMatrixMin, _A, (uint64_t) (&contx), NULL
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

		if(Graph_RelationshipContainsMultiEdge(g, id))
		{
			if(weight == ATTRIBUTE_ID_NONE) 
			{
				info = GrB_Matrix_apply(
					_A, NULL, minID, toMatrix, M, NULL);
			} 
			else 
			{
				// Cannot use indexbinaryOp in accum so I use eWiseAdd
				// FIXME if possible.
				// info = GrB_Matrix_apply_BinaryOp2nd_UINT64(
				// 	_A, NULL, minID, toMatrixMin, M, (uint64_t) (&contx), NULL
				// ) ;
				info = GrB_Matrix_apply_BinaryOp2nd_UINT64(
					M, NULL, NULL, toMatrixMin, M, (uint64_t) (&contx), NULL
				) ;
				ASSERT(info == GrB_SUCCESS);
				info = GrB_Matrix_eWiseAdd_BinaryOp(
					_A, NULL, NULL, minID, _A, M, NULL
				) ;
				ASSERT(info == GrB_SUCCESS);
			}
		}
		else
		{
			info = GrB_Matrix_eWiseAdd_BinaryOp(
				_A, NULL, NULL, minID, _A, M, NULL) ;
		}
		ASSERT(info == GrB_SUCCESS);
		M = NULL;
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
	} 
	else if(rows != NULL) {
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
	if(weight != ATTRIBUTE_ID_NONE)
	{
		// Remove edges without the specified attribute
		info = GrB_Matrix_select_UINT64(
			_A, NULL, NULL, hasAtt, _A, (uint64_t) (&contx), NULL);
		ASSERT(info == GrB_SUCCESS);
	}
	if(A_w)
	{
		info = GrB_Matrix_new(A_w, GrB_INT64, n, n);
		ASSERT(info == GrB_SUCCESS);
		if(weight == ATTRIBUTE_ID_NONE)
		{
			// A_w = A
			info = GrB_Matrix_assign_UINT64(
				*A_w, _A, NULL, 0, GrB_ALL, n, GrB_ALL, n, NULL
			) ;
			ASSERT(info == GrB_SUCCESS);
		}
		else
		{
			info = GrB_Matrix_apply_BinaryOp2nd_UINT64(
				*A_w, NULL, NULL, weightOp, _A, (uint64_t) (&contx), NULL
			);
			ASSERT(info == GrB_SUCCESS);
		}
	}
	// set outputs
	*A = _A;
	if(rows) *rows = _N;
	_A = NULL;
	_N = NULL;
	BWM_FREE;

	// TODO do I need a read lock?
	// Graph_ReleaseLock(g);
	return info;
}

