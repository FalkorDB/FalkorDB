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
	GrB_Vector_free(&_N);                                        \
	GrB_Scalar_free(&theta);                                     \
	GrB_Type_free(&contx_type);                                  \
	GrB_Descriptor_free(&desc);                                  \
	GrB_UnaryOp_free(&toMatrix);                                 \
	GrB_BinaryOp_free(&weightOp);                                \
	GrB_BinaryOp_free(&toMatrixMin);                             \
	GxB_IndexBinaryOp_free(&minID_indexOP);                      \
}

#define COMPARE_AND_CHANGE_MINID                                 \
Graph_GetEdge(ctx->g, currID, &currE);                           \
GraphEntity_GetProperty((GraphEntity *) &currE, ctx->w, &currV); \
                                                                 \
/* only update minV if edge attribute is numeric */              \
bool replace = (SI_TYPE(currV) & SI_NUMERIC) &&                  \
	SIValue_Compare(minV, currV, NULL) == ctx->comp;             \
minV  = replace ? currV : minV;                                  \
minID = replace ? currID : minID;

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

		EdgeID minID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
		SIValue currV;

		// -infinity if max or +infinity if min
		SIValue minV = SI_DoubleVal(ctx->comp * INFINITY);
		SIValue tempV;

		Edge currE;
		Graph_GetEdge(ctx->g, minID, &currE);
		GraphEntity_GetProperty ((GraphEntity *) &currE, ctx->w, &currV) ;
		info = GxB_Vector_Iterator_next(i);

		// treat edges without the attribute or with a non-numeric attribute
		// as infinite length
		if (SI_TYPE(currV) & SI_NUMERIC) {
			minV = currV;
		}

		while (info != GxB_EXHAUSTED) {
			EdgeID currID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
			COMPARE_AND_CHANGE_MINID;
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

	SIValue v ;
	GraphEntity_GetProperty ((GraphEntity *) &e, ctx->w, &v) ;

	if(SI_TYPE(v) & SI_NUMERIC) {
		int info = SIValue_ToDouble(&v, z);
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

// collapse the entries in a two matrices down to a single value by finding the 
// lowest or highest weight entry
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
	uint64_t _x;
	Edge currE;
	EdgeID minID;
	EdgeID currID;
	GrB_Info info;
	struct GB_Iterator_opaque _i;  // stack allocate the iterator

	SIValue currV ;
	GxB_Iterator i     = &_i;
	GrB_Vector   _v    = GrB_NULL;
	SIValue      minV  = SI_DoubleVal(ctx->comp * INFINITY); // -inf if max or +inf min

	//--------------------------------------------------------------------------
	// search for min / max weight attribute on both x and y
	//--------------------------------------------------------------------------

	uint64_t operands[2] = {*x, *y};

	for (int k = 0; k < 2; k++) {
		_x = operands[k];

		if (SCALAR_ENTRY(_x)) {
			currID = (EdgeID) _x;
			COMPARE_AND_CHANGE_MINID;
		} else {
			// find the minimum weighted edge in the vector
			_v = AS_VECTOR(_x);

			info = GxB_Vector_Iterator_attach(i, _v, NULL);
			ASSERT(info == GrB_SUCCESS);

			info = GxB_Vector_Iterator_seek(i, 0);
			ASSERT(info == GrB_SUCCESS);
			
			while (info != GxB_EXHAUSTED) {
				currID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
				COMPARE_AND_CHANGE_MINID;

				info = GxB_Vector_Iterator_next(i);
			}
		}
	}

	// in case minV wasn't modified
	// (both x and y weight attribute isn't numeric)
	// set minID to the last set edgeID
	if (minV.doubleval == (ctx->comp * INFINITY)) {
		minID = currID;
	}

	// set the entry
	*z = minID;
}

// compose multiple label & relation matrices into a single matrix
// L = L0 U L1 U ... Lm
// A = (R0 + R1 + ... Rn) (compressed to only include the rows/cols from L)
//
// if a weight attribute is specified, this function will pick which edge to 
// return given a BWM_reduce_strategy
// for example, BWM_MIN returns the edge with minimum weight
// 
// A_w  = [attribute values of A]
// rows = nodes with specified labels
// in case no labels are specified rows is a dense 1 vector: [1, 1, ...1]
GrB_Info get_sub_weight_matrix
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
	bool symmetric                 // build a symmetric matrix
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
				(GxB_index_binary_function) _pickBinary, GrB_UINT64,
				GrB_UINT64, GrB_UINT64, contx_type, NULL, NULL);

		GxB_BinaryOp_new_IndexOp(&minID, minID_indexOP, theta);
	}

	GrB_Matrix     M           = NULL;  // temporary matrix
	GrB_Matrix     _A          = NULL;  // output matrix containing EdgeIDs
	GrB_Vector     _N          = NULL;  // output filtered rows
	Delta_Matrix   D           = NULL;  // graph delta matrix
	GrB_Index      nrows       = 0;     // number of rows in matrix
	GrB_Index      rows_nvals  = 0;     // number of rows being returned in matrix
	GrB_Descriptor desc        = NULL;  // Use row and column indecies 

	bool compact = false;

	nrows = Graph_RequiredMatrixDim(g);
	// create vector N denoting all nodes participating in the algorithm
	GrB_OK (GrB_Vector_new(&_N, GrB_BOOL, nrows));

	// enforce labels
	if (n_lbls > 0) {
		compact = true;
		Delta_Matrix DL = Graph_GetLabelMatrix(g, lbls[0]);
		ASSERT(DL != NULL);

		GrB_Matrix L;
		GrB_OK (Delta_Matrix_export(&L, DL, GrB_BOOL));

		// L = L U M
		for (unsigned short i = 1; i < n_lbls; i++) {
			DL = Graph_GetLabelMatrix(g, lbls[i]);
			ASSERT(DL != NULL);

			GrB_OK (Delta_Matrix_export(&M, DL, GrB_BOOL));

			GrB_OK (GrB_Matrix_eWiseAdd_Semiring(L, NULL, NULL,
					GxB_ANY_PAIR_BOOL, L, M, NULL));

			GrB_Matrix_free(&M);
		}

		// set N to L's main diagonal denoting all participating nodes 
		GrB_OK (GxB_Vector_diag(_N, L, 0, NULL));

		// free L matrix
		GrB_OK (GrB_Matrix_free(&L));
	} else {
		GrB_OK(GrB_Vector_resize(_N, n));
		// no labels, N = present nodes
		GrB_OK (GrB_Vector_assign_BOOL(
			_N, NULL, NULL, true, GrB_ALL, n, NULL));

		// remove deleted nodes from N
		if(Graph_DeletedNodeCount(g) > 0) {
			compact = true;
			NodeID *deleted_n = NULL;
			uint64_t deleted_n_count = 0;
			Graph_DeletedNodes(g, &deleted_n, &deleted_n_count);
			
			// TODO: is there a more efficient way to remove deleted nodes?
			// this also does a whole extract operation for what is likely a 
			// small number of nodes
			for(uint64_t i = 0; i < deleted_n_count; i++) {
				// remove deleted nodes from N
				GrB_OK (GrB_Vector_removeElement(_N, deleted_n[i]));
			}
			rm_free(deleted_n);
		}
	}

	GrB_OK(GrB_Vector_resize(_N, n));

	// if no relationships are specified, use all relationships
	// can't use adj matrix since we need access to the edgeIds of all edges
	if (rels == NULL) {
		n_rels = Graph_RelationTypeCount(g);
	}

	if (n_rels == 0) {
		// graph does not have any relations, return empty matrix

		GrB_OK(GrB_Vector_nvals(&rows_nvals, _N));
		GrB_OK(GrB_Matrix_new(A, GrB_UINT64, rows_nvals, rows_nvals));

		if (A_w) {
			GrB_OK(GrB_Matrix_new(A_w, GrB_FP64, rows_nvals, rows_nvals));
		}

		if (rows) {
			*rows = _N;	
			_N = NULL;
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

	GrB_OK(Delta_Matrix_export(&_A, D, GrB_UINT64));

	bool multiEdgeFlag = Graph_RelationshipContainsMultiEdge(g, rel_id);

	for (unsigned short i = 1; i < n_rels; i++) {
		rel_id = GETRELATIONID(i);
		multiEdgeFlag = multiEdgeFlag || 
			Graph_RelationshipContainsMultiEdge(g, rel_id);

		D = Graph_GetRelationMatrix(g, rel_id, false);
		ASSERT(D != NULL);

		GrB_OK (Delta_Matrix_export(&M, D, GrB_UINT64));

		// add and pick the min (max) valued edge if there is overlap
		GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp(_A, NULL, NULL, minID, _A, M, NULL));
		GrB_free(&M);
	}

	GrB_OK(GrB_Vector_nvals(&rows_nvals, _N));
	
	if (compact) {
		// shrink A to the requested row / column sizes
		GrB_Matrix temp = NULL;
		GrB_OK(GrB_Matrix_new(&temp, GrB_UINT64, rows_nvals, rows_nvals));

		GrB_Descriptor_new(&desc);
		GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST);
		GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST);

		GrB_OK(GxB_Matrix_extract_Vector(temp, NULL, NULL, _A, _N, _N, desc));
		GrB_OK(GrB_Matrix_free(&_A));
		_A = temp;
	} else {
		// get rid of extra unused rows and columns
		n = Graph_UncompactedNodeCount(g);
		ASSERT(n == rows_nvals);
		GrB_OK(GrB_Matrix_resize(_A, n, n));
	}

	// if _A has tensor entries, reduce it to a matrix
	// these entries wouldn't be removed by the previous operation if they did
	// not intersect with any other entries, or if minID was GrB_FIRST
	if (multiEdgeFlag) {
		if (weight == ATTRIBUTE_ID_NONE) {
			GrB_OK (GrB_Matrix_apply(_A, NULL, NULL, toMatrix, _A, NULL));
		} else {
			GrB_OK (GrB_Matrix_apply_BinaryOp2nd_UDT(_A, NULL, NULL, 
				toMatrixMin, _A, (void *) (&ctx), NULL));
		}
	}

	//--------------------------------------------------------------------------
	// compute L
	//--------------------------------------------------------------------------

	if (symmetric) {
		// make A symmetric A = A + At
		GrB_OK (GrB_Matrix_eWiseAdd_BinaryOp(_A, NULL, NULL, minID, _A, _A,
				GrB_DESC_T1));
	}

	//--------------------------------------------------------------------------
	// compute weight matrix
	//--------------------------------------------------------------------------

	if (A_w) {
		GrB_OK(GrB_Matrix_new(A_w, GrB_FP64, rows_nvals, rows_nvals));

		// if no attribute was specified by the user return any EdgeID and 
		// zero weight
		if (weight == ATTRIBUTE_ID_NONE) {
			// if no weight specified, weights are zero
			GrB_OK (GrB_Matrix_assign_FP64(
				*A_w, _A, NULL, 0.0, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));
		} else {
			// get the weight value from the edge ids
			GrB_OK (GrB_Matrix_apply_BinaryOp2nd_UDT(
				*A_w, NULL, NULL, weightOp, _A, (void *) (&ctx), NULL
			));
		}
	}

	// set outputs
	*A = _A;
	_A = NULL;

	if (rows) {
		*rows = _N;
		_N = NULL;
	}

	BWM_FREE;
	return GrB_SUCCESS;
}

