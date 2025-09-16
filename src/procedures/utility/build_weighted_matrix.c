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
	if (weight != ATTRIBUTE_ID_NONE) GrB_BinaryOp_free(&reduceBiop);  \
	GrB_Vector_free(&_N);                                        \
	GrB_Scalar_free(&theta);                                     \
	GrB_Type_free(&contx_type);                                  \
	GrB_UnaryOp_free(&toMatrix);                                 \
	GrB_BinaryOp_free(&weightOp);                                \
	GrB_BinaryOp_free(&minWeightID);                             \
	GxB_IndexBinaryOp_free(&reduceIdxBiop);                      \
}

#define COMPARE_AND_CHANGE_MINID                                 \
Graph_GetEdge(ctx->g, currID, &currE);                           \
currV = GraphEntity_GetProperty((GraphEntity *) &currE, ctx->w); \
                                                                 \
/* only update minV if edge attribute is numeric */              \
bool replace = (SI_TYPE(*currV) & SI_NUMERIC) &&                 \
	SIValue_Compare(minV, *currV, NULL) == ctx->comp;            \
minV  = replace? *currV: minV;                                   \
minID = replace? currID: minID;

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
	SIValue minV = SI_DoubleVal(ctx->comp * INFINITY); // +inf min / -inf max

	if(SCALAR_ENTRY(*x)) {
		*z = *x;
	} else { // find the minimum weighted edge in the vector
		GrB_Vector _v = AS_VECTOR(*x);
		
		// vector should not be NULL at this point.
		ASSERT(_v != NULL);
		// stack allocate the iterator
		struct GB_Iterator_opaque _i;
		GxB_Iterator i = &_i;

		GrB_OK(GxB_Vector_Iterator_attach(i, _v, NULL));

		GrB_Info info = GxB_Vector_Iterator_seek(i, 0);

		Edge currE;
		SIValue *currV = NULL;
		EdgeID  minID = (EdgeID) GxB_Vector_Iterator_getIndex(i);

		while (info != GxB_EXHAUSTED) {
			EdgeID currID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
			COMPARE_AND_CHANGE_MINID;
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
	// -infinity if max or +infinity if min
	EdgeID minID;

	// stack allocate the iterator
	struct GB_Iterator_opaque _i;
	GxB_Iterator i = &_i;
	GrB_Vector _v  = GrB_NULL;
	SIValue minV   = SI_DoubleVal(ctx->comp * INFINITY);
	SIValue *currV = NULL;
	uint64_t _x    = *x;
	EdgeID currID;
	Edge currE;
	GrB_Info info;
	
	// if either are zombie values, allow the other value to be passed through
	if(*x == U64_ZOMBIE) {
		*z = *y;
		return;
	}
	if(*y == U64_ZOMBIE) {
		*z = *x;
		return;
	}

	if(SCALAR_ENTRY(_x)) {
		minID = _x;
	} else {
		// find the minimum weighted edge in the vector
		_v = AS_VECTOR(_x);

		GrB_OK(GxB_Vector_Iterator_attach(i, _v, NULL));

		info = GxB_Vector_Iterator_seek(i, 0);
		ASSERT(info == GrB_SUCCESS);

		minID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
	}

	for(int k = 0; k < 2; k++)
	{
		_x = k? *y: *x;
		if(SCALAR_ENTRY(_x)) {
			currID = (EdgeID) _x;
			COMPARE_AND_CHANGE_MINID;
		} else {
			// find the minimum weighted edge in the vector
			_v = AS_VECTOR(_x);

			if(_v == NULL) {
				continue;
			}

			GrB_OK(GxB_Vector_Iterator_attach(i, _v, NULL));

			info = GxB_Vector_Iterator_seek(i, 0);

			while (info != GxB_EXHAUSTED) {
				currID = (EdgeID) GxB_Vector_Iterator_getIndex(i);
				COMPARE_AND_CHANGE_MINID;

				info = GxB_Vector_Iterator_next(i);
			}
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

		GrB_OK(GxB_Vector_Iterator_attach(i, _v, NULL));

		// find the first edge in the vector (must be found)
		GrB_OK(GxB_Vector_Iterator_seek(i, 0));

		// get the first edge ID in the vector
		*z = GxB_Vector_Iterator_getIndex(i);
	}
}

void _add_matrix_chain
(
	GrB_Matrix *A,             // [output] matrix
	const Delta_Matrix *mats,  // matricies to consider
	unsigned short n_mats,     // number of matricies
	GrB_BinaryOp op            // Binary operator to use
) {
	ASSERT(n_mats > 0);

	Delta_Matrix C = NULL;
	GrB_Index nrows;
	GrB_Index ncols;

	GrB_OK(Delta_Matrix_nrows(&nrows, mats[0]));
	GrB_OK(Delta_Matrix_ncols(&ncols, mats[0]));
	
	if (n_mats == 1){
		// export relation matrix to A
		Delta_Matrix_export(A, mats[0]);
	} else {
		// given the semiring I am using, C will be an invalid delta matrix.
		// but it is quickly exported so it should not be an issue.
		Delta_Matrix_new(&C, GrB_UINT64, nrows, ncols, false);

		Delta_eWiseAdd(C, op, mats[0], mats[1]);

		// in case there are multiple relation types, include them in A
		for(unsigned short i = 2; i < n_mats; i++) {
			Delta_eWiseAdd(C, op, C, mats[i]);
		}

		Delta_Matrix_wait(C, true);
		*A = DELTA_MATRIX_M(C);
		DELTA_MATRIX_M(C) = NULL;
	}

	Delta_Matrix_free(&C);
}

void _combine_matricies_weighted
(
	GrB_Matrix *A,            // [output] matrix
	const Delta_Matrix *mats, // matricies to consider
	unsigned short n_mats,    // number of matricies
	const GrB_Vector rows,    // filtered rows
	const GrB_BinaryOp op     // Binary operator to use
) {
	ASSERT(A    != NULL);
	ASSERT(op   != NULL);
	ASSERT(mats != NULL);
	ASSERT(rows != NULL);
	ASSERT(n_mats > 0);

	GrB_Index nrows;
	GrB_Index nvals;
	GrB_Descriptor desc = NULL;

	GrB_OK(GrB_Descriptor_new(&desc));
	GrB_OK(GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST));
	GrB_OK(GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST));

	GrB_OK(GrB_Vector_nvals(&nvals, rows));
	GrB_OK(GrB_Vector_nvals(&nrows, rows));
	bool extractFirst = nvals < nrows /2;

	Delta_Matrix *rel_ms = NULL;
	if (extractFirst) {
		rel_ms = rm_calloc(n_mats, sizeof(Delta_Matrix));
		// extract first matrix
		for(int i = 0; i < n_mats; ++i) {
			_get_rows_delta(&rel_ms[i], mats[i], rows);
		}
	}

	_add_matrix_chain(A, rel_ms ? rel_ms : mats, n_mats, op);

	if(rel_ms != NULL) {
		for(int i = 0; i < n_mats; ++i) {
			Delta_Matrix_free(&rel_ms[i]);
		}
		rm_free(rel_ms);
	}

	GrB_Matrix temp = NULL;
	GrB_OK (GrB_Matrix_new(&temp, GrB_UINT64, nvals, nvals));	
	GrB_OK (GxB_Matrix_extract_Vector(
		temp, NULL, NULL, *A, (extractFirst? NULL: rows), rows, desc));
	GrB_free(A);
	GrB_free(&desc);
	*A = temp;
}

// -----------------------------------------------------------------------------
// Make the sum function
// -----------------------------------------------------------------------------

#define GET_VALUE(z, x)                                                        \
{                                                                              \
	Edge edge;                                                                 \
	bool _info = Graph_GetEdge(ctx->g, (x), &edge);                            \
	ASSERT(_info == true);                                                     \
	SIValue *temp = GraphEntity_GetProperty(                                   \
		(GraphEntity *) &edge, ctx->w);                                        \
	if(SI_NUMERIC & SI_TYPE(*temp)) {                                          \
		SIValue_ToDouble(temp, &(z));                                          \
	} else {                                                                   \
		(z) = FUNCTION_IDENTITY;                                               \
    }                                                                          \
}

#define FUNCTION_IDENTITY 0.0
#define ACCUM(z, x) z += x

static void _reduceToMatrixSum
(
	double *z,                 // [output] edge ID
	const double *x,           // edge ID
	GrB_Index ix,              // unused
	GrB_Index jx,              // unused
	const uint64_t *y,         // edge ID
	GrB_Index iy,              // unused
	GrB_Index jy,              // unused
	const compareContext *ctx  // context
)
TENSOR_ACCUM_BIOP(double)

#undef  FUNCTION_IDENTITY
#undef  ACCUM

#define FUNCTION_IDENTITY INFINITY
#define ACCUM(z, x) z = z <= x ? z : x

static void _reduceToMatrixMin
(
	double *z,                 // [output] edge ID
	const double *x,           // edge ID
	GrB_Index ix,              // unused
	GrB_Index jx,              // unused
	const uint64_t *y,         // edge ID
	GrB_Index iy,              // unused
	GrB_Index jy,              // unused
	const compareContext *ctx  // context
)
TENSOR_ACCUM_BIOP(double)

#undef  FUNCTION_IDENTITY
#undef  ACCUM

#define FUNCTION_IDENTITY -INFINITY
#define ACCUM(z, x) z = z >= x ? z : x

static void _reduceToMatrixMax
(
	double *z,                 // [output] edge ID
	const double *x,           // edge ID
	GrB_Index ix,              // unused
	GrB_Index jx,              // unused
	const uint64_t *y,         // edge ID
	GrB_Index iy,              // unused
	GrB_Index jy,              // unused
	const compareContext *ctx  // context
)
TENSOR_ACCUM_BIOP(double)

GrB_Info _combine_attributes
(
	GrB_Matrix *A,              // [output] matrix
	const Delta_Matrix *mats,   // matricies to consider
	unsigned short n_mats,      // number of matricies
	const GrB_Vector rows,      // filtered rows
	const GrB_BinaryOp op,      // Binary operator to use
	double zombie_value         // value to use for zombie entries
) {
	ASSERT (A    != NULL);
	ASSERT (op   != NULL);
	ASSERT (mats != NULL);
	ASSERT (rows != NULL);
	ASSERT (n_mats > 0);

	GrB_Index nrows;
	GrB_Index ncols;
	GrB_Index nvals;
	GrB_Descriptor desc     = NULL;
	GrB_Type       t        = GrB_FP64;
	GrB_Scalar     a_zomb   = NULL;
	GrB_Scalar     rel_zomb = NULL;
	GrB_Matrix     M        = NULL;
	GrB_Matrix     _A       = NULL;

	GrB_Scalar_new(&a_zomb, t);
	GrB_Scalar_new(&rel_zomb, GrB_UINT64);
	GrB_Scalar_setElement(a_zomb, zombie_value);
	GrB_Scalar_setElement(rel_zomb, U64_ZOMBIE);

	Delta_Matrix_nrows(&nrows, mats[0]);
	Delta_Matrix_ncols(&ncols, mats[0]);
	GrB_Matrix_new(&_A, t, nrows, ncols);

	GrB_OK(GrB_Descriptor_new(&desc));
	GrB_OK(GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_ROWINDEX_LIST));
	GrB_OK(GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_COLINDEX_LIST));

	GrB_OK(GrB_Vector_nvals(&nvals, rows));
	GrB_OK(GrB_Vector_nvals(&nrows, rows));
	bool extractFirst = nvals < nrows /2;

	Delta_Matrix *rel_ms = NULL;
	if (extractFirst) {
		rel_ms = rm_calloc(n_mats, sizeof(Delta_Matrix));
		for(int i = 0; i < n_mats; ++i) {
			_get_rows_delta(&rel_ms[i], mats[i], rows);
		}
	}


	for (int i = 0; i < n_mats; ++i){
		Delta_Matrix_export(&M, mats[i]);
		GxB_Matrix_eWiseUnion(_A, NULL, NULL, op, _A, a_zomb, M, rel_zomb, NULL);
		GrB_Matrix_free(&M);
	}

	if(rel_ms != NULL) {
		for(int i = 0; i < n_mats; ++i) {
			Delta_Matrix_free(&rel_ms[i]);
		}
		rm_free(rel_ms);
	}

	GrB_Matrix temp = NULL;
	GrB_OK (GrB_Matrix_new(&temp, t, nvals, nvals));	
	GrB_OK (GxB_Matrix_extract_Vector(
		temp, NULL, NULL, _A, (extractFirst? NULL: rows), rows, desc));

	GrB_free(&_A);
	GrB_free(&desc);
	GrB_free(&a_zomb);
	GrB_free(&rel_zomb);

	*A = temp;

	return GrB_SUCCESS;
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
	ASSERT((lbls != NULL && n_lbls > 0) || (lbls == NULL && n_lbls == 0));
	ASSERT((rels != NULL && n_rels > 0) || (rels == NULL && n_rels == 0));
	if (A != NULL) ASSERT(strategy == BWM_MIN || strategy == BWM_MAX);

	// context for GrB operations
	compareContext ctx = {
		.g = g,                              // the input graph
		.w = weight,                         // the weight attribute to consider 
		.comp = (strategy == BWM_MAX)? -1: 1 // -1 if max, 1 if min
	};

	GrB_Scalar        theta         = NULL;  // Scalar containing the context
	GrB_UnaryOp       toMatrix      = NULL;  // get any ID from vector entry
	GrB_BinaryOp      weightOp      = NULL;  // get the weight from edgeID
	GrB_BinaryOp      minWeightID   = NULL;  // get min weight ID from vectors
	GrB_Type          contx_type    = NULL;  // GB equivalent of compareContext
	GrB_BinaryOp      reduceBiop    = NULL;  // gets two edge IDs and picks one
	GxB_IndexBinaryOp reduceIdxBiop = NULL;  // reduceBiop's underlying index op 
	size_t            n             = Graph_UncompactedNodeCount(g);
	bool              get_w         = (A_w != NULL);
	bool              get_id        = (A != NULL);

	GxB_index_binary_function toMatrixFuncs[3];
	GrB_BinaryOp doubleOps[3];
	double zombie_vals [3];
	
	toMatrixFuncs[BWM_MIN] = (GxB_index_binary_function) _reduceToMatrixMin;
	toMatrixFuncs[BWM_MAX] = (GxB_index_binary_function) _reduceToMatrixMax;
	toMatrixFuncs[BWM_SUM] = (GxB_index_binary_function) _reduceToMatrixSum;

	zombie_vals[BWM_MIN] = INFINITY;
	zombie_vals[BWM_MAX] = -INFINITY;
	zombie_vals[BWM_SUM] = 0.0;

	doubleOps[BWM_MIN] = GrB_MIN_FP64;
	doubleOps[BWM_MAX] = GrB_MAX_FP64;
	doubleOps[BWM_SUM] = GrB_PLUS_FP64;

	if(weight == ATTRIBUTE_ID_NONE) {
		ASSERT(get_id);
		// weight attribute wasn't specified, use a dummy weight with value: 0
		reduceBiop = GrB_FIRST_UINT64;
		GrB_UnaryOp_new (
			&toMatrix, (GxB_unary_function) _reduceToMatrixAny, GrB_UINT64, 
			GrB_UINT64
		) ;
	} else {
		GrB_OK(GrB_Type_new(&contx_type, sizeof(compareContext)));
		GrB_OK(GrB_Scalar_new(&theta, contx_type));
		GrB_OK(GrB_Scalar_setElement_UDT(theta, (void *) &ctx));

		if (get_id) {
			GrB_OK (GrB_BinaryOp_new(&weightOp, (GxB_binary_function) _getAttFromID, 
				GrB_FP64, GrB_UINT64, contx_type)) ;

			// reduce a matrix to its minimum (or maximum) valued edge
			GrB_OK (GrB_BinaryOp_new(
				&minWeightID, (GxB_binary_function) _reduceToMatrix, 
				GrB_UINT64, GrB_UINT64, contx_type)) ;

			// pick the minimum (or maximum) valued edge from the two matricies
			GrB_OK (GxB_IndexBinaryOp_new(
				&reduceIdxBiop, (GxB_index_binary_function) _pickBinary, 
				GrB_UINT64, GrB_UINT64, GrB_UINT64, contx_type, NULL, NULL));
		} else {
			ASSERT (get_w);
			GrB_OK (GxB_IndexBinaryOp_new (&reduceIdxBiop, toMatrixFuncs[strategy], 
				GrB_FP64, GrB_FP64, GrB_UINT64, contx_type, NULL, NULL)) ;
		}
		GrB_OK (GxB_BinaryOp_new_IndexOp(&reduceBiop, reduceIdxBiop, theta));
	}

	GrB_Matrix      _A     = NULL;  // output matrix containing EdgeIDs
	GrB_Vector      _N     = NULL;  // output filtered rows
	GrB_Index       nrows  = 0;     // number of rows in matrix
	GrB_Index       ncols  = 0;     // number of columns in matrix
	GrB_Type        A_type = NULL;  // type of the matrix
	GrB_Descriptor  desc   = NULL;

	GrB_Index rows_nvals;

	// if no relationships are specified, use all relationships
	// can't use adj matrix since I need access to the edgeIds of all edges
	if (rels == NULL) {
		n_rels = Graph_RelationTypeCount(g);
	}

	//--------------------------------------------------------------------------
	// compute L
	//--------------------------------------------------------------------------
	nrows = Graph_RequiredMatrixDim(g);

	// create vector N denoting all nodes participating in the algorithm
	GrB_OK(GrB_Vector_new(&_N, GrB_BOOL, nrows));

	// enforce labels
	_get_rows_with_labels(_N, g, lbls, n_lbls);

	GrB_OK(GrB_Vector_nvals(&rows_nvals, _N));

	if (n_rels == 0) {
		// graph does not have any relations, return empty matrix
		if (get_id){
			GrB_OK (GrB_Matrix_new(A, GrB_UINT64, rows_nvals, rows_nvals));
		}

		if (get_w) {
			GrB_OK (GrB_Matrix_new(A_w, GrB_FP64, rows_nvals, rows_nvals));
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

	Delta_Matrix *rel_ms = rm_calloc(n_rels, sizeof(Delta_Matrix)) ;

	bool multiEdgeFlag = false;
	for(int i = 0; i < n_rels; ++i) {
		RelationID id = GETRELATIONID(i);
		multiEdgeFlag = multiEdgeFlag
			|| Graph_RelationshipContainsMultiEdge(g, id);
		rel_ms[i] = Graph_GetRelationMatrix(g, id, false);
	}

	if (get_id) {
		_combine_matricies_weighted(&_A, rel_ms, n_rels, _N, reduceBiop);
	} else {
		_combine_attributes(&_A, rel_ms, n_rels, _N, reduceBiop, 
			zombie_vals[strategy]);
	}

	rm_free(rel_ms);


	// if _A has tensor entries, reduce it to a matrix
	if (get_id && multiEdgeFlag) {
		if (weight == ATTRIBUTE_ID_NONE) {
			GrB_OK(GrB_Matrix_apply(_A, NULL, NULL, toMatrix, _A, NULL));
		} else {
			GrB_OK(GrB_Matrix_apply_BinaryOp2nd_UDT(_A, NULL, NULL, minWeightID,
					_A, (void *) (&ctx), NULL));
		}
	}

	if (symmetric) {
		GrB_BinaryOp op = get_id ? reduceBiop : doubleOps[strategy];
		// make A symmetric A = A + At
		GrB_OK(GrB_Matrix_eWiseAdd_BinaryOp(_A, NULL, NULL, op, _A, _A,
				GrB_DESC_T1));
	}

	//--------------------------------------------------------------------------
	// compute weight matrix
	//--------------------------------------------------------------------------

	if (get_id && get_w) {
		GrB_OK(GrB_Matrix_new(A_w, GrB_FP64, rows_nvals, rows_nvals));

		if (weight == ATTRIBUTE_ID_NONE) {
			// if no weight specified, weights are zero
			GrB_OK(GrB_Matrix_assign_FP64(
				*A_w, _A, NULL, 0.0, GrB_ALL, n, GrB_ALL, n, GrB_DESC_S
			));
		} else {
			// get the weight value from the edge ids
			GrB_OK(GrB_Matrix_apply_BinaryOp2nd_UDT(
				*A_w, NULL, NULL, weightOp, _A, (void *) (&ctx), NULL
			));
		}
	} else if (get_w) {
		*A_w = _A;
		_A = NULL;
	}

	// set outputs
	if(_A != NULL) {
		*A = _A;
		_A = NULL;
	}

	if (rows != NULL) {
		*rows = _N;
		_N = NULL;
	}

	BWM_FREE;
	return GrB_SUCCESS;
}
