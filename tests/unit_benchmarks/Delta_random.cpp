#include "create_random.h"
void Delta_Random_Matrix
(
	Delta_Matrix *A,
	GrB_Type type,
	GrB_Index n,
	double density,
	double add_density,
	double del_density,
	uint64_t seed
) {
	ASSERT(A != NULL);
	ASSERT(type == GrB_BOOL || type == GrB_UINT64);
	Delta_Matrix mtx = NULL;
	GrB_OK(Delta_Matrix_new(&mtx, type, n, n, false));
	GrB_Matrix *M    = &DELTA_MATRIX_M(mtx);
	GrB_Matrix *DP   = &DELTA_MATRIX_DELTA_PLUS(mtx);
	GrB_Matrix *DM   = &DELTA_MATRIX_DELTA_MINUS(mtx);
	GrB_Scalar empty = NULL;

    GrB_OK(GrB_Scalar_new(&empty, GrB_BOOL));
	
	
	// LAGraph does not give good spread if random seeds are consecutive, so we
	// randomize the seed before using it.
	simple_rand(&seed);

	ASSERT(type != NULL);
	GrB_OK(GrB_Matrix_free(M));
	GrB_OK(LAGraph_Random_Matrix(M, type, n, n, density, seed, NULL));
	simple_rand(&seed);

	if(add_density > 0) {
		GrB_OK(GrB_Matrix_free(DP));
		GrB_OK(LAGraph_Random_Matrix(DP, type, n, n, add_density, seed, NULL));
		simple_rand(&seed);
		GrB_OK(GrB_Matrix_set_INT32(*DP, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL));
	}

	if(del_density > 0) {
		GrB_OK(GrB_Matrix_free(DM));
		GrB_OK(LAGraph_Random_Matrix(DM, GrB_BOOL, n, n, del_density, seed, NULL));
		GrB_OK(GrB_Matrix_set_INT32(*DM, GxB_HYPERSPARSE, GxB_SPARSITY_CONTROL));
	}

	GrB_OK(GrB_Matrix_assign_BOOL(*DM, *DM, NULL, true, GrB_ALL, n, GrB_ALL, 
		n, GrB_DESC_S));

	if(type == GrB_BOOL){
		// Set all entries to true
		GrB_OK(GrB_Matrix_assign_BOOL(*M, *M, NULL, true, GrB_ALL, n, GrB_ALL, 
			n, GrB_DESC_S));
		GrB_OK(GrB_Matrix_assign_BOOL(*DP, *DP, NULL, true, GrB_ALL, n, GrB_ALL, 
			n, GrB_DESC_S));

		// M <DM> = 0 
		GrB_OK(GrB_Matrix_assign_BOOL(*M, *DM, NULL, false, GrB_ALL, n, GrB_ALL, 
			n, GrB_DESC_S));
	} else {
		GrB_OK(GrB_Matrix_assign_UINT64(*M, *DM, NULL, U64_ZOMBIE, GrB_ALL, n, 
			GrB_ALL, n, GrB_DESC_S));
	}

	// DP = DP - M
	GrB_OK(GrB_Matrix_assign_Scalar(*DP, *M, NULL, empty, GrB_ALL, n, GrB_ALL, 
		n, GrB_DESC_S));
	GrB_OK(GrB_Matrix_apply(*DP, NULL, NULL, GxB_ONE_BOOL,*DP, NULL));

	Delta_Matrix_wait(mtx, false);
	Delta_Matrix_validate(mtx, true);

	*A = mtx;
	GrB_OK(GrB_Scalar_free(&empty));
}

static void _push_element
(
    uint64_t *z,
    const uint64_t *x,
    const uint64_t *y
) {
    ASSERT(SCALAR_ENTRY(*y));
    if(SCALAR_ENTRY(*x)){
        GrB_Vector v = NULL;
		GrB_Vector_new(&v, GrB_BOOL, GrB_INDEX_MAX);
        GrB_Vector_setElement_BOOL(v, true, *x);
        GrB_Vector_setElement_BOOL(v, true, *y);
        *z = SET_MSB((uint64_t) v);
    } else {
        ASSERT(*z == *x);
        GrB_Vector v = AS_VECTOR(*z);
        GrB_Vector_setElement_BOOL(v, true, *y);
    }
}

static void _mod_function
(
    uint64_t *z,
    const uint64_t *x,
    const uint64_t *y
)
{
    *(z) = (*x) % (*y);
}

static void _select_random(
	bool *z,
	uint64_t *x,
	GrB_Index i,
	GrB_Index j,
	double *t // probability of selecting an entry
) {
	uint64_t rand = i + 324897982391ull;
	simple_rand(&rand);
	rand ^= j;
	simple_rand(&rand);
	*z    = rand < (*t) * UINT64_MAX;
}

void _make_single_tensor
(
	GrB_Matrix A,
	GrB_Index n,
	GrB_Index e,
	const GrB_BinaryOp dup_handler,
	const GrB_BinaryOp mod_op,
	const GrB_Descriptor desc,
	uint64_t seed
) {
	ASSERT(A != NULL);
	GrB_Vector     i_v         = NULL;
	GrB_Vector     j_v         = NULL;

	GrB_OK(GrB_Vector_new(&i_v, GrB_UINT64, e));
	GrB_OK(GrB_Vector_new(&j_v, GrB_UINT64, e));
	GrB_OK(GrB_Vector_assign_UINT64(i_v, NULL, NULL, 0, GrB_ALL, e, NULL));
	GrB_OK(GrB_Vector_assign_UINT64(j_v, NULL, NULL, 0, GrB_ALL, e, NULL));

    GrB_OK((GrB_Info) LAGraph_Random_Seed(i_v, seed, NULL));
    GrB_OK((GrB_Info) LAGraph_Random_Seed(j_v, seed + e, NULL));

    GrB_OK(GrB_Vector_apply_BinaryOp2nd_UINT64(i_v, NULL, NULL, mod_op, i_v, n, 
		NULL));

    GrB_OK(GrB_Vector_apply_BinaryOp2nd_UINT64(j_v, NULL, NULL, mod_op, j_v, n, 
		NULL));
	
	// Want to build a tensor
	GrB_OK(GxB_Matrix_build_Vector(A, i_v, j_v, i_v, dup_handler, desc));

	GrB_Vector_free(&i_v);
	GrB_Vector_free(&j_v);
}

// Make a random tensor
void Random_Tensor
(
	Tensor *A,
	GrB_Index n,
	double density,
	double add_density,
	double del_density,
	uint64_t seed
) {
	ASSERT(A != NULL);

	GrB_BinaryOp     mod_op      = NULL;
	GrB_BinaryOp     dup_handler = NULL;
	GrB_UnaryOp      free_entry  = Global_GrB_Ops_Get()->free_tensors;
	GrB_IndexUnaryOp select_op   = NULL;
	GrB_Descriptor   desc        = NULL;

	GrB_OK(GrB_BinaryOp_new(&mod_op, (GxB_binary_function) _mod_function, 
		GrB_UINT64, GrB_UINT64, GrB_UINT64));
	GrB_OK(GrB_BinaryOp_new(&dup_handler, (GxB_binary_function) _push_element, 
		GrB_UINT64, GrB_UINT64, GrB_UINT64));
	GrB_OK(GrB_IndexUnaryOp_new(
		&select_op, (GxB_index_unary_function) _select_random, 
		GrB_BOOL, GrB_UINT64, GrB_FP64));

	GrB_OK(GrB_Descriptor_new(&desc));
	GrB_OK(GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_VALUE_LIST));

	Delta_Matrix_new(A, GrB_UINT64, n, n, false);
	GrB_Matrix M    = DELTA_MATRIX_M(*A);
	GrB_Matrix DP   = DELTA_MATRIX_DELTA_PLUS(*A);
	GrB_Matrix DM   = DELTA_MATRIX_DELTA_MINUS(*A);
	GrB_Matrix temp = NULL;
	GrB_Index nvals = 0;

	//--------------------------------------------------------------------------
	// Create M and DP
	//--------------------------------------------------------------------------
	_make_single_tensor(M, n, (uint64_t) (n * n * density), dup_handler, mod_op, 
		desc, seed);
	GrB_Matrix_nvals(&nvals, M);
	ASSERT(nvals > 0);

	simple_rand(&seed);

	_make_single_tensor(DP, n, (uint64_t) (n * n * add_density), dup_handler, 
		mod_op, desc, seed);
	GrB_Matrix_nvals(&nvals, DP);
	ASSERT(nvals > 0);

	//--------------------------------------------------------------------------
	// Make M and DP disjoint
	//--------------------------------------------------------------------------
	GrB_OK (GrB_Matrix_apply(DP, M, NULL, free_entry, DP, GrB_DESC_S)); 
	GrB_OK (GrB_Matrix_select_UINT64(DP, NULL, NULL, GrB_VALUENE_UINT64, DP, 
		U64_ZOMBIE, NULL));


	//--------------------------------------------------------------------------
	// Create DM
	//--------------------------------------------------------------------------
	GrB_OK (GrB_Matrix_new(&temp, GrB_BOOL, n, n));
	GrB_OK (GrB_Matrix_select_FP64(temp, NULL, NULL, select_op, M, 
		del_density / density, NULL));
	GrB_OK (GrB_Matrix_apply(temp, NULL, NULL, free_entry, temp, GrB_DESC_S));
	GrB_OK (GrB_Matrix_assign_BOOL(DM, temp, NULL, true, GrB_ALL, n, 
		GrB_ALL, n, GrB_DESC_S));
	GrB_OK (GrB_Matrix_free(&temp));
	
	GrB_OK (GrB_Matrix_assign_UINT64(M, DM, NULL, U64_ZOMBIE, GrB_ALL, n, 
		GrB_ALL, n, GrB_DESC_S));

	GrB_OK (GrB_Matrix_free(&temp));
	GrB_OK (GrB_BinaryOp_free(&dup_handler));
	GrB_OK (GrB_BinaryOp_free(&mod_op));
	GrB_OK (GrB_IndexUnaryOp_free(&select_op));
	GrB_OK (GrB_Descriptor_free(&desc));
}
