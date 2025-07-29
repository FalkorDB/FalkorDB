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
	
	GrB_OK(GrB_Matrix_free(M));
	
	// LAGraph does not give good spread if random seeds are consecutive, so we
	// randomize the seed before using it.
	simple_rand(&seed);

	ASSERT(M != NULL);
	ASSERT(type != NULL);
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

	GrB_OK(GrB_Matrix_wait(*M, GrB_MATERIALIZE));
	Delta_Matrix_validate(mtx);

	*A = mtx;
	GrB_OK(GrB_Scalar_free(&empty));
}

// static void _push_element
// (
//     uint64_t *z,
//     uint64_t *x,
//     uint64_t *y
// ) {
//     ASSERT(SCALAR_ENTRY(*y));
//     if(SCALAR_ENTRY(*x)){
//         GrB_Vector v = NULL;
//         GrB_Vector_new(&v, GrB_UINT64, 1);
//         GrB_Vector_setElement_BOOL(v, true, *x);
//         GrB_Vector_setElement_BOOL(v, true, *y);
//         *z = SET_MSB((uint64_t) v);
//     } else {
//         ASSERT(z == x);
//         GrB_Vector v = AS_VECTOR(*z);
//         GrB_Vector_setElement_BOOL(v, true, *y);
//     }
// }

static void _mod_function
(
    uint64_t *z,
    const uint64_t *x,
    const uint64_t *y
)
{
    ASSERT(z != NULL && x != NULL && y != NULL);
    *(z) = (*x) % (*y);
}

// void Random_Tensor
// (
// 	Tensor *A,
// 	GrB_Type type,
// 	GrB_Index n,
// 	double density,
// 	double add_density,
// 	double del_density,
// 	uint64_t seed
// ) {
//     GrB_Vector i_v = NULL;
//     GrB_Vector j_v = NULL;
//     GrB_Vector x_v = NULL;

//     GrB_OK(LAGraph_Random_Seed(i_v, seed, NULL));
//     simple_rand(&seed);
//     GrB_OK(LAGraph_Random_Seed(j_v, seed, NULL));

//     GrB_Vector_apply_BinaryOp2nd_UINT64(i_v, NULL, NULL, 

//     GrB_Matrix_build_UINT64()
// }