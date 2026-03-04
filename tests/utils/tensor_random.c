#include "GraphBLAS.h"
#include "LAGraph.h"
#include "LAGraphX.h"
#include "util/simple_rand.h"
#include "src/graph/tensor/tensor.h"
#include "src/graph/delta_matrix/delta_utils.h"

void Delta_Random_Matrix
(
	Delta_Matrix *A,     // delta matrix to be initialized and output
	GrB_Type type,       // type of the matrix
	GrB_Index n,         // dimension of the matrix (nxn)
	double density,      // estimated density of entries in M
	double add_density,  // estimated density of entries in DP
	double del_density,  // estimated density of entries in DM
	uint64_t seed        // seed to be used for generating the matrix
) {
	// Random Matrix is created with the following steps:
	//   1) Make M (A random matrix with random values of type `type`)
	//   2) Make DP (Same but the density of add_density)
	//   3) Make temp_dm (Same but del_density)
	//   4) Add the entries of temp_dm into M
	//   5) Make DM = temp_dm but boolean isovalued
	//   6) DP -= M
	//   7) Pack M, DP, DM into a delta matrix and return

	ASSERT(A != NULL);
	ASSERT(type == GrB_BOOL || type == GrB_UINT64);

	// Allocate the matrix with no transpose
	Delta_Matrix mtx = NULL;
	GrB_Matrix M       = NULL;
	GrB_Matrix DP      = NULL;
	GrB_Matrix temp_dm = NULL;
	GrB_Matrix DM      = NULL;

	GrB_OK(Delta_Matrix_new(&mtx, type, n, n, false));

	// LAGraph does not give good spread if random seeds are consecutive, so we
	// randomize the seed before using it.
	simple_rand(&seed);

	GrB_OK(LAGraph_Random_Matrix(&M, type, n, n, density, seed, NULL));
	simple_rand(&seed);

	if(add_density > 0) {
		GrB_OK(LAGraph_Random_Matrix(&DP, type, n, n, add_density, seed, NULL));
	} else {
		GrB_OK (GrB_Matrix_new(&DP, type, n, n));
	}
	simple_rand(&seed);

	if(del_density > 0) {
		GrB_OK(LAGraph_Random_Matrix(
			&temp_dm, type, n, n, del_density, seed, NULL));

		// M += DM
		GrB_OK (GrB_Matrix_assign(
			M, temp_dm, NULL, temp_dm, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));

		// Make boolean if not already
		if (type != GrB_BOOL) {
			GrB_OK (GrB_Matrix_new(&DM, GrB_BOOL, n, n));
		}

		// Assign DM and isovalue of true
		DM = DM != NULL ? DM : temp_dm;
		GrB_OK(GrB_Matrix_assign_BOOL(
			DM, temp_dm, NULL, true, GrB_ALL, n, GrB_ALL, n, GrB_DESC_S));

		if (temp_dm != DM) {
			GrB_free(&temp_dm);
		}

		temp_dm = NULL;
	} else {
		GrB_OK (GrB_Matrix_new(&DM, GrB_BOOL, n, n));
	}

	// Make the rest of the matricies iso true if GrB_BOOL
	if(type == GrB_BOOL){
		GrB_OK(GrB_Matrix_assign_BOOL(
			M, M, NULL, true, GrB_ALL, n, GrB_ALL, n, GrB_DESC_S));
		GrB_OK(GrB_Matrix_assign_BOOL(
			DP, DP, NULL, true, GrB_ALL, n, GrB_ALL, n, GrB_DESC_S));
	}
	// DP = DP - M
	GrB_OK(GrB_transpose(DP, M, NULL, DP, GrB_DESC_RSCT0));

	//--------------------------------------------------------------------------
	// Wait before returning
	//--------------------------------------------------------------------------
	GrB_wait(M,  GrB_MATERIALIZE);
	GrB_wait(DP, GrB_MATERIALIZE);
	GrB_wait(DM, GrB_MATERIALIZE);

	Delta_Matrix_setMatrices(mtx, &M, &DP, &DM);
	Delta_Matrix_validate(mtx, false);

	*A = mtx;
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

static void _select_random
(
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

// Create a single 3D GrB_Matrix
void _make_single_tensor
(
	GrB_Matrix A,                    // GrB_Matrix to add tensor values into
	GrB_Index n,                     // dimension of the matrix (nxn)
	GrB_Index e,                     // number of edges
	const GrB_BinaryOp dup_handler,  // function that creates tensors
	const GrB_BinaryOp mod_op,       // function for modulo operation
	const GrB_Descriptor desc,
	uint64_t seed
) {
	// 1) make two random integer vectors i_v, j_v
	// 2) i_v %= n and j_v %= n
	// 3) buil a matrix
	//     values: [0, 1, ..., e]
	//     rows: i_v
	//     cols: j_v
	// 4) handle duplicate values by creating a vector at that entry
	//
	// Example: n = 3,
	// i_v = [0, 0, 1, 2, 0, 1, 2, 2, 2],
	// j_v = [0, 0, 0, 1, 1, 1, 2, 1, 0]
	// Output:
	// |x 4 .|  Vector x: [0, 1]
	// |2 5 .|
	// |8 y 6|  Vector y: [3, 7]

	ASSERT(A != NULL);
	ASSERT(n > 0);

	if (e == 0) {
		return;
	}

	GrB_Vector     i_v         = NULL;
	GrB_Vector     j_v         = NULL;
	GrB_OK(GrB_Vector_new(&i_v, GrB_UINT64, e));
	GrB_OK(GrB_Vector_new(&j_v, GrB_UINT64, e));

	// make i_v an j_v full vectors
	GrB_OK(GrB_Vector_assign_UINT64(i_v, NULL, NULL, 0, GrB_ALL, e, NULL));
	GrB_OK(GrB_Vector_assign_UINT64(j_v, NULL, NULL, 0, GrB_ALL, e, NULL));

	// assign random values into all entries in the vectors
	// makes two full vectors of random values
	GrB_OK((GrB_Info) LAGraph_Random_Seed(i_v, seed, NULL));
	GrB_OK((GrB_Info) LAGraph_Random_Seed(j_v, seed + e, NULL));

	// take the values mod n so that the indexes are within the matrix
	GrB_OK(GrB_Vector_apply_BinaryOp2nd_UINT64(i_v, NULL, NULL, mod_op, i_v, n, 
		NULL));
	GrB_OK(GrB_Vector_apply_BinaryOp2nd_UINT64(j_v, NULL, NULL, mod_op, j_v, n, 
		NULL));
	
	// Want to build a tensor
	// i_v: the random row indexes
	// j_v: the random column index
	// i_v (interpretend by index): the values [0, 1, ..., e]
	// values landing in the same slot are put into a vector
	GrB_OK(GxB_Matrix_build_Vector(A, i_v, j_v, i_v, dup_handler, desc));

	GrB_OK (GrB_Vector_free(&i_v));
	GrB_OK (GrB_Vector_free(&j_v));
}

// free vector entries of a tensor
static void _free_vectors
(
	uint64_t *z,       // [ignored] new value
	const uint64_t *x  // current entry
) {
	// see if entry is a vector
	if(!SCALAR_ENTRY(*x)) {
		// free vector
		GrB_Vector V = AS_VECTOR(*x);
		GrB_free(&V);
	}
	*z = MSB_MASK;
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
		GrB_OK (GrB_Vector_new(&v, GrB_BOOL, GrB_INDEX_MAX));
		GrB_OK (GrB_Vector_setElement_BOOL(v, true, *x));
		GrB_OK (GrB_Vector_setElement_BOOL(v, true, *y));
		*z = SET_MSB((uint64_t) v);
	} else {
		GrB_Vector v = AS_VECTOR(*x);
		GrB_OK (GrB_Vector_setElement_BOOL(v, true, *y));
		*z = SET_MSB((uint64_t) v);
	}
}

// Make a random tensor
void Random_Tensor
(
	Tensor *A,           // Tensor to allocate and add rendom entries to
	GrB_Index n,         // dimension of Tensor (nxn)
	double density,      // estimated density of edges in M
	double add_density,  // estimated density of edges in DP
	double del_density,  // estimated density of edges in DM
	uint64_t seed        // random seed
) {
	// create a Tensor in these sets
	//   1) Create M with multiedge entries
	//   2) Create DP with multiedge entries
	//   3) temp = random sample of M with a density of del_density / density
	//   4) free the entries in temp
	//   5) DM <temp> = true
	//   6) DP -= M (make sure to free the entries being deleted from DP)
	ASSERT(A != NULL);
	ASSERT(del_density < density);

	// z = x % y
	GrB_BinaryOp     mod_op      = NULL;

	// Output: Vector z containing all of the entries of x plus the entry y
	// x: Vector or Scalar
	// y: Scalar to add to x
	GrB_BinaryOp     dup_handler = NULL;

	// z = MSB_MASK
	// GrB_free (x) if x is vector
	GrB_UnaryOp      free_entry  = NULL;

	// z = true if (rand() < y)
	// where rand() is in [0,1]
	GrB_IndexUnaryOp select_random_sample = NULL;

	// Descriptor. GxB_VALUE_LIST = GxB_USE_INDICES
	GrB_Descriptor   desc        = NULL;

	//--------------------------------------------------------------------------
	// initialize ops
	//--------------------------------------------------------------------------

	GrB_OK(GrB_BinaryOp_new(&mod_op, (GxB_binary_function) _mod_function,
		GrB_UINT64, GrB_UINT64, GrB_UINT64));
	GrB_OK(GrB_BinaryOp_new(&dup_handler, (GxB_binary_function) _push_element,
		GrB_UINT64, GrB_UINT64, GrB_UINT64));
	GrB_OK(GrB_UnaryOp_new(&free_entry, (GxB_unary_function) _free_vectors,
		GrB_UINT64, GrB_UINT64));
	GrB_OK(GrB_IndexUnaryOp_new(
		&select_random_sample, (GxB_index_unary_function) _select_random,
		GrB_BOOL, GrB_UINT64, GrB_FP64));

	GrB_OK(GrB_Descriptor_new(&desc));
	GrB_OK(GrB_Descriptor_set_INT32(desc, GxB_USE_INDICES, GxB_VALUE_LIST));

	Delta_Matrix_new(A, GrB_UINT64, n, n, false);
	GrB_Matrix M    = DELTA_MATRIX_M(*A);
	GrB_Matrix DP   = DELTA_MATRIX_DELTA_PLUS(*A);
	GrB_Matrix DM   = DELTA_MATRIX_DELTA_MINUS(*A);
	GrB_Matrix temp = NULL;
	GrB_Index nvals = 0;
	GrB_Index M_e   = n * n * density;
	GrB_Index DP_e  = n * n * add_density;

	//--------------------------------------------------------------------------
	// Create M and DP
	//--------------------------------------------------------------------------
	_make_single_tensor(M, n, M_e, dup_handler, mod_op,
		desc, seed);
	simple_rand(&seed);


	_make_single_tensor(DP, n, DP_e, dup_handler,
		mod_op, desc, seed);
	simple_rand(&seed);

	//--------------------------------------------------------------------------
	// Assure that at least one vector entry has been made
	//--------------------------------------------------------------------------
	// TODO: this is rather simple but insures there is at least one tensor
	// (at 0,0) in M. Allow caller to control number of multiedge entries
	// in the future
	GrB_Index zero = 0;
	GrB_OK (GrB_Matrix_assign_UINT64(
		M, NULL, dup_handler, M_e, &zero, 1, &zero, 1, NULL));
	GrB_OK (GrB_Matrix_assign_UINT64(
		M, NULL, dup_handler, M_e + 1, &zero, 1, &zero, 1, NULL));

	//--------------------------------------------------------------------------
	// Make M and DP disjoint
	//--------------------------------------------------------------------------
	GrB_OK (GrB_Matrix_new(&temp, GrB_UINT64, n, n));
	GrB_OK (GrB_Matrix_assign(
		temp, M, NULL, DP, GrB_ALL, 0, GrB_ALL, 0, GrB_DESC_S));

	// free the overlaping entries
	GrB_OK (GrB_Matrix_apply(temp, NULL, NULL, free_entry, temp, GrB_DESC_S));

	// eliminate them from DP
	GrB_OK (GrB_transpose(DP, temp, NULL, DP, GrB_DESC_RSCT0));
	GrB_OK (GrB_Matrix_clear(temp));

	//--------------------------------------------------------------------------
	// Create DM
	//--------------------------------------------------------------------------
	// select entries to delete from M
	GrB_OK (GrB_Matrix_select_FP64(temp, NULL, NULL, select_random_sample, M,
		del_density / density, NULL));

	// free those entries
	GrB_OK (GrB_Matrix_apply(temp, NULL, NULL, free_entry, temp, GrB_DESC_S));

	// assign true in DM
	GrB_OK (GrB_Matrix_assign_BOOL(DM, temp, NULL, true, GrB_ALL, n,
		GrB_ALL, n, GrB_DESC_S));

	//--------------------------------------------------------------------------
	// Wait before returning
	//--------------------------------------------------------------------------
	GrB_wait(M,  GrB_MATERIALIZE);
	GrB_wait(DP, GrB_MATERIALIZE);
	GrB_wait(DM, GrB_MATERIALIZE);


	//--------------------------------------------------------------------------
	// frees
	//--------------------------------------------------------------------------
	GrB_OK (GrB_free(&temp));
	GrB_OK (GrB_free(&mod_op));
	GrB_OK (GrB_free(&free_entry));
	GrB_OK (GrB_free(&dup_handler));
	GrB_OK (GrB_free(&select_random_sample));
	GrB_OK (GrB_free(&desc));
}
