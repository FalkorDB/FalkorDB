#include "RG.h"
#include "delta_utils.h"
#include "delta_matrix.h"

GrB_Info Delta_cache_transpose
(
    Delta_Matrix A
) {
    GrB_Index nrows = 0;
    GrB_Index ncols = 0;
    GrB_Type  type = NULL;

    GrB_OK(Delta_Matrix_nrows(&nrows, A));
    GrB_OK(Delta_Matrix_ncols(&ncols, A));
    GrB_OK(Delta_Matrix_type(&type, A));

    ASSERT(A != NULL);
    // TODO: maybe this should error if the transpose already exists?
    if(A->transposed != NULL){
        Delta_Matrix_clear(A->transposed);
    } else {
        Delta_Matrix_new(&A->transposed, GrB_BOOL, nrows, ncols, false);
    }

    Delta_Matrix T = A->transposed;
    GrB_Matrix M   = DELTA_MATRIX_M(A);
    GrB_Matrix DP  = DELTA_MATRIX_DELTA_PLUS(A);
    GrB_Matrix DM  = DELTA_MATRIX_DELTA_MINUS(A);
    GrB_Matrix Mt  = DELTA_MATRIX_M(T);
    GrB_Matrix DPt = DELTA_MATRIX_DELTA_PLUS(T);
    GrB_Matrix DMt = DELTA_MATRIX_DELTA_MINUS(T);
    if(type == GrB_BOOL){
        GrB_OK(GrB_transpose(Mt, NULL, NULL, M, NULL));
        GrB_OK(GrB_transpose(DPt, NULL, NULL, DP, NULL));
    } else {
        // type is GrB_UINT64
        GrB_OK(GrB_Matrix_apply_BinaryOp2nd_UINT64(Mt, NULL, NULL, 
            GrB_NE_UINT64, M, U64_ZOMBIE, GrB_DESC_T0));
        GrB_OK(GrB_Matrix_apply_BinaryOp2nd_UINT64(DPt, NULL, NULL, 
            GrB_NE_UINT64, DP, U64_ZOMBIE, GrB_DESC_T0));
    }
    GrB_OK(GrB_transpose(DMt, NULL, NULL, DM, NULL));
    return GrB_SUCCESS;
}