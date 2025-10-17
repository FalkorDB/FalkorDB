/*
 * Copyright FalkorDB Ltd. 2023 - present
 * Licensed under the Server Side Public License v1 (SSPLv1).
 */

#include "RG.h"
#include "delta_matrix.h"

// applies on m and dp
GrB_Info Delta_Matrix_apply  // C = op(A)
(
    Delta_Matrix C,       // input/output matrix for results
    const GrB_UnaryOp op, // operator to apply to the entries
    const Delta_Matrix A  // first input:  matrix A
) {
    GrB_Matrix CM  = DELTA_MATRIX_M(C);
    GrB_Matrix CDP = DELTA_MATRIX_DELTA_PLUS(C);
    GrB_Matrix CDM = DELTA_MATRIX_DELTA_MINUS(C);
    GrB_Matrix AM  = DELTA_MATRIX_M(A);
    GrB_Matrix ADP = DELTA_MATRIX_DELTA_PLUS(A);
    GrB_Matrix ADM = DELTA_MATRIX_DELTA_MINUS(A);
    
    // do not apply on transpose since it is structural
    if(DELTA_MATRIX_MAINTAIN_TRANSPOSE(C)) {
        GrB_OK (Delta_Matrix_copy(C->transposed, A->transposed));
    }

    GrB_OK (GrB_Matrix_apply(CM, NULL, NULL, op, AM, NULL)) ; 

    GrB_OK (GrB_Matrix_apply(CDP, NULL, NULL, op, ADP, NULL)) ; 

    if(C != A) // copy the DM matricies if not alliased.
        GrB_OK (GrB_transpose(CDM, NULL, NULL, ADM, GrB_DESC_T0));
        
    return GrB_SUCCESS;
}
