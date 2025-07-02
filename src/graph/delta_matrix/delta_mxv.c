#include "RG.h"
#include "delta_matrix.h"

// Computes c = A * u. 
// Does not look at dm. Assumes that any "zombie" value is a semiring zero.
GrB_Info Delta_mxv
(
    GrB_Vector c,                 // [output] vector
    const GrB_Vector mask,        // [input] mask
    const GrB_BinaryOp accum,     // [input] accum 
    const GrB_Semiring semiring,  // [input] must treat zombie values as 
                                  //    semiring zero
    const Delta_Matrix A,         // [input] Delta_Matrix
    const GrB_Vector u,           // [input] GrB_BOOL vector 
    const GrB_Descriptor desc     // [input] descriptor 
) {
    GrB_BinaryOp monOP = NULL;
    GrB_Semiring_get_VOID(semiring, &monOP, GxB_MONOID_OPERATOR);

    ASSERT(c != u); //TODO: allocate _c (temp vector) to handle
    ASSERT(accum == NULL || accum == monOP); //TODO: allocate _c (temp vector) to handle

    GrB_Info info = GrB_mxv(
        c, mask, accum, semiring, DELTA_MATRIX_M(A), u, desc) ; 
    if(info != GrB_SUCCESS) return info;

    info = GrB_mxv(
        c, mask, monOP, semiring, DELTA_MATRIX_DELTA_PLUS(A), u, desc) ;
    return info;
}

// Computes c = A * u. 
// Assumes that the addition operation is plus. This same strategy could work 
// for any invertable monoid.
// TODO: make a better name for this function. 
GrB_Info Delta_mxv_count
(
    GrB_Vector c,                 // [output] vector
    const GrB_Vector mask,        // [input] mask
    const GrB_BinaryOp accum,     // [input] accum 
    const GrB_Semiring semiring,  // [input] must treat zombie values as 
                                  //    semiring zero
    const Delta_Matrix A,         // [input] Delta_Matrix
    const GrB_Vector u,           // [input] GrB_BOOL vector 
    const GrB_Descriptor desc     // [input] descriptor 
) {
    GrB_BinaryOp monOP = NULL;
    GrB_Semiring_get_VOID(semiring, &monOP, GxB_MONOID_OPERATOR);
    
    ASSERT(c != u); //TODO: allocate _c (temp vector) to handle
    ASSERT(accum == NULL || accum == monOP); //TODO: allocate _c to handle
    ASSERT(monOP == GrB_PLUS_UINT64)

    GrB_Matrix m  = DELTA_MATRIX_M(A);
    GrB_Matrix dp = DELTA_MATRIX_DELTA_PLUS(A);
    GrB_Matrix dm = DELTA_MATRIX_DELTA_MINUS(A);

    GrB_Info info = GrB_mxv(
        c, mask, accum, semiring, m, u, desc) ; 
    if(info != GrB_SUCCESS) return info;

    info = GrB_mxv(
        c, mask, GrB_PLUS_UINT64, semiring, dp, u, desc) ;
    if(info != GrB_SUCCESS) return info;

    info = GrB_mxv(
        c, mask, GrB_MINUS_UINT64, semiring, dm, u, desc) ;
    return info;
}