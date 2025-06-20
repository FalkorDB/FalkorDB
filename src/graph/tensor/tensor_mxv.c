#include "RG.h"
#include "tensor.h"
#include "util/arr.h"

GrB_Info Tensor_mxv
(
    GrB_Vector c,                 // output vector
    const GrB_Vector mask,        // [input] mask
    const GrB_BinaryOp accum,     // [input] accum 
    const GrB_Semiring semiring,  // [input] semiring see TENSORPICK in  
                                  //   tensor_util for multiplication definition
    const Tensor A,               // [input] Tensor
    const GrB_Vector u,           // [input] GrB_BOOL vector 
    const GrB_Descriptor desc     // [input] descriptor 
) {
    GrB_BinaryOp monOP = NULL;
    GrB_Semiring_get_VOID(semiring, &monOP, GxB_MONOID_OPERATOR);
    if(accum && accum != monOP) //TODO
        return GrB_NOT_IMPLEMENTED;
    GrB_Info info = GrB_mxv(
        c, mask, accum, semiring, Delta_Matrix_M(A), u, desc) ; 
    if(info != GrB_SUCCESS) return info;
    info = GrB_mxv(
        c, mask, accum, semiring, DELTA_MATRIX_DELTA_PLUS(A), u, desc) ;
    return info;
}
