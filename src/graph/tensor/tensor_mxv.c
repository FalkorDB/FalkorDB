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

// Returns the degree vector of a tensor with no multi-edges
GrB_Info Tensor_flat_degree
(
	GrB_Vector degree,  // [input / output] degree vector with values where 
						// the degree should be added
	GrB_Vector dest,    // [input] possible destination / source nodes
	Tensor T,           // matrix with tensor entries
    bool transpose      
) {
    GrB_Matrix M         = DELTA_MATRIX_M(T);
    GrB_Matrix Dp        = DELTA_MATRIX_DELTA_PLUS(T);
    GrB_Matrix Dm        = DELTA_MATRIX_DELTA_MINUS(T);
    GrB_Descriptor desc  = transpose? GrB_DESC_ST0: GrB_DESC_S;
    GrB_Info info;

    info = GrB_mxv(
        degree, degree, GrB_PLUS_UINT64, GxB_PLUS_PAIR_UINT64, M, dest, desc);
	if(info != GrB_SUCCESS) return info;

	info = GrB_mxv(
		degree, degree, GrB_PLUS_UINT64, GxB_PLUS_PAIR_UINT64, Dp, dest, desc);
	if(info != GrB_SUCCESS) return info;

	info = GrB_mxv(
		degree, degree, GrB_MINUS_UINT64, GxB_PLUS_PAIR_UINT64, Dm, dest, desc);
	return info;
}