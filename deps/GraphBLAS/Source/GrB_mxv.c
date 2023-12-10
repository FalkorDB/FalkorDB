//------------------------------------------------------------------------------
// GrB_mxv: matrix-vector multiply
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// w<M> = accum (w,t) where t = A*u or A'*u (u is never transposed)

// The input matrix A is optionally transposed, as determined by the
// Descriptor desc.

#include "GB_mxm.h"
#include "GB_get_mask.h"

GrB_Info GrB_mxv                    // w<M> = accum (w, A*u)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M_in,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Semiring semiring,    // defines '+' and '*' for matrix multiply
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Vector u,             // second input: vector u
    const GrB_Descriptor desc       // descriptor for w, M, A,
                                    // and method used for C=A*B
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (w, "GrB_mxv (w, M, accum, semiring, A, u, desc)") ;
    GB_BURBLE_START ("GrB_mxv") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;
    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (M_in == NULL || GB_VECTOR_OK (M_in)) ;
    ASSERT (GB_VECTOR_OK (u)) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx, AxB_method, do_sort) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // w<M> = accum (w,A*u) and variations, using the mxm kernel
    //--------------------------------------------------------------------------

    // w, M, and u are passed as matrices to GB_mxm.
    info = GB_mxm (
        (GrB_Matrix) w,     C_replace,      // w and its descriptor
        M, Mask_comp, Mask_struct,          // mask and its descriptor
        accum,                              // for accum (w,t)
        semiring,                           // definition of matrix multiply
        A,                  A_transpose,    // allow A to be transposed
        (GrB_Matrix) u,     false,          // u is never transposed
        false,                              // fmult(x,y), flipxy = false
        AxB_method, do_sort,                // algorithm selector
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

