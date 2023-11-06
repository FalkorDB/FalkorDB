//------------------------------------------------------------------------------
// GrB_vxm: vector-matrix multiply
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// w'<M'> = accum (w',t) where t = u'*A or u'*A'

// Rows w', u', and M' are simply columns w, u, and M.  Thus:
// w<M> = accum (w,t) where t = A'*u or A*u, but with the multiply operator
// flipped.  The input descriptor for A, inp1, is also negated.

#include "GB_mxm.h"
#include "GB_get_mask.h"

GrB_Info GrB_vxm                    // w'<M> = accum (w', u'*A)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M_in,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Semiring semiring,    // defines '+' and '*' for matrix multiply
    const GrB_Vector u,             // first input:  vector u
    const GrB_Matrix A,             // second input: matrix A
    const GrB_Descriptor desc       // descriptor for w, M, and A
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (w, "GrB_vxm (w, M, accum, semiring, u, A, desc)") ;
    GB_BURBLE_START ("GrB_vxm") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (M_in == NULL || GB_VECTOR_OK (M_in)) ;
    ASSERT (GB_VECTOR_OK (u)) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx, A_transpose, AxB_method, do_sort) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // w'<M'> = accum (w',u'*A) and variations, using the mxm kernel
    //--------------------------------------------------------------------------

    // w, M, and u are treated as column vectors and passed as n-by-1 matrices
    // to GB_mxm A and u are swapped, and A_transpose is negated:
    //      u'*A  == A'*u
    //      u'*A' == A*u
    // Since A and u are swapped, in all the matrix multiply kernels,
    // the multiplier must be flipped, so flipxy is passed in as true.

    info = GB_mxm (
        (GrB_Matrix) w,     C_replace,      // w and its descriptor
        M, Mask_comp, Mask_struct,          // mask and its descriptor
        accum,                              // for accum (w,t)
        semiring,                           // definition of matrix multiply
        A,                  !A_transpose,   // allow A to be transposed
        (GrB_Matrix) u,     false,          // u is never transposed
        true,                               // fmult(y,x), flipxy = true
        AxB_method, do_sort,                // algorithm selector
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

