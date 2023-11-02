//------------------------------------------------------------------------------
// GxB_Vector_eWiseUnion: vector element-wise operations, set union
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// w<M> = accum (w,u+v)

// if u(i) and v(i) both appear:
//      C(i,j) = add (u(i), v(i))
// else if u(i) appears but v(i) does not:
//      C(i,j) = add (u(i), beta)
// else if u(i) does not appear but v(i) does:
//      C(i,j) = add (alpha, v(i))

#include "GB_ewise.h"
#include "GB_get_mask.h"

#define GB_EWISE(op)                                                        \
    /* check inputs */                                                      \
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;                                       \
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;                                       \
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;                                       \
    GB_RETURN_IF_FAULTY (M_in) ;                                            \
    ASSERT (GB_VECTOR_OK (w)) ;                                             \
    ASSERT (GB_VECTOR_OK (u)) ;                                             \
    ASSERT (GB_VECTOR_OK (v)) ;                                             \
    ASSERT (M_in == NULL || GB_VECTOR_OK (M_in)) ;                          \
    /* get the descriptor */                                                \
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,       \
        xx1, xx2, xx3, xx7) ;                                               \
    /* get the mask */                                                      \
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) M_in, &Mask_comp, &Mask_struct) ; \
    /* w<M> = accum (w,t) where t = u+v, u'+v, u+v', or u'+v' */            \
    info = GB_ewise (                                                       \
        (GrB_Matrix) w, C_replace,  /* w and its descriptor        */       \
        M, Mask_comp, Mask_struct,  /* mask and its descriptor */           \
        accum,                      /* accumulate operator         */       \
        op,                         /* operator that defines '+'   */       \
        (GrB_Matrix) u, false,      /* u, never transposed         */       \
        (GrB_Matrix) v, false,      /* v, never transposed         */       \
        true,                       /* eWiseAdd                    */       \
        true, alpha, beta,          /* eWiseUnion                  */       \
        Werk)

//------------------------------------------------------------------------------
// GxB_Vector_eWiseUnion: vector addition
//------------------------------------------------------------------------------

GrB_Info GxB_Vector_eWiseUnion      // w<M> = accum (w, u+v)
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M_in,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_BinaryOp add,         // defines '+' for t=u+v
    const GrB_Vector u,             // first input:  vector u
    const GrB_Scalar alpha,
    const GrB_Vector v,             // second input: vector v
    const GrB_Scalar beta,
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (w, "GxB_Vector_eWiseUnion (w, M, accum, add, u, alpha,"
        " v, beta, desc)") ;
    GB_BURBLE_START ("GxB_eWiseUnion") ;
    GB_RETURN_IF_NULL_OR_FAULTY (add) ;

    //--------------------------------------------------------------------------
    // apply the eWise kernel (using set union)
    //--------------------------------------------------------------------------

    GB_EWISE (add) ;
    GB_BURBLE_END ;
    return (info) ;
}

