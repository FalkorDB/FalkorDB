//------------------------------------------------------------------------------
// GrB_Vector_extract: w<M> = accum (w, u(I))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_extract.h"
#include "GB_get_mask.h"

GrB_Info GrB_Vector_extract         // w<M> = accum (w, u(I))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector M_in,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Vector u,             // first input:  vector u
    const GrB_Index *I,             // row indices
    GrB_Index ni,                   // number of row indices
    const GrB_Descriptor desc       // descriptor for w and M
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (w, "GrB_Vector_extract (w, M, accum, u, I, ni, desc)") ;
    GB_BURBLE_START ("GrB_extract") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;
    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (M_in == NULL || GB_VECTOR_OK (M_in)) ;
    ASSERT (GB_VECTOR_OK (u)) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // extract entries
    //--------------------------------------------------------------------------

    // If a column list J is constructed containing the single column index 0,
    // then T = A(I,0) followed by C<M>=accum(C,T) does the right thing
    // where all matrices (C, M, and T) are columns of size ni-by-1.  Thus,
    // GB_extract does the right thing for this case.  Note that the input u is
    // not transposed.  All GrB_Matrix objects will be in CSC format, and no
    // matrices are transposed via the C_is_vector option in GB_extract.

    //--------------------------------------------------------------------------
    // do the work in GB_extract
    //--------------------------------------------------------------------------

    info = GB_extract (
        (GrB_Matrix) w,     C_replace,  // w as a matrix, and its descriptor
        M, Mask_comp, Mask_struct,      // mask and its descriptor
        accum,                          // optional accum for z=accum(w,t)
        (GrB_Matrix) u,     false,      // u as matrix; never transposed
        I, ni,                          // row indices I and length ni
        GrB_ALL, 1,                     // all columns
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

