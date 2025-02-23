//------------------------------------------------------------------------------
// GrB_Vector_extract: w<M> = accum (w, u(I))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "extract/GB_extract.h"
#include "mask/GB_get_mask.h"

GrB_Info GrB_Vector_extract         // w<mask> = accum (w, u(I))
(
    GrB_Vector w,                   // input/output vector for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Vector u,             // first input:  vector u
    const uint64_t *I,              // row indices (64-bit)
    uint64_t ni,                    // number of row indices
    const GrB_Descriptor desc       // descriptor for w and mask
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE3 (w, mask, u,
        "GrB_Vector_extract (w, M, accum, u, I, ni, desc)") ;
    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_NULL (u) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (w) ;
    GB_BURBLE_START ("GrB_extract") ;

    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (mask == NULL || GB_VECTOR_OK (mask)) ;
    ASSERT (GB_VECTOR_OK (u)) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) mask, &Mask_comp, &Mask_struct) ;

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
        I, false, ni,                   // row indices I and length ni (64-bit)
        GrB_ALL, false, 1,              // all columns (64-bit)
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

