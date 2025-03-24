//------------------------------------------------------------------------------
// GxB_Col_extract_Vector: w<M> = accum (w, A(I,j)) or A(j,I)'
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Extract a single row or column from a matrix.  Note that in the
// GraphBLAS spec, row and column vectors are indistinguishable.  In this
// implementation, both are the same as an n-by-1 GrB_Matrix, except with
// restrictions on the matrix operations that can be performed on them.

#include "extract/GB_extract.h"
#include "mask/GB_get_mask.h"
#include "ij/GB_ij.h"
#define GB_FREE_ALL                             \
    if (I_size > 0) GB_FREE_MEMORY (&I, I_size) ;

GrB_Info GxB_Col_extract_Vector     // w<mask> = accum (w, A(I,j))
(
    GrB_Vector w,                   // input/output matrix for results
    const GrB_Vector mask,          // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(w,t)
    const GrB_Matrix A,             // first input:  matrix A
    const GrB_Vector I_vector,      // row indices
    uint64_t j,                     // column index
    const GrB_Descriptor desc       // descriptor for w, mask, and A
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (w) ;
    GB_WHERE4 (w, mask, A, I_vector,
        "GxB_Col_extract_Vector (w, M, accum, A, I, j, desc)") ;
    GB_BURBLE_START ("GrB_extract") ;

    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (GB_IMPLIES (mask != NULL, GB_VECTOR_OK (mask))) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx1, xx2, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) mask, &Mask_comp, &Mask_struct) ;

    uint64_t ancols = (A_transpose ? GB_NROWS (A) : GB_NCOLS (A)) ;
    if (j >= ancols)
    { 
        GB_ERROR (GrB_INVALID_INDEX,
            "Column index j=" GBu " out of bounds; must be < " GBu ,
            j, ancols) ;
    }

    //--------------------------------------------------------------------------
    // get the index vector
    //--------------------------------------------------------------------------

    void *I = NULL ;
    size_t I_size = 0 ;
    int64_t ni = 0 ;
    GrB_Type I_type = NULL ;
    GB_OK (GB_ijxvector (I_vector, (w == I_vector), 0, desc, false,
        &I, &ni, &I_size, &I_type, Werk)) ;
    bool I_is_32 = (I_type == GrB_UINT32) ;

    //--------------------------------------------------------------------------
    // extract the jth column (or jth row if A is transposed) using GB_extract
    //--------------------------------------------------------------------------

    // construct the column index list J = [ j ] of length nj = 1
    uint64_t J [1] ;
    J [0] = j ;

    //--------------------------------------------------------------------------
    // do the work in GB_extract
    //--------------------------------------------------------------------------

    GB_OK (GB_extract (
        (GrB_Matrix) w,    C_replace,   // w as a matrix, and descriptor
        M, Mask_comp, Mask_struct,      // mask and its descriptor
        accum,                          // optional accum for z=accum(w,t)
        A,                 A_transpose, // A and its descriptor
        I, I_is_32, ni,                 // row indices
        J, false, 1,                    // one column index, nj = 1 (64-bit)
        Werk)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    GB_BURBLE_END ;
    return (GrB_SUCCESS) ;
}

