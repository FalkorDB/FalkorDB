//------------------------------------------------------------------------------
// GrB_Col_extract: w<M> = accum (w, A(I,j)) or A(j,I)'
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Extract a single row or column from a matrix.  Note that in the
// GraphBLAS spec, row and column vectors are indistinguishable.  In this
// implementation, both are the same as an n-by-1 GrB_Matrix, except with
// restrictions on the matrix operations that can be performed on them.

#include "GB_extract.h"
#include "GB_get_mask.h"

GrB_Info GrB_Col_extract        // w<M> = accum (w, A(I,j)) or (A(j,I))'
(
    GrB_Vector w,               // input/output vector for results
    const GrB_Vector M_in,      // optional mask for w, unused if NULL
    const GrB_BinaryOp accum,   // optional accum for z=accum(w,t)
    const GrB_Matrix A,         // first input:  matrix A
    const GrB_Index *I,         // row indices
    GrB_Index ni,               // number of row indices
    GrB_Index j,                // column index
    const GrB_Descriptor desc   // descriptor for w, M, and A
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (w, "GrB_Col_extract (w, M, accum, A, I, ni, j, desc)") ;
    GB_BURBLE_START ("GrB_extract") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    ASSERT (GB_VECTOR_OK (w)) ;
    ASSERT (GB_IMPLIES (M_in != NULL, GB_VECTOR_OK (M_in))) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        A_transpose, xx1, xx2, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) M_in, &Mask_comp, &Mask_struct) ;

    GrB_Index ancols = (A_transpose ? GB_NROWS (A) : GB_NCOLS (A)) ;
    if (j >= ancols)
    { 
        GB_ERROR (GrB_INVALID_INDEX,
            "Column index j=" GBu " out of bounds; must be < " GBu ,
            j, ancols) ;
    }

    //--------------------------------------------------------------------------
    // extract the jth column (or jth row if A is transposed) using GB_extract
    //--------------------------------------------------------------------------

    // construct the column index list J = [ j ] of length nj = 1
    GrB_Index J [1] ;
    J [0] = j ;

    //--------------------------------------------------------------------------
    // do the work in GB_extract
    //--------------------------------------------------------------------------

    info = GB_extract (
        (GrB_Matrix) w,    C_replace,   // w as a matrix, and descriptor
        M, Mask_comp, Mask_struct,      // mask and its descriptor
        accum,                          // optional accum for z=accum(w,t)
        A,                 A_transpose, // A and its descriptor
        I, ni,                          // row indices I and length ni
        J, 1,                           // one column index, nj = 1
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

