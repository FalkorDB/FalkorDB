//------------------------------------------------------------------------------
// GxB_Col_subassign: C(Rows,col)<M> = accum (C(Rows,col),u)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Compare with GrB_Col_assign, which uses M and C_replace differently

#include "GB_subassign.h"
#include "GB_get_mask.h"

GrB_Info GxB_Col_subassign          // C(Rows,col)<M> = accum (C(Rows,col),u)
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector M_in,          // mask for C(Rows,col), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(Rows,col),t)
    const GrB_Vector u,             // input vector
    const GrB_Index *Rows,          // row indices
    GrB_Index nRows,                // number of row indices
    GrB_Index col,                  // column index
    const GrB_Descriptor desc       // descriptor for C(Rows,col) and M
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GxB_Col_subassign (C, M, accum, u, Rows, nRows, col, desc)") ;
    GB_BURBLE_START ("GxB_subassign") ;
    GB_RETURN_IF_NULL_OR_FAULTY (C) ;
    GB_RETURN_IF_FAULTY (M_in) ;
    GB_RETURN_IF_NULL_OR_FAULTY (u) ;

    ASSERT (M_in == NULL || GB_VECTOR_OK (M_in)) ;
    ASSERT (GB_VECTOR_OK (u)) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask ((GrB_Matrix) M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // C(Rows,col)<M> = accum (C(Rows,col), u) and variations
    //--------------------------------------------------------------------------

    // construct the column index list Cols = [ col ] of length nCols = 1
    GrB_Index Cols [1] ;
    Cols [0] = col ;

    info = GB_subassign (
        C, C_replace,                   // C matrix and its descriptor
        M, Mask_comp, Mask_struct,      // mask and its descriptor
        false,                          // do not transpose the mask
        accum,                          // for accum (C(Rows,col),u)
        (GrB_Matrix) u, false,          // u as a matrix; never transposed
        Rows, nRows,                    // row indices
        Cols, 1,                        // a single column index
        false, NULL, GB_ignore_code,    // no scalar expansion
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

