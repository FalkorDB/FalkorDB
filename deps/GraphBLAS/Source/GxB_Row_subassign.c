//------------------------------------------------------------------------------
// GxB_Row_subassign: C(row,Cols)<M'> = accum (C(row,Cols),u')
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Compare with GrB_Row_assign, which uses the M and C_replace differently

#include "GB_subassign.h"
#include "GB_get_mask.h"

GrB_Info GxB_Row_subassign          // C(row,Cols)<M'> += u'
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Vector M_in,          // mask for C(row,Cols), unused if NULL
    const GrB_BinaryOp accum,       // optional accum for z=accum(C(row,Cols),t)
    const GrB_Vector u,             // input vector
    GrB_Index row,                  // row index
    const GrB_Index *Cols,          // column indices
    GrB_Index nCols,                // number of column indices
    const GrB_Descriptor desc       // descriptor for C(row,Cols) and M
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (C, "GxB_Row_subassign (C, M, accum, u, row, Cols, nCols, desc)") ;
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
    // C(row,Cols)<M'> = accum (C(row,Cols), u')
    //--------------------------------------------------------------------------

    // construct the row index list Rows = [ row ] of length nRows = 1
    GrB_Index Rows [1] ;
    Rows [0] = row ;

    info = GB_subassign (
        C, C_replace,                   // C matrix and its descriptor
        M, Mask_comp, Mask_struct,      // mask and its descriptor
        true,                           // transpose the mask
        accum,                          // for accum (C(Rows,col),u)
        (GrB_Matrix) u, true,           // u as a matrix; always transposed
        Rows, 1,                        // a single row index
        Cols, nCols,                    // column indices
        false, NULL, GB_ignore_code,    // no scalar expansion
        Werk) ;

    GB_BURBLE_END ;
    return (info) ;
}

