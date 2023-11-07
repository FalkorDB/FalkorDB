//------------------------------------------------------------------------------
// GB_subassign_scalar: C(Rows,Cols)<M> = accum (C(Rows,Cols),x)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Assigns a single scalar to a submatrix:

// C(Rows,Cols)<M> = accum (C(Rows,Cols),x)

// This function does the work for GxB_Matrix_subassign_TYPE and
// GxB_Vector_subassign_[type], where [type] is one of the 11 types, or the
// type-generic "_UDT".

// Compare with GB_assign_scalar, which uses M and C_replace differently

#include "GB_subassign.h"
#include "GB_get_mask.h"

GrB_Info GB_subassign_scalar        // C(Rows,Cols)<M> += x
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // mask for C(Rows,Cols), unused if NULL
    const GrB_BinaryOp accum,       // accum for Z=accum(C(Rows,Cols),T)
    const void *scalar,             // scalar to assign to C(Rows,Cols)
    const GB_Type_code scalar_code, // type code of scalar to assign
    const GrB_Index *Rows,          // row indices
    const GrB_Index nRows,          // number of row indices
    const GrB_Index *Cols,          // column indices
    const GrB_Index nCols,          // number of column indices
    const GrB_Descriptor desc,      // descriptor for C(Rows,Cols) and M
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (scalar) ;
    GB_RETURN_IF_NULL (Rows) ;
    GB_RETURN_IF_NULL (Cols) ;
    ASSERT (scalar_code <= GB_UDT_code) ;

    // get the descriptor
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // C(Rows,Cols)<M> = accum (C(Rows,Cols), scalar)
    //--------------------------------------------------------------------------

    if (M == NULL && !Mask_comp && nRows == 1 && nCols == 1 && !C_replace)
    { 
        // C(i,j) = scalar or C(i,j) += scalar
        return (GB_setElement (C, accum, scalar, Rows [0], Cols [0],
            scalar_code, Werk)) ;
    }
    else
    { 
        return (GB_subassign (
            C, C_replace,               // C matrix and its descriptor
            M, Mask_comp, Mask_struct,  // mask matrix and its descriptor
            false,                      // do not transpose the mask
            accum,                      // for accum (C(Rows,Cols),scalar)
            NULL, false,                // no explicit matrix A
            Rows, nRows,                // row indices
            Cols, nCols,                // column indices
            true,                       // do scalar expansion
            scalar,                     // scalar to assign, expands to become A
            scalar_code,                // type code of scalar to expand
            Werk)) ;
    }
}
