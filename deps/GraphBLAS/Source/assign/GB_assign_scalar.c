//------------------------------------------------------------------------------
// GB_assign_scalar:    C<M>(Rows,Cols) = accum (C(Rows,Cols),x)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Assigns a single scalar to a submatrix:

// C<M>(Rows,Cols) = accum (C(Rows,Cols),x)

// This function does the work for GrB_Matrix_assign_TYPE and
// GrB_Vector_assign_[type], where [type] is one of the 11 types, or the
// type-generic "_UDT".

// Compare with GB_subassign_scalar, which uses M and C_replace differently

#include "assign/GB_assign.h"
#include "assign/GB_bitmap_assign.h"
#include "mask/GB_get_mask.h"

GrB_Info GB_assign_scalar           // C<M>(Rows,Cols) += x
(
    GrB_Matrix C,                   // input/output matrix for results
    const GrB_Matrix M_in,          // mask for C(Rows,Cols), unused if NULL
    const GrB_BinaryOp accum,       // accum for Z=accum(C(Rows,Cols),T)
    const void *scalar,             // scalar to assign to C(Rows,Cols)
    const GB_Type_code scalar_code, // type code of scalar to assign
    const void *Rows,               // row indices
    const bool Rows_is_32,          // if true, Rows is 32-bit; else 64-bit
    const uint64_t nRows,           // number of row indices
    const void *Cols,               // column indices
    const bool Cols_is_32,          // if true, Cols is 32-bit; else 64-bit
    const uint64_t nCols,           // number of column indices
    const GrB_Descriptor desc,      // descriptor for C and M
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (scalar) ;
    GB_RETURN_IF_NULL (Rows) ;
    GB_RETURN_IF_NULL (Cols) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (C) ;
    ASSERT (scalar_code <= GB_UDT_code) ;

    // get the descriptor
    GrB_Info info ;
    GB_GET_DESCRIPTOR (info, desc, C_replace, Mask_comp, Mask_struct,
        xx1, xx2, xx3, xx7) ;

    // get the mask
    GrB_Matrix M = GB_get_mask (M_in, &Mask_comp, &Mask_struct) ;

    //--------------------------------------------------------------------------
    // C<M>(Rows,Cols) = accum (C(Rows,Cols), scalar)
    //--------------------------------------------------------------------------

    if (M == NULL && !Mask_comp && nRows == 1 && nCols == 1 && !C_replace)
    { 
        // C(i,j) = scalar or C(i,j) += scalar
        GB_IDECL (Rows, const, u) ; GB_IPTR (Rows, Rows_is_32) ;
        GB_IDECL (Cols, const, u) ; GB_IPTR (Cols, Cols_is_32) ;
        return (GB_setElement (C, accum, scalar,
            GB_IGET (Rows, 0), GB_IGET (Cols, 0), scalar_code, Werk)) ;
    }
    else
    { 
        return (GB_assign (
            C, C_replace,               // C matrix and its descriptor
            M, Mask_comp, Mask_struct,  // mask matrix and its descriptor
            false,                      // do not transpose the mask
            accum,                      // for accum (C(Rows,Cols),scalar)
            NULL, false,                // no explicit matrix A
            Rows, Rows_is_32, nRows,    // row indices
            Cols, Cols_is_32, nCols,    // column indices
            true,                       // do scalar expansion
            scalar,                     // scalar to assign, expands to become A
            scalar_code,                // type code of scalar to expand
            GB_ASSIGN,
            Werk)) ;
    }
}

