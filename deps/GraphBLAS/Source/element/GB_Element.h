//------------------------------------------------------------------------------
// GB_Element.h: definitions for GB_*Element methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ELEMENT_H
#define GB_ELEMENT_H

GrB_Info GB_setElement              // set a single entry, C(row,col) = scalar
(
    GrB_Matrix C,                   // matrix to modify
    const GrB_BinaryOp accum,       // if NULL: C(row,col) = scalar
                                    // else: C(row,col) += scalar
    const void *scalar,             // scalar to set
    const GrB_Index row,            // row index
    const GrB_Index col,            // column index
    const GB_Type_code scalar_code, // type of the scalar
    GB_Werk Werk
) ;

GrB_Info GB_Vector_removeElement
(
    GrB_Vector V,               // vector to remove entry from
    GrB_Index i,                // index
    GB_Werk Werk
) ;

GrB_Info GB_Matrix_removeElement
(
    GrB_Matrix C,               // matrix to remove entry from
    GrB_Index row,              // row index
    GrB_Index col,              // column index
    GB_Werk Werk
) ;

#endif

