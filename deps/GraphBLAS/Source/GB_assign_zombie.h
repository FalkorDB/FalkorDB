//------------------------------------------------------------------------------
// GB_assign_zombie.h: definitions for GB_assign_zombie* functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_ASSIGN_ZOMBIE_H
#define GB_ASSIGN_ZOMBIE_H
#include "GB_ij.h"

GrB_Info GB_assign_zombie1
(
    GrB_Matrix C,
    const int64_t j
) ;

GrB_Info GB_assign_zombie2
(
    GrB_Matrix C,
    const int64_t i
) ;

GrB_Info GB_assign_zombie3
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,         // if true, use the only structure of M
    const int64_t j,
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3]
) ;

GrB_Info GB_assign_zombie4
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,         // if true, use the only structure of M
    const int64_t i,
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3]
) ;

GrB_Info GB_assign_zombie5
(
    GrB_Matrix C,
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Index *I,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    GB_Werk Werk
) ;

#endif

