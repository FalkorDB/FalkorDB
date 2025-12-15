//------------------------------------------------------------------------------
// GB_subassign_methods.h: definitions for GB_subassign methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_SUBASSIGN_METHODS_H
#define GB_SUBASSIGN_METHODS_H

#include "add/GB_add.h"
#include "ij/GB_ij.h"
#include "pending/GB_Pending.h"
#include "include/GB_unused.h"

//------------------------------------------------------------------------------
// GB_subassign_symbolic: S = C(I,J)
//------------------------------------------------------------------------------

GrB_Info GB_subassign_symbolic  // S = C(I,J), extracting pattern not values
(
    // output
    GrB_Matrix S,           // S = symbolic(C(I,J)), static header
    // inputs, not modified:
    const GrB_Matrix C,     // matrix to extract the pattern of
    const void *I,          // I index list
    const bool I_is_32,
    const int64_t ni,       // length of I, or special
    const void *J,          // J index list
    const bool J_is_32,
    const int64_t nj,       // length of J, or special
    const bool S_must_not_be_jumbled,   // if true, S cannot be jumbled
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_zombie: C(I,J) = empty ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_zombie
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_01: C(I,J) = scalar ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_01
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_02: C(I,J) = A ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_02
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix A,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_03: C(I,J) += scalar ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_03
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_BinaryOp accum,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_04: C(I,J) += A ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_04
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_05: C(I,J)<M> = scalar ; no S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_05
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_struct,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_05e: C(:,:)<M,struct> = scalar ; no S, C empty
//------------------------------------------------------------------------------

GrB_Info GB_subassign_05e
(
    GrB_Matrix C,
    // input:
    const GrB_Matrix M,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_06n: C(I,J)<M> = A ; no S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_06n
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_Matrix A,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_06s_and_14: C(I,J)<M or !M> = A ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_06s_and_14
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const GrB_Matrix A,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_07: C(I,J)<M> += scalar ; no S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_07
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_BinaryOp accum,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_08n: C(I,J)<M> += A ; no S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_08n
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_09: C(I,J)<M,repl> = scalar ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_09
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_struct,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_10_and_18: C(I,J)<M or !M,repl> = A ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_10_and_18
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const GrB_Matrix A,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_11: C(I,J)<M,repl> += scalar ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_11
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_BinaryOp accum,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_12_and_20: C(I,J)<M or !M,repl> += A ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_12_and_20
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_comp,           // if true, !M, else use M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_13: C(I,J)<!M> = scalar ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_13
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_struct,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_15: C(I,J)<!M> += scalar ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_15
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_BinaryOp accum,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_16:  C(I,J)<!M> += A ; using S
// GB_subassign_08s: C(I,J)<M> += A ; using S.  Compare with method 08n
//------------------------------------------------------------------------------

GrB_Info GB_subassign_08s_and_16
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_comp,
    const bool Mask_struct,
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_17: C(I,J)<!M,repl> = scalar ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_17
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_struct,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_19: C(I,J)<!M,repl> += scalar ; using S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_19
(
    GrB_Matrix C,
    // input:
    const void *I,              // I index list
    const bool I_is_32,
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const void *J,              // J index list
    const bool J_is_32,
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    const GrB_Matrix M,
    const bool Mask_struct,
    const GrB_BinaryOp accum,
    const void *scalar,
    const GrB_Type scalar_type,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_26: C(:,j) = A, append column, no S
//------------------------------------------------------------------------------

GrB_Info GB_subassign_26
(
    GrB_Matrix C,
    // input:
    const int64_t Jcolon [3],       // j1:j2, with an increment of 1
    const GrB_Matrix A,
    GB_Werk Werk
) ;

//------------------------------------------------------------------------------
// GB_subassign_27: C<C,struct> += A
//------------------------------------------------------------------------------

GrB_Info GB_subassign_27
(
    GrB_Matrix C,
    // input:
    const GrB_BinaryOp accum,
    const GrB_Matrix A,
    GB_Werk Werk
) ;

#endif

