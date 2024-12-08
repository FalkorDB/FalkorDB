//------------------------------------------------------------------------------
// GB_bitmap_assign_5: C bitmap, no M, with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<>(I,J) += A            assign
// C(I,J)<> += A            subassign

// C<repl>(I,J) += A        assign
// C(I,J)<repl> += A        subassign

// C<!>(I,J) += A           assign: no work to do
// C(I,J)<!> += A           subassign: no work to do

// C<!,repl>(I,J) += A      assign: just clear C(I,J) of all entries
// C(I,J)<!,repl> += A      subassign: just clear C(I,J) of all entries
//------------------------------------------------------------------------------

// C:           bitmap
// M:           none
// Mask_comp:   true or false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign, row assign, col assign, or subassign (all the same)

// If Mask_comp is true, then an empty mask is complemented.  This case has
// already been handled by GB_assign_prep, which calls GB_bitmap_assign with a
// scalar (which is unused).

// If C were full: entries can be deleted only if C_replace is true.

#include "assign/GB_bitmap_assign_methods.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GB_bitmap_assign_5     // C bitmap, no M, with accum
(
    // input/output:
    GrB_Matrix C,               // input/output matrix in bitmap format
    // inputs:
    const bool C_replace,       // descriptor for C
    const GrB_Index *I,         // I index list
    const int64_t ni,
    const int64_t nI,
    const int Ikind,
    const int64_t Icolon [3],
    const GrB_Index *J,         // J index list
    const int64_t nj,
    const int64_t nJ,
    const int Jkind,
    const int64_t Jcolon [3],
    #define M NULL              /* mask matrix, not present here */
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // present
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    const int assign_kind,      // row assign, col assign, assign, or subassign
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_assign_burble ("bit5", C_replace, Ikind, Jkind,
        M, Mask_comp, Mask_struct, accum, A, assign_kind) ;

    ASSERT (GB_IS_BITMAP (C)) ;
    ASSERT_MATRIX_OK (C, "C for bitmap_assign_5", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (A, "A for bitmap_assign_5", GB0) ;
    ASSERT_BINARYOP_OK (accum, "accum for bitmap_assign_5", GB0) ; 

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    GrB_Info info = GB_subassign_jit (C, C_replace,
        I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
        M, Mask_comp, Mask_struct, accum, A, scalar, scalar_type,
        /* S: */ NULL, assign_kind,
        GB_JIT_KERNEL_BITMAP_ASSIGN_5, "bitmap_assign_5",
        Werk) ;
    if (info != GrB_NO_VALUE)
    { 
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    GBURBLE ("(generic assign) ") ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    #define GB_GENERIC
    #include "assign/include/GB_assign_shared_definitions.h"
    #include "assign/template/GB_bitmap_assign_5_template.c"
}

