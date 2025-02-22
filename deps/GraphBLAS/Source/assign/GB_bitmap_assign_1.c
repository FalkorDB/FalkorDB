//------------------------------------------------------------------------------
// GB_bitmap_assign_1: C bitmap, M bitmap/full, with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<M>(I,J) += A           assign
// C(I,J)<M> += A           subassign

// C<M,repl>(I,J) += A      assign
// C(I,J)<M,repl> += A      subassign

// C<!M>(I,J) += A          assign
// C(I,J)<!M> += A          subassign

// C<!M,repl>(I,J) += A     assign
// C(I,J)<!M,repl> += A     subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, bitmap or full (not hypersparse or sparse)
// Mask_comp:   true or false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign, row assign, col assign, or subassign

// If C were full: entries can be deleted only if C_replace is true.

#include "assign/GB_bitmap_assign_methods.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GB_bitmap_assign_1     // C bitmap, M bitmap/full, with accum
(
    // input/output:
    GrB_Matrix C,               // input/output matrix in bitmap format
    // inputs:
    const bool C_replace,       // descriptor for C
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
    const GrB_Matrix M,         // mask matrix, which is present here
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // present here
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

    GB_assign_burble ("bit1", C_replace, Ikind, Jkind,
        M, Mask_comp, Mask_struct, accum, A, assign_kind) ;

    ASSERT (GB_IS_BITMAP (C)) ;
    ASSERT (GB_IS_BITMAP (M) || GB_IS_FULL (M)) ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign, M full, accum", GB0) ;
    ASSERT_MATRIX_OK (M, "M for bitmap assign, M full, accum", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (A, "A for bitmap assign, M full, accum", GB0) ;

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    GrB_Info info = GB_subassign_jit (C, C_replace,
        I, I_is_32, ni, nI, Ikind, Icolon, J, J_is_32, nj, nJ, Jkind, Jcolon,
        M, Mask_comp, Mask_struct, accum, A, scalar, scalar_type,
        /* S: */ NULL, assign_kind,
        GB_JIT_KERNEL_BITMAP_ASSIGN_1, "bitmap_assign_1",
        Werk) ;
    if (info != GrB_NO_VALUE)
    { 
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    GB_IDECL (I, const, u) ; GB_IPTR (I, I_is_32) ;
    GB_IDECL (J, const, u) ; GB_IPTR (J, J_is_32) ;

    GBURBLE ("(generic assign) ") ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    #define GB_GENERIC
    #include "assign/include/GB_assign_shared_definitions.h"
    #include "assign/template/GB_bitmap_assign_1_template.c"
}

