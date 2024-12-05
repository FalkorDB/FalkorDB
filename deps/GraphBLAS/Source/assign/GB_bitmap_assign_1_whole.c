//------------------------------------------------------------------------------
// GB_bitmap_assign_1_whole: C bitmap, M bitmap/full, with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<M> += A           assign
// C<M> += A           subassign

// C<M,repl> += A      assign
// C<M,repl> += A      subassign

// C<!M> += A          assign
// C<!M> += A          subassign

// C<!M,repl> += A     assign
// C<!M,repl> += A     subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, bitmap or full (not hypersparse or sparse)
// Mask_comp:   true or false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign or subassign (same action)

// If C were full: entries can be deleted only if C_replace is true.

#include "assign/GB_bitmap_assign_methods.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GB_bitmap_assign_1_whole   // C bitmap, M bitmap/full, with accum
(
    // input/output:
    GrB_Matrix C,               // input/output matrix in bitmap format
    // inputs:
    const bool C_replace,       // descriptor for C
    #define I NULL              /* I index list */
    #define ni 0
    #define nI 0
    #define Ikind GB_ALL
    #define Icolon NULL
    #define J NULL              /* J index list */
    #define nj 0
    #define nJ 0
    #define Jkind GB_ALL
    #define Jcolon NULL
    const GrB_Matrix M,         // mask matrix, which is present here
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // present here
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    #define assign_kind GB_ASSIGN
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_assign_burble ("bit1_whole", C_replace, Ikind, Jkind,
        M, Mask_comp, Mask_struct, accum, A, assign_kind) ;

    ASSERT (GB_IS_BITMAP (C)) ;
    ASSERT (GB_IS_BITMAP (M) || GB_IS_FULL (M)) ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign, M full, accum, whole", GB0) ;
    ASSERT_MATRIX_OK (M, "M for bitmap assign, M full, accum, whole", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (A, "A for bitmap_assign_1_whole", GB0) ;

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    GrB_Info info = GB_subassign_jit (C, C_replace,
        I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
        M, Mask_comp, Mask_struct, accum, A, scalar, scalar_type,
        /* S: */ NULL, assign_kind,
        GB_JIT_KERNEL_BITMAP_ASSIGN_1_WHOLE, "bitmap_assign_1_whole",
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
    #include "assign/template/GB_bitmap_assign_1_whole_template.c"
}

