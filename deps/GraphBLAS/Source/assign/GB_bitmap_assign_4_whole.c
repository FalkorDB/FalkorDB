//------------------------------------------------------------------------------
// GB_bitmap_assign_4_whole: C bitmap, M sparse/hyper, no accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<M> = A       assign
// C<M> = A       subassign

// C<M,repl> = A       assign
// C<M,repl> = A       subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, hypersparse or sparse, (not bitmap or full)
// Mask_comp:   false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       not present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign or subassign (same action)

// If C were full: entries can be deleted if C_replace is true,
// or if A is not full and missing at least one entry.

#include "assign/GB_bitmap_assign_methods.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GB_bitmap_assign_4_whole   // C bitmap, M sparse/hyper, no accum
(
    // input/output:
    GrB_Matrix C,               // input/output matrix in bitmap format
    // inputs:
    const bool C_replace,       // descriptor for C
    #define I NULL              /* I index list */
    #define I_is_32 false
    #define ni 0
    #define nI 0
    #define Ikind GB_ALL
    #define Icolon NULL
    #define J NULL              /* J index list */
    #define J_is_32 false
    #define nj 0
    #define nJ 0
    #define Jkind GB_ALL
    #define Jcolon NULL
    const GrB_Matrix M,         // mask matrix, which is present here
    #define Mask_comp false
    const bool Mask_struct,     // true if M is structural, false if valued
    #define accum NULL
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    #define assign_kind         GB_ASSIGN
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_assign_burble ("bit4_whole", C_replace, Ikind, Jkind,
        M, Mask_comp, Mask_struct, accum, A, assign_kind) ;

    ASSERT (GB_IS_BITMAP (C)) ;
    ASSERT (GB_IS_HYPERSPARSE (M) || GB_IS_SPARSE (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign, M, noaccum", GB0) ;
    ASSERT_MATRIX_OK (M, "M for bitmap assign, M, noaccum", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (A, "A for bitmap assign, M, noaccum", GB0) ;

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    GrB_Info info = GB_subassign_jit (C, C_replace,
        I, I_is_32, ni, nI, Ikind, Icolon, J, J_is_32, nj, nJ, Jkind, Jcolon,
        M, Mask_comp, Mask_struct, accum, A, scalar, scalar_type,
        /* S: */ NULL, assign_kind,
        GB_JIT_KERNEL_BITMAP_ASSIGN_4_WHOLE, "bitmap_assign_4_whole",
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
    #include "assign/template/GB_bitmap_assign_4_whole_template.c"
}

