//------------------------------------------------------------------------------
// GB_bitmap_assign_5_whole: C bitmap, no M, with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<> += A            assign
// C<> += A            subassign

// C<repl> += A        assign
// C<repl> += A        subassign

// C<!> += A           assign: no work to do
// C<!> += A           subassign: no work to do

// C<!,repl> += A      assign: just clear C of all entries, not done here
// C<!,repl> += A      subassign: just clear C of all entries, not done here
//------------------------------------------------------------------------------

// C:           bitmap
// M:           none
// Mask_comp:   true or false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign or subassign (same action)

// If Mask_comp is true, then an empty mask is complemented.  This case has
// already been handled by GB_assign_prep, which calls GB_clear, and thus
// Mask_comp is always false in this method.

// If C were full: entries can be deleted only if C_replace is true.

#include "assign/GB_bitmap_assign_methods.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GB_bitmap_assign_5_whole   // C bitmap, no M, with accum
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
    #define M NULL              /* mask matrix, not present here */
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
    const GrB_BinaryOp accum,   // present
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

    GB_assign_burble ("bit5_whole", C_replace, Ikind, Jkind,
        M, Mask_comp, Mask_struct, accum, A, assign_kind) ;

    ASSERT (GB_IS_BITMAP (C)) ;
    ASSERT_MATRIX_OK (C, "C for bitmap_assign_5_whole", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (A, "A for bitmap_assign_5_whole", GB0) ;
    ASSERT_BINARYOP_OK (accum, "accum for bitmap_assign_5_whole", GB0) ;

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    GrB_Info info = GB_subassign_jit (C, C_replace,
        I, I_is_32, ni, nI, Ikind, Icolon, J, J_is_32, nj, nJ, Jkind, Jcolon,
        M, Mask_comp, Mask_struct, accum, A, scalar, scalar_type,
        /* S: */ NULL, assign_kind,
        GB_JIT_KERNEL_BITMAP_ASSIGN_5_WHOLE, "bitmap_assign_5_whole",
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
    #include "assign/template/GB_bitmap_assign_5_whole_template.c"
}

