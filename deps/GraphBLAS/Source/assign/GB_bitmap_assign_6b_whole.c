//------------------------------------------------------------------------------
// GB_bitmap_assign_6b_whole:  C bitmap, no M, no accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C = A, C is bitmap, A is sparse/hyper

#include "assign/GB_bitmap_assign_methods.h"
#include "assign/GB_subassign_dense.h"
#include "jitifyer/GB_stringify.h"

GrB_Info GB_bitmap_assign_6b_whole  // C bitmap, no M, no accum
(
    // input/output:
    GrB_Matrix C,               // input/output matrix in bitmap format
    // inputs:
    #define C_replace false
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
    #define M NULL              /* mask matrix, not present here */
    #define Mask_comp false
    #define Mask_struct true
    #define accum NULL          /* not present */
    const GrB_Matrix A,         // input matrix, not transposed
    #define scalar NULL
    #define scalar_type NULL
    #define assign_kind         GB_ASSIGN
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_BITMAP (C)) ;
    ASSERT (GB_IS_HYPERSPARSE (A) || GB_IS_SPARSE (A)) ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign_6b_whole", GB0) ;
    ASSERT_MATRIX_OK (A, "A for bitmap assign_6b_whole", GB0) ;

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    GrB_Info info = GB_subassign_jit (C, C_replace,
        I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
        M, Mask_comp, Mask_struct, accum, A, scalar, scalar_type,
        /* S: */ NULL, assign_kind,
        GB_JIT_KERNEL_BITMAP_ASSIGN_6b_WHOLE, "bitmap_assign_6b_whole",
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
    #include "assign/template/GB_bitmap_assign_6b_whole_template.c"
}

