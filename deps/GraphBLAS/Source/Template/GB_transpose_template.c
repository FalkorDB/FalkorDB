//------------------------------------------------------------------------------
// GB_transpose_template: C=op(cast(A')), transpose, typecast, and apply op
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

{

    // Ax unused for some uses of this template
    #include "GB_unused.h"

    //--------------------------------------------------------------------------
    // get A and C
    //--------------------------------------------------------------------------

    #undef GBH_S
    #undef GB_S_TYPE

    #ifdef GB_BIND_1ST
        // A is the name of the matrix passed in to this kernel, but it takes
        // the place of the B matrix for C=op(x,A').  As a result, the B macros
        // must be used to access its contents.
        #define GBH_S(Ah,k) GBH_B(Ah,k)
        #define GB_S_TYPE GB_B_TYPE
    #else
        // for bind2nd, unary ops, and mere typecasting, use the A macros to
        // access the A matrix.
        #define GBH_S(Ah,k) GBH_A(Ah,k)
        #define GB_S_TYPE GB_A_TYPE
    #endif

    #ifndef GB_ISO_TRANSPOSE
    const GB_S_TYPE *restrict Ax = (GB_S_TYPE *) A->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    //--------------------------------------------------------------------------
    // C = op (cast (A'))
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    {
        #if GB_C_IS_FULL
        {
            // A and C are both full
            #include "GB_transpose_full.c"
        }
        #elif GB_C_IS_BITMAP
        {
            // A and C are both bitmap
            #include "GB_transpose_bitmap.c"
        }
        #else
        {
            // A is sparse or hypersparse; C is sparse
            #include "GB_transpose_sparse.c"
        }
        #endif
    }
    #else
    {
        if (Workspaces == NULL)
        {
            // A and C are both full or both bitmap
            if (A->b == NULL)
            { 
                // A and C are both full
                #include "GB_transpose_full.c"
            }
            else
            { 
                // A and C are both bitmap
                #include "GB_transpose_bitmap.c"
            }
        }
        else
        { 
            // A is sparse or hypersparse; C is sparse
            #include "GB_transpose_sparse.c"
        }
    }
    #endif
}

#undef GB_ISO_TRANSPOSE
#undef GBH_S
#undef GB_S_TYPE

