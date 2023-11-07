//------------------------------------------------------------------------------
// GB_emult_02_template: C = A.*B when A is sparse/hyper and B is bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse, with the same sparsity structure as A.  No mask is present, or
// M is bitmap/full.  A is sparse/hyper, and B is bitmap/full.

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const int64_t vlen = A->vlen ;

    const int8_t  *restrict Bb = B->b ;

    const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
    const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
    const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;

    #ifdef GB_JIT_KERNEL
    #define A_iso GB_A_ISO
    #define B_iso GB_B_ISO
    #else
    const bool A_iso = A->iso ;
    const bool B_iso = B->iso ;
    #endif

    #ifdef GB_ISO_EMULT
    ASSERT (C->iso) ;
    #else
    ASSERT (!C->iso) ;
    ASSERT (!(A_iso && B_iso)) ;    // one of A or B can be iso, but not both
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    const int64_t  *restrict Cp = C->p ;
          int64_t  *restrict Ci = C->i ;

    #ifdef GB_JIT_KERNEL
    #define Mask_comp   GB_MASK_COMP
    #define Mask_struct GB_MASK_STRUCT
    #endif

    //--------------------------------------------------------------------------
    // C=A.*B or C<#M>=A.*B
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL

        #if GB_NO_MASK
        {
            #if GB_B_IS_BITMAP
            {
                // C=A.*B, where A is sparse/hyper and B is bitmap
                #include "GB_emult_02a.c"
            }
            #else
            {
                // C=A.*B, where A is sparse/hyper and B is full
                #include "GB_emult_02b.c"
            }
            #endif
        }
        #else
        {
            // C<#M>=A.*B, where A is sparse/hyper; M and B are bitmap/full
            #include "GB_emult_02c.c"
        }
        #endif

    #else

        if (M == NULL)
        {
            if (GB_IS_BITMAP (B))
            { 
                // C=A.*B, where A is sparse/hyper and B is bitmap
                #include "GB_emult_02a.c"
            }
            else
            { 
                // C=A.*B, where A is sparse/hyper and B is full
                #include "GB_emult_02b.c"
            }
        }
        else
        { 
            // C<#M>=A.*B, where A is sparse/hyper; M and B are bitmap/full
            #include "GB_emult_02c.c"
        }

    #endif
}

