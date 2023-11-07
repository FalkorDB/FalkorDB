//------------------------------------------------------------------------------
// GB_emult_03_template: C = A.*B when A is bitmap/full and B is sparse/hyper
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C is sparse, with the same sparsity structure as B.  No mask is present, or
// M is bitmap/full.  A is bitmap/full, and B is sparse/hyper.

{

    //--------------------------------------------------------------------------
    // get A, B, and C
    //--------------------------------------------------------------------------

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int64_t *restrict Bi = B->i ;
    const int64_t vlen = B->vlen ;

    const int8_t  *restrict Ab = A->b ;

    const int64_t *restrict kfirst_Bslice = B_ek_slicing ;
    const int64_t *restrict klast_Bslice  = B_ek_slicing + B_ntasks ;
    const int64_t *restrict pstart_Bslice = B_ek_slicing + B_ntasks * 2 ;

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
            #if GB_A_IS_BITMAP
            {
                // C=A.*B, where A is bitmap and B is sparse/hyper
                #include "GB_emult_03a.c"
            }
            #else
            {
                // C=A.*B, where A is full and B is sparse/hyper
                #include "GB_emult_03b.c"
            }
            #endif
        }
        #else
        {
            // C<#M>=A.*B, where M and A are bitmap/full and B is sparse/hyper
            #include "GB_emult_03c.c"
        }
        #endif

    #else

        if (M == NULL)
        {
            if (GB_IS_BITMAP (A))
            { 
                // C=A.*B, where A is bitmap and B is sparse/hyper
                #include "GB_emult_03a.c"
            }
            else
            { 
                // C=A.*B, where A is full and B is sparse/hyper
                #include "GB_emult_03b.c"
            }
        }
        else
        { 
            // C<#M>=A.*B, where M and A are bitmap/full and B is sparse/hyper
            #include "GB_emult_03c.c"
        }

    #endif
}

