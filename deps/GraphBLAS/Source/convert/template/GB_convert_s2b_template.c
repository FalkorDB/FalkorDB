//------------------------------------------------------------------------------
// GB_convert_s2b_template: convert A from sparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is sparse or hypersparse.  Cx and Cb have the same type as A,
// and represent a bitmap format.

{

    //--------------------------------------------------------------------------
    // get A and Cx_new
    //--------------------------------------------------------------------------

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    GB_Ai_DECLARE (Ai, const) ; GB_Ai_PTR (Ai, A) ;
    const int64_t avlen = A->vlen ;
    #ifdef GB_A_TYPE
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_A_TYPE *restrict Cx = (GB_A_TYPE *) Cx_new ;
    #endif

    //--------------------------------------------------------------------------
    // convert from sparse/hyper to bitmap
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    {
        const int64_t *restrict kfirst_Aslice = A_ek_slicing ;
        const int64_t *restrict klast_Aslice  = A_ek_slicing + A_ntasks ;
        const int64_t *restrict pstart_Aslice = A_ek_slicing + A_ntasks * 2 ;
        #if GB_A_HAS_ZOMBIES
        {
            #include "template/GB_convert_s2b_zombies.c"
        }
        #else
        {
            #include "template/GB_convert_s2b_nozombies.c"
        }
        #endif
    }
    #else
    {
        if (nzombies > 0)
        { 
            #include "template/GB_convert_s2b_zombies.c"
        }
        else
        { 
            #include "template/GB_convert_s2b_nozombies.c"
        }
    }
    #endif
}

#undef GB_A_TYPE

