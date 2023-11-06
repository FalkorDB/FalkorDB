//------------------------------------------------------------------------------
// GB_convert_s2b_template: convert A from sparse to bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is sparse or hypersparse.  Axnew and Ab have the same type as A,
// and represent a bitmap format.

{

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t *restrict Ai = A->i ;
    const int64_t avlen = A->vlen ;

    #if defined ( GB_A_TYPE )
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
          GB_A_TYPE *restrict Axnew = (GB_A_TYPE *) Ax_new ;
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
            #include "GB_convert_s2b_zombies.c"
        }
        #else
        {
            #include "GB_convert_s2b_nozombies.c"
        }
        #endif
    }
    #else
    {
        if (nzombies > 0)
        { 
            #include "GB_convert_s2b_zombies.c"
        }
        else
        { 
            #include "GB_convert_s2b_nozombies.c"
        }
    }
    #endif
}

#undef GB_A_TYPE

