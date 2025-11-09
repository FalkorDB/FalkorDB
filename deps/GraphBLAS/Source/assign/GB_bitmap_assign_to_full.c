//------------------------------------------------------------------------------
// GB_bitmap_assign_to_full:  all entries present in C; set bitmap to all 1's
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All entries in C are now present.  Either set all of C->b to 1, or free it
// and make C full.

#include "assign/GB_bitmap_assign_methods.h"

GB_CALLBACK_BITMAP_ASSIGN_TO_FULL_PROTO (GB_bitmap_assign_to_full)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_BITMAP (C)) ;

    //--------------------------------------------------------------------------
    // free the bitmap or set it to all ones
    //--------------------------------------------------------------------------

    if (GB_sparsity_control (C->sparsity_control, C->vdim) & GxB_FULL)
    { 
        // C is bitmap but can become full; convert it to full
        GB_FREE_MEMORY (&(C->b), C->b_size) ;
        C->nvals = -1 ;
    }
    else
    { 
        // all entries in C are now present; C remains bitmap
        int64_t cnzmax = C->vlen * C->vdim ;
        GB_memset (C->b, 1, cnzmax, nthreads_max) ;
        C->nvals = cnzmax ;
    }
}

