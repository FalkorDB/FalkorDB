//------------------------------------------------------------------------------
// GB_bitmap_assign_to_full:  make a full bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// All entries in C are now present.  Either set all of C->b to 1, or free it
// and make C full.

#include "GB_bitmap_assign_methods.h"

void GB_bitmap_assign_to_full   // set all C->b to 1, or free it and make C full
(
    GrB_Matrix C,
    int nthreads_max
)
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
        GB_FREE (&(C->b), C->b_size) ;
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

