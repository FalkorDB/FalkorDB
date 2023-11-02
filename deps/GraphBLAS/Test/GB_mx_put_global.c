//------------------------------------------------------------------------------
// GB_mx_put_global: put the GraphBLAS status in global workspace
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A prior version of this method would call GB_mx_at_exit to finalize
// GraphBLAS (and allow it to be called again).  This is slow, however,
// so it has been removed.

#include "GB_mex.h"

void GB_mx_put_global
(
    bool cover
)
{

    //--------------------------------------------------------------------------
    // free the complex type and operators
    //--------------------------------------------------------------------------

    Complex_finalize ( ) ;

    //--------------------------------------------------------------------------
    // log the statement coverage
    //--------------------------------------------------------------------------

    GB_cover_put (cover) ;

    //--------------------------------------------------------------------------
    // check nmemtable and nmalloc
    //--------------------------------------------------------------------------

    int nmemtable = GB_Global_memtable_n ( ) ;
    if (nmemtable != 0)
    {
        printf ("in GB_mx_put_global: GraphBLAS nmemtable %d!\n", nmemtable) ;
        GB_Global_memtable_dump ( ) ;
        mexErrMsgTxt ("memory leak in test!") ;
    }

    int64_t nmalloc = GB_Global_nmalloc_get ( ) ;
    if (nmalloc != 0)
    {
        printf ("in GB_mx_put_global: GraphBLAS nmalloc "GBd"!\n", nmalloc) ;
        GB_Global_memtable_dump ( ) ;
        mexErrMsgTxt ("memory leak in test!") ;
    }
}

