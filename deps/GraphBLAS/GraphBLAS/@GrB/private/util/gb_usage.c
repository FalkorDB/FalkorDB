//------------------------------------------------------------------------------
// gb_usage: check usage and make sure GrB.init has been called
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

void gb_usage       // check usage and make sure GrB.init has been called
(
    bool ok,                // if false, then usage is not correct
    const char *usage       // error message if usage is not correct
)
{

    //--------------------------------------------------------------------------
    // clear the debug memory table (for debugging only)
    //--------------------------------------------------------------------------

    GB_Global_memtable_clear ( ) ;

    //--------------------------------------------------------------------------
    // make sure GrB.init has been called
    //--------------------------------------------------------------------------

    if (!GB_Global_GrB_init_called_get ( ))
    {

        //----------------------------------------------------------------------
        // tell MATLAB to call GrB_finalize when this mexFunction is cleared
        //----------------------------------------------------------------------

        mexAtExit (gb_at_exit) ;

        //----------------------------------------------------------------------
        // tell GraphBLAS how to tell MATLAB to make memory persistent
        //----------------------------------------------------------------------

        GB_Global_persistent_set (mexMakeMemoryPersistent) ;

        //----------------------------------------------------------------------
        // initialize GraphBLAS and set defaults for its use in MATLAB
        //----------------------------------------------------------------------

        OK (GxB_init (GrB_NONBLOCKING, mxMalloc, mxCalloc, mxRealloc, mxFree)) ;
        gb_defaults ( ) ;
    }

    //--------------------------------------------------------------------------
    // check usage
    //--------------------------------------------------------------------------

    if (!ok)
    {
        ERROR (usage) ;
    }

    //--------------------------------------------------------------------------
    // get test coverage
    //--------------------------------------------------------------------------

    #ifdef GBCOV
    gbcov_get ( ) ;
    #endif
}

