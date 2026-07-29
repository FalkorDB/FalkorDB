//------------------------------------------------------------------------------
// gbmx_usage: check usage and make sure GrB.init has been called
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is a gbmx_* utility but it calls GrB_* methods.  However, if GrB_init
// fails, it frees any memory it has allocated (such as the JIT hash table).
// Since GrB_init relies on the default allocates in arena 0 (malloc/free),
// memory failures are properly handled.

//------------------------------------------------------------------------------
// gbmx_usage
//------------------------------------------------------------------------------

void gbmx_usage     // check usage and make sure GrB.init has been called
(
    bool ok,                // if false, then usage is not correct
    const char *usage,      // error message if usage is not correct
    char err [ERRLEN]
)
{

    //--------------------------------------------------------------------------
    // make sure GrB.init has been called
    //--------------------------------------------------------------------------

    int GrB_init_has_been_called = 0 ;
    GxB_initialized (&GrB_init_has_been_called) ;

    if (!GrB_init_has_been_called)
    {

        //----------------------------------------------------------------------
        // tell MATLAB to call GrB_finalize when this mexFunction is cleared
        //----------------------------------------------------------------------

        mexAtExit (gb_at_exit) ;

        //----------------------------------------------------------------------
        // initialize GraphBLAS and set defaults for its use in MATLAB
        //----------------------------------------------------------------------

        OK (GrB_init (GrB_NONBLOCKING)) ;

        // use mxMalloc/mxFree for the MATLAB arena
        OK (GxB_arena_init (MXARENA, mxMalloc, mxCalloc, mxRealloc, mxFree)) ;

        OK (gbmx_defaults (err)) ;        // no memory allocated; "cannot" fail
    }

    //--------------------------------------------------------------------------
    // check usage
    //--------------------------------------------------------------------------

    if (!ok)
    {
        ERROR (usage, GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // get test coverage (not used in production; for testing only)
    //--------------------------------------------------------------------------

    gbcov_get ( ) ;
}

