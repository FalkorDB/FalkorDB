//------------------------------------------------------------------------------
// GxB_finalized: determine if GraphBLAS has been finalized
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GxB_finalized returns true if GrB_finalize has been called, or false if it
// not been.  When GrB_init is called again, this routine will return false,
// thus indicating that GrB_finalize can be called again.

// This method can be called at any time: before GrB_init, after GrB_init but
// before GrB_finalized, or after GrB_finalize.  If GrB_init has not yet been
// called, this method returns true.

#include "GB.h"
#include "init/GB_init.h"

GrB_Info GxB_finalized      // determine if GraphBLAS is finalized
(
    int *flag               // returns true if GrB_init or GxB_init has not
                            // yet been called or if GrB_finalize has been
                            // called, false otherwise
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (flag == NULL)
    { 
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // check if GraphBLAS has been finalized
    //--------------------------------------------------------------------------

    (*flag) = !GB_Global_GrB_init_called_get ( ) ;
    return (GrB_SUCCESS) ;
}

