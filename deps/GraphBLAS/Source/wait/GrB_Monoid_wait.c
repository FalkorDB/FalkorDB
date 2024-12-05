//------------------------------------------------------------------------------
// GrB_Monoid_wait: wait for a user-defined GrB_Monoid to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// In SuiteSparse:GraphBLAS, a user-defined GrB_Monoid has no pending
// operations to wait for.  All this method does is verify that the monoid is
// properly initialized, and then it does an OpenMP flush.

#include "GB.h"

GrB_Info GrB_Monoid_wait   // no work, just check if the GrB_Monoid is valid
(
    GrB_Monoid monoid,
    GrB_WaitMode waitmode
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Monoid_wait (monoid, waitmode)") ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

