//------------------------------------------------------------------------------
// GrB_Semiring_wait: wait for a user-defined GrB_Semiring to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// In SuiteSparse:GraphBLAS, a user-defined GrB_Semiring has no pending
// operations to wait for.  All this method does is verify that the semiring is
// properly initialized, and then it does an OpenMP flush.

#include "GB.h"

GrB_Info GrB_Semiring_wait   // no work, just check if the GrB_Semiring is valid
(
    GrB_Semiring semiring,
    int waitmode
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

