//------------------------------------------------------------------------------
// GxB_Monoid_identity: return the identity of a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Monoid_identity        // return the monoid identity
(
    void *identity,                 // returns the identity of the monoid
    GrB_Monoid monoid               // monoid to query
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Monoid_identity (&identity, monoid)") ;
    GB_RETURN_IF_NULL (identity) ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    ASSERT_MONOID_OK (monoid, "monoid for identity", GB0) ;

    //--------------------------------------------------------------------------
    // return the identity
    //--------------------------------------------------------------------------

    memcpy (identity, monoid->identity, monoid->op->ztype->size) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

