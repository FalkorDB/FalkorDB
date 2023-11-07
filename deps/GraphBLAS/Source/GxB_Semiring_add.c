//------------------------------------------------------------------------------
// GxB_Semiring_add: return the additive monoid of a semiring
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Semiring_add           // return the additive monoid of a semiring
(
    GrB_Monoid *add,                // returns additive monoid of the semiring
    GrB_Semiring semiring           // semiring to query
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Semiring_add (&add, semiring)") ;
    GB_RETURN_IF_NULL (add) ;
    GB_RETURN_IF_NULL_OR_FAULTY (semiring) ;
    ASSERT_SEMIRING_OK (semiring, "semiring for add", GB0) ;

    //--------------------------------------------------------------------------
    // return the ztype
    //--------------------------------------------------------------------------

    (*add) = semiring->add ;
    return (GrB_SUCCESS) ;
}

