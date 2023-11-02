//------------------------------------------------------------------------------
// GxB_Monoid_terminal: return the terminal of a monoid (if any)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Monoid_terminal        // return the monoid terminal
(
    bool *has_terminal,             // true if the monoid has a terminal value
    void *terminal,                 // returns the terminal of the monoid,
                                    // unmodified if has_terminal is false
    GrB_Monoid monoid               // monoid to query
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Monoid_terminal (&has_terminal, &terminal, monoid)") ;
    GB_RETURN_IF_NULL (has_terminal) ;
    GB_RETURN_IF_NULL (terminal) ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    ASSERT_MONOID_OK (monoid, "monoid for terminal", GB0) ;

    //--------------------------------------------------------------------------
    // return the terminal
    //--------------------------------------------------------------------------

    (*has_terminal) = (monoid->terminal != NULL) ;
    if (*has_terminal)
    { 
        memcpy (terminal, monoid->terminal, monoid->op->ztype->size) ;
    }
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

