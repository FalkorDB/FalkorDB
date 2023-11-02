//------------------------------------------------------------------------------
// GxB_Context_disengage: disengage a Context for this user thread
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// If a NULL Context is provided or if the Context input parameter is
// GxB_CONTEXT_WORLD, then any current Context for this user thread is
// disengaged.  If a valid non-NULL Context is provided and it matches the
// current Context for this user thread, it is disengaged.  In all of these
// cases, GrB_SUCCESS is returned.  The user thread has no Context object and
// any subsequent calls to GraphBLAS functions will use the world Context,
// GxB_CONTEXT_WORLD.

// Error cases:
// If a non-NULL Context is provided but it is faulty in some way, then an
// error code is returned (GrB_INVALID_OBJECT or GrB_UNINITIALIZED_OBJECT).
// If a non-NULL Context is provided on input that doesn't match the current
// Context for this thread, then GrB_INVALID_VALUE is returned.
// If an error code is returned, the current Context for this user thread is
// unmodified.

#include "GB.h"

GrB_Info GxB_Context_disengage      // disengage a Context
(
    GxB_Context Context             // Context to disengage
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_FAULTY (Context) ;

    //--------------------------------------------------------------------------
    // disengage the Context
    //--------------------------------------------------------------------------

    return (GB_Context_disengage (Context)) ;
}

