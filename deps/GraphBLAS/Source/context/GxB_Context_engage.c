//------------------------------------------------------------------------------
// GxB_Context_engage: engage a Context for this user thread
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GxB_Context_engage sets the provided Context object as the Context for this
// user thread.  Multiple user threads can share a single Context.  Any prior
// Context for this user thread is superseded by the new Context (the prior one
// is not freed).  GrB_SUCCESS is returned, and future calls to GraphBLAS by
// this user thread will use the provided Context.

// If the Context on input is the GxB_CONTEXT_WORLD object, then the current
// Context is disengaged.  That is, the following calls have the same effect,
// setting the Context of this user thread to GxB_CONTEXT_WORLD:
//
//      GxB_Context_engage (GxB_CONTEXT_WORLD) ;
//      GxB_Context_disengage (NULL) ;
//
// The result for both cases above is GrB_SUCCESS.

// Error cases:
// If Context is NULL on input, GrB_NULL_POINTER is returned.
// If a non-NULL Context is provided but it is faulty in some way, then an
// error code is returned (GrB_INVALID_OBJECT or GrB_UNINITIALIZED_OBJECT).
// If an error code is returned, the current Context for this user thread is
// unmodified.

#include "GB.h"

GrB_Info GxB_Context_engage         // engage a Context
(
    GxB_Context Context             // Context to engage
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (Context) ;

    //--------------------------------------------------------------------------
    // engage the Context
    //--------------------------------------------------------------------------

    return (GB_Context_engage (Context)) ;
}

