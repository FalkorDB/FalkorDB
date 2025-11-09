//------------------------------------------------------------------------------
// GrB_Monoid_set_*: set a field in a monoid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Monoid_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_set_Scalar
(
    GrB_Monoid monoid,
    GrB_Scalar scalar,
    int field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GrB_Monoid_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_set_String
( 
    GrB_Monoid monoid,
    char * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_FAULTY (monoid) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_MONOID_OK (monoid, "monoid to get option", GB0) ;

    if (monoid->header_size == 0 || field != GrB_NAME)
    { 
        // built-in monoids may not be modified
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_user_name_set (&(monoid->user_name),
        &(monoid->user_name_size), value, true)) ;
}

//------------------------------------------------------------------------------
// GrB_Monoid_set_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_set_INT32
(
    GrB_Monoid monoid,
    int32_t value,
    int field
)
{ 
    return (GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GrB_Monoid_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Monoid_set_VOID
(
    GrB_Monoid monoid,
    void * value,
    int field,
    size_t size
)
{ 
    return (GrB_INVALID_VALUE) ;
}

