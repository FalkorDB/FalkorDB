//------------------------------------------------------------------------------
// GrB_Scalar_set_*: set a field in a scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

//------------------------------------------------------------------------------
// GrB_Scalar_set_Scalar
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_set_Scalar
(
    GrB_Scalar s,
    GrB_Scalar scalar,
    int field
)
{ 
    // all settings are ignored
    return ((field == GrB_STORAGE_ORIENTATION_HINT) ?
        GrB_SUCCESS : GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GrB_Scalar_set_String
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_set_String
(
    GrB_Scalar s,
    char * value,
    int field
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL_OR_INVALID (s) ;
    GB_RETURN_IF_NULL (value) ;
    ASSERT_SCALAR_OK (s, "s to set option", GB0) ;

    //--------------------------------------------------------------------------
    // set the field
    //--------------------------------------------------------------------------

    return (GB_matvec_name_set ((GrB_Matrix) s, value, field)) ;
}

//------------------------------------------------------------------------------
// GrB_Scalar_set_INT32
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_set_INT32
(
    GrB_Scalar s,
    int32_t value,
    int field
)
{ 
    // all settings are ignored
    return ((field == GrB_STORAGE_ORIENTATION_HINT) ?
        GrB_SUCCESS : GrB_INVALID_VALUE) ;
}

//------------------------------------------------------------------------------
// GrB_Scalar_set_VOID
//------------------------------------------------------------------------------

GrB_Info GrB_Scalar_set_VOID
(
    GrB_Scalar s,
    void * value,
    int field,
    size_t size
)
{ 
    return (GrB_INVALID_VALUE) ;
}

