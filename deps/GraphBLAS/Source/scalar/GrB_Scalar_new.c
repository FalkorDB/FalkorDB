//------------------------------------------------------------------------------
// GrB_Scalar_new: create a new GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The new GrB_Scalar has no entry.  Internally, it is identical to a
// GrB_Vector of length 1.  If this method fails, *s is set to NULL.

#include "GB.h"

GrB_Info GrB_Scalar_new     // create a new GrB_Scalar with no entries
(
    GrB_Scalar *s,          // handle of GrB_Scalar to create
    GrB_Type type           // type of GrB_Scalar to create
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Scalar_new (&s, type)") ;
    GB_RETURN_IF_NULL (s) ;
    (*s) = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;

    //--------------------------------------------------------------------------
    // create the GrB_Scalar
    //--------------------------------------------------------------------------

    GrB_Info info ;

    info = GB_new ((GrB_Matrix *) s, // new user header
        type, 1, 1, GB_Ap_calloc, true, GxB_SPARSE,
        GB_Global_hyper_switch_get ( ), 1) ;
    ASSERT (GB_IMPLIES (info == GrB_SUCCESS, GB_SCALAR_OK (*s))) ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GxB_Scalar_new: create a new GrB_Scalar (historical)
//------------------------------------------------------------------------------

GrB_Info GxB_Scalar_new     // create a new GrB_Scalar with no entries
(
    GrB_Scalar *s,          // handle of GrB_Scalar to create
    GrB_Type type           // type of GrB_Scalar to create
)
{
    return (GrB_Scalar_new (s, type)) ;
}

