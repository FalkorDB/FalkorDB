//------------------------------------------------------------------------------
// GrB_Scalar_new: create a new GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The new GrB_Scalar has no entry.  Internally, it is identical to a
// GrB_Vector of length 1.  If this method fails, *s is set to NULL.

#include "GB.h"

#define GB_FREE_ALL GB_Matrix_free ((GrB_Matrix *) s)

GrB_Info GrB_Scalar_new     // create a new GrB_Scalar with no entries
(
    GrB_Scalar *s,          // handle of GrB_Scalar to create
    GrB_Type type           // type of GrB_Scalar to create
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (s) ;
    (*s) = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;

    //--------------------------------------------------------------------------
    // create the GrB_Scalar
    //--------------------------------------------------------------------------

    // determine the p_is_32, j_is_32, and i_is_32 settings for the new scalar
    bool Sp_is_32, Sj_is_32, Si_is_32 ;
    GB_determine_pji_is_32 (&Sp_is_32, &Sj_is_32, &Si_is_32,
        GxB_SPARSE, 1, 1, 1, NULL) ;

    GB_OK (GB_new ((GrB_Matrix *) s, // new user header
        type, 1, 1, GB_ph_calloc, true, GxB_SPARSE,
        GB_Global_hyper_switch_get ( ), 1, Sp_is_32, Sj_is_32, Si_is_32)) ;

    return (GrB_SUCCESS) ;
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

