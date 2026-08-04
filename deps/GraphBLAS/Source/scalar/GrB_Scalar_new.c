//------------------------------------------------------------------------------
// GrB_Scalar_new: create a new GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The new GrB_Scalar has no entry.  Internally, it is identical to a
// GrB_Vector of length 1.  If this method fails, *s is set to NULL.

// The scalar is allocated in the arenas determined by the current Context.

#include "GB.h"

GrB_Info GrB_Scalar_new     // create a new GrB_Scalar with no entries
(
    GrB_Scalar *s,          // handle of GrB_Scalar to create
    GrB_Type type           // type of GrB_Scalar to create
)
{ 
    int header_arena = GB_Context_header_arena ( ) ;
    int data_arena = GB_Context_data_arena ( ) ;
    return (GxB_Scalar_new_arena (s, type, header_arena, data_arena)) ;
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

