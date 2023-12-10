//------------------------------------------------------------------------------
// GrB_Scalar_free: free a sparse GrB_Scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// free all the content of a GrB_Scalar.  After GrB_Scalar_free (&s), s is set
// to NULL

#include "GB.h"

GrB_Info GrB_Scalar_free    // free a GrB_Scalar
(
    GrB_Scalar *s           // handle of GrB_Scalar to free
)
{ 
    GB_Matrix_free ((GrB_Matrix *) s) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Scalar_free: free a sparse GrB_Scalar (historical)
//------------------------------------------------------------------------------

GrB_Info GxB_Scalar_free    // free a GrB_Scalar
(
    GrB_Scalar *s           // handle of GrB_Scalar to free
)
{
    return (GrB_Scalar_free (s)) ;
}

