//------------------------------------------------------------------------------
// GrB_Vector_free: free a sparse vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// free all the content of a vector.  After GrB_Vector_free (&v), v is set
// to NULL.  The vector may have readonly content; it is simply removed from s
// and not modified.

#include "GB.h"

GrB_Info GrB_Vector_free    // free a vector
(
    GrB_Vector *v           // handle of vector to free
)
{ 

    GB_Matrix_free ((GrB_Matrix *) v) ;
    return (GrB_SUCCESS) ;
}

