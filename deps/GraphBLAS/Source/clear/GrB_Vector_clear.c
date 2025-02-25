//------------------------------------------------------------------------------
// GrB_Vector_clear: clears the content of a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// See GrB_Matrix_clear for details.

#include "GB.h"

GrB_Info GrB_Vector_clear   // clear a vector of all entries;
(                           // type and dimension remain unchanged
    GrB_Vector v            // vector to clear
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (v) ;
    GB_WHERE_1 (v, "GrB_Vector_clear (v)") ;

    ASSERT (GB_VECTOR_OK (v)) ;

    //--------------------------------------------------------------------------
    // clear the vector
    //--------------------------------------------------------------------------

    return (GB_clear ((GrB_Matrix) v, Werk)) ;
}

