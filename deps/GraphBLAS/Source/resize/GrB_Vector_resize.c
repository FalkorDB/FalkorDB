//------------------------------------------------------------------------------
// GrB_Vector_resize: change the size of a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "resize/GB_resize.h"

GrB_Info GrB_Vector_resize      // change the size of a vector
(
    GrB_Vector w,               // vector to modify
    uint64_t nrows_new          // new number of rows in vector
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (w) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (w) ;
    GB_WHERE1 (w, "GrB_Vector_resize (w, nrows_new)") ;
    GB_BURBLE_START ("GrB_Vector_resize") ;

    //--------------------------------------------------------------------------
    // resize the vector
    //--------------------------------------------------------------------------

    info = GB_resize ((GrB_Matrix) w, nrows_new, 1, Werk) ;
    GB_BURBLE_END ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GxB_Vector_resize: historical
//------------------------------------------------------------------------------

// This function now appears in the C API Specification as GrB_Vector_resize.
// The new name is preferred.  The old name will be kept for historical
// compatibility.

GrB_Info GxB_Vector_resize      // change the size of a vector
(
    GrB_Vector u,               // vector to modify
    uint64_t nrows_new          // new number of rows in vector
)
{ 
    return (GrB_Vector_resize (u, nrows_new)) ;
}

