//------------------------------------------------------------------------------
// GrB_Vector_resize: change the size of a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GrB_Vector_resize      // change the size of a vector
(
    GrB_Vector w,               // vector to modify
    GrB_Index nrows_new         // new number of rows in vector
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE (w, "GrB_Vector_resize (w, nrows_new)") ;
    GB_BURBLE_START ("GrB_Vector_resize") ;
    GB_RETURN_IF_NULL_OR_FAULTY (w) ;

    //--------------------------------------------------------------------------
    // resize the vector
    //--------------------------------------------------------------------------

    GrB_Info info = GB_resize ((GrB_Matrix) w, nrows_new, 1, Werk) ;
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
    GrB_Index nrows_new         // new number of rows in vector
)
{ 
    return (GrB_Vector_resize (u, nrows_new)) ;
}

