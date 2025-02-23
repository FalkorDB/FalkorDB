//------------------------------------------------------------------------------
// GxB_unload_Matrix_into_Container: unload a GrB_Matrix into a Container
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A is returned as a 0-by-0 matrix in full data format, with no content.

#include "GB_container.h"

GrB_Info GxB_unload_Matrix_into_Container   // GrB_Matrix -> GxB_Container
(
    GrB_Matrix A,               // matrix to unload into the Container
    GxB_Container Container,    // Container to hold the contents of A
    const GrB_Descriptor desc   // currently unused
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL (Container) ;
    GB_WHERE_1 (A, "GxB_Matrix_unload_into_Container") ;

    //--------------------------------------------------------------------------
    // unload the matrix into the container
    //--------------------------------------------------------------------------

    return (GB_unload_into_container (A, Container, Werk)) ;
}

