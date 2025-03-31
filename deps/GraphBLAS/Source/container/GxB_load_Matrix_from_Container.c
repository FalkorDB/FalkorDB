//------------------------------------------------------------------------------
// GxB_load_Matrix_from_Container: load a GrB_Matrix from a Container
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The matrix A may have readonly components on input; they are simply removed
// from A and not modified.

#include "GB_container.h"

GrB_Info GxB_load_Matrix_from_Container     // GrB_Matrix <- GxB_Container
(
    GrB_Matrix A,               // matrix to load from the Container.  On input,
                                // A is a matrix of any size or type; on output
                                // any prior size, type, or contents is freed
                                // and overwritten with the Container.
    GxB_Container Container,    // Container with contents to load into A
    const GrB_Descriptor desc   // currently unused
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL_OR_FAULTY (A) ;
    GB_RETURN_IF_NULL (Container) ;
    GB_WHERE_1 (A, "GxB_load_Matrix_from_Container") ;

    //--------------------------------------------------------------------------
    // load the matrix from the container
    //--------------------------------------------------------------------------

    return (GB_load_from_container (A, Container, Werk)) ;
}

