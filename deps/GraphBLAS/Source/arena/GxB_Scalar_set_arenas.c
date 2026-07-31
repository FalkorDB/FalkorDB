//------------------------------------------------------------------------------
// GxB_Scalar_set_arenas: set the arenas (header and data) of a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

GrB_Info GxB_Scalar_set_arenas
(
    // input/output
    GrB_Scalar *Shandle,        // handle of vector to modify
    // input
    const int new_header_arena, // new arena for the header of S
    const int new_data_arena    // new arena for the data content of S
)
{ 
    return (GxB_Matrix_set_arenas ((GrB_Matrix *) Shandle,
        new_header_arena, new_data_arena)) ;
}

