//------------------------------------------------------------------------------
// GB_Vector_check: print a GraphBLAS GrB_Vector and check if it is valid
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_Vector: same as GrB_Matrix, except it has exactly one column

#include "GB.h"

GrB_Info GB_Vector_check    // check a GraphBLAS vector
(
    const GrB_Vector v,     // GraphBLAS vector to print and check
    const char *name,       // name of the vector
    int pr,                 // 0: print nothing, 1: print header and errors,
                            // 2: print brief, 3: print all
    FILE *f                 // file for output
)
{

    //--------------------------------------------------------------------------
    // check GrB_Matrix conditions
    //--------------------------------------------------------------------------

    GrB_Info info = GB_matvec_check ((GrB_Matrix) v, name, pr, f, "vector") ;
    if (info != GrB_SUCCESS)
    { 
        // GrB_Matrix form is invalid already
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // check GrB_Vector specific conditions
    //--------------------------------------------------------------------------

    if (!GB_VECTOR_OK (v))
    { 
        GBPR0 ("    GrB_Vector is invalid [%s]\n", name) ;
        return (GrB_INVALID_OBJECT) ;
    }

    return (GrB_SUCCESS) ;
}

