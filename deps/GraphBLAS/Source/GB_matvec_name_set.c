//------------------------------------------------------------------------------
// GB_matvec_name_set: set the user_name of a matrix/vector/scalar
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Info GB_matvec_name_set
(
    GrB_Matrix A,
    char *value,
    int field
)
{ 

    if (field != GrB_NAME)
    { 
        return (GrB_INVALID_VALUE) ;
    }

    return (GB_user_name_set (&(A->user_name), &(A->user_name_size), value,
        false)) ;
}

