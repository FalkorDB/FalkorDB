//------------------------------------------------------------------------------
// GrB_getVersion: get the version number of the GraphBLAS C API standard
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// For compile-time access, use GRB_VERSION and GRB_SUBVERSION.

#include "GB.h"

GrB_Info GrB_getVersion         // runtime access to C API version number
(
    unsigned int *version,      // returns GRB_VERSION
    unsigned int *subversion    // returns GRB_SUBVERSION
)
{ 

    //--------------------------------------------------------------------------
    // get the version number
    //--------------------------------------------------------------------------

    if (version    != NULL) (*version   ) = GRB_VERSION ;
    if (subversion != NULL) (*subversion) = GRB_SUBVERSION ;

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

