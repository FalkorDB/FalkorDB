//------------------------------------------------------------------------------
// GrB_finalize: finalize GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_finalize must be called as the last GraphBLAS function, per the
// GraphBLAS C API Specification.  Only one user thread can call this function.
// Results are undefined if more than one thread calls this function at the
// same time.

#include "GB.h"
#include "GB_jitifyer.h"

GrB_Info GrB_finalize ( )
{ 
    GB_jitifyer_finalize ( ) ;
    return (GrB_SUCCESS) ;
}

