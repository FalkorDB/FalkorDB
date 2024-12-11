//------------------------------------------------------------------------------
// GrB_Descriptor_wait: wait for a user-defined GrB_Descriptor to complete
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// In SuiteSparse:GraphBLAS, a user-defined GrB_Descriptor has no pending
// operations to wait for.  All this method does is verify that the descriptor
// is properly initialized, and then it does an OpenMP flush.  Note that unlike
// other methods, passing in a NULL pointer, or a pointer to a NULL descriptor
// is valid, since a NULL descriptor results in default settings.

#include "GB.h"

GrB_Info GrB_Descriptor_wait // no work, just check if GrB_Descriptor is valid
(
    GrB_Descriptor desc,
    GrB_WaitMode waitmode
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Descriptor_wait (desc, waitmode)") ;
    if (desc != NULL) GB_RETURN_IF_FAULTY (desc) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

