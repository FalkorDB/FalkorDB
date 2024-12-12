//------------------------------------------------------------------------------
// GrB_Descriptor_new: create a new descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Default values are set to GxB_DEFAULT

#include "GB.h"

GrB_Info GrB_Descriptor_new     // create a new descriptor
(
    GrB_Descriptor *descriptor  // handle of descriptor to create
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GrB_Descriptor_new (&descriptor)") ;
    GB_RETURN_IF_NULL (descriptor) ;
    (*descriptor) = NULL ;

    //--------------------------------------------------------------------------
    // create the descriptor
    //--------------------------------------------------------------------------

    // allocate the descriptor
    size_t header_size ;
    (*descriptor) = GB_MALLOC (1, struct GB_Descriptor_opaque, &header_size) ;
    if (*descriptor == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    // initialize the descriptor
    GrB_Descriptor desc = *descriptor ;
    desc->magic = GB_MAGIC ;
    desc->header_size = header_size ;
    desc->user_name = NULL ;       // user_name for GrB_get/GrB_set
    desc->user_name_size = 0 ;
    desc->logger = NULL ;          // error string
    desc->logger_size = 0 ;
    desc->out  = GxB_DEFAULT ;     // descriptor for output
    desc->mask = GxB_DEFAULT ;     // descriptor for the mask input
    desc->in0  = GxB_DEFAULT ;     // descriptor for the first input
    desc->in1  = GxB_DEFAULT ;     // descriptor for the second input
    desc->axb  = GxB_DEFAULT ;     // descriptor for selecting the C=A*B method
    desc->do_sort = false ;        // do not sort in GrB_mxm and others
    desc->compression = GxB_DEFAULT ;
    desc->import = GxB_DEFAULT ;   // trust input data in GxB import/deserialize
    return (GrB_SUCCESS) ;
}

