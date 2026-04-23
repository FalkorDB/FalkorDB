//------------------------------------------------------------------------------
// GrB_Descriptor_new: create a new descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
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

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (descriptor) ;
    (*descriptor) = NULL ;

    int memlane = GB_Context_memlane ( ) ;
    uint64_t mem = GB_mem (memlane, 0) ;

    //--------------------------------------------------------------------------
    // create the descriptor
    //--------------------------------------------------------------------------

    // allocate the descriptor
    uint64_t header_mem = mem ;
    (*descriptor) = GB_MALLOC_MEMORY (1, sizeof (struct GB_Descriptor_opaque),
        &header_mem) ;
    if (*descriptor == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }

    // initialize the descriptor
    GrB_Descriptor desc = *descriptor ;
    desc->magic = GB_MAGIC ;
    desc->header_mem = header_mem ;
    // user_name for GrB_get/GrB_set:
    desc->user_name = NULL ; desc->user_name_mem = 0 ;
    // error string:
    desc->logger = NULL ; desc->logger_mem = 0 ;
    desc->out  = GxB_DEFAULT ;     // descriptor for output
    desc->mask = GxB_DEFAULT ;     // descriptor for the mask input
    desc->in0  = GxB_DEFAULT ;     // descriptor for the first input
    desc->in1  = GxB_DEFAULT ;     // descriptor for the second input
    desc->axb  = GxB_DEFAULT ;     // descriptor for selecting the C=A*B method
    desc->do_sort = false ;        // do not sort in GrB_mxm and others
    desc->compression = GxB_DEFAULT ;
    desc->import = GxB_DEFAULT ;   // trust input data in GxB import/deserialize
    desc->row_list = GxB_DEFAULT ; // use List->x of row index vector
    desc->col_list = GxB_DEFAULT ; // use List->x of col index vector
    desc->val_list = GxB_DEFAULT ; // use List->x of value vector
    return (GrB_SUCCESS) ;
}

