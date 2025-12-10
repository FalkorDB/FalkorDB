//------------------------------------------------------------------------------
// GB_JITpackage.h: definitions to package GraphBLAS source code for the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#undef GRAPHBLAS_VANILLA
#include "GraphBLAS.h"

typedef struct
{
    size_t uncompressed_size ;  // size of the uncompressed file
    size_t compressed_size ;    // size of the compressed blob
    uint8_t *blob ;             // the compressed blob
    char *filename ;            // filename, including the suffix
}
GB_JITpackage_index_struct ;

int GB_JITpackage_nfiles_get (void) ;
void *GB_JITpackage_index_get (void) ;

