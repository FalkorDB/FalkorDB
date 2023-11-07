//------------------------------------------------------------------------------
// GB_JITpackage.h: definitions to package GraphBLAS source code for the JIT
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef GBMATLAB
#include "GB_rename.h"
#endif

typedef struct
{
    size_t uncompressed_size ;  // size of the uncompressed file
    size_t compressed_size ;    // size of the compressed blob
    uint8_t *blob ;             // the compressed blob
    char *filename ;            // filename, including the suffix
}
GB_JITpackage_index_struct ;

extern int GB_JITpackage_nfiles ;
extern GB_JITpackage_index_struct GB_JITpackage_index [ ] ;

