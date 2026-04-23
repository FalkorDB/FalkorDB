//------------------------------------------------------------------------------
// GB_ph_code.h: definitions for basic methods for the GrB_Matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_PH_CODE_H
#define GB_PH_CODE_H

typedef enum                    // input parameter to GB_new and GB_new_bix
{
    GB_ph_calloc,               // 0: calloc A->p, malloc A->h if hypersparse
    GB_ph_malloc,               // 1: malloc A->p, malloc A->h if hypersparse
    GB_ph_null                  // 2: do not allocate A->p or A->h
}
GB_ph_code ;

#endif

