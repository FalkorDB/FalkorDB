//------------------------------------------------------------------------------
// GB_is_nonzero.h: determine if a scalar is zero
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_IS_NONZERO_H
#define GB_IS_NONZERO_H

static inline bool GB_is_nonzero (const GB_void *value, int64_t size)
{ 
    for (int64_t i = 0 ; i < size ; i++)
    {
        if (value [i] != 0) return (true) ;
    }
    return (false) ;
}

#endif

