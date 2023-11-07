//------------------------------------------------------------------------------
// GB_copy_user_user.c: copy a user type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"

void GB_copy_user_user (void *z, const void *x, size_t s)
{ 
    memcpy (z, x, s) ;
}

