//------------------------------------------------------------------------------
// GB_jitifyer_get_function.c: get a function pointer from a (void *)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "jitifyer/GB_jitifyer.h"
#include "include/GB_pedantic_disable.h"

GB_jit_query_func GB_jitifyer_get_query (void *p)
{
    return ((GB_jit_query_func) p) ;
}

GB_user_op_f GB_jitifyer_get_user_op (void *p)
{
    return ((GB_user_op_f) p) ;
}

GB_user_type_f GB_jitifyer_get_user_type (void *p)
{
    return ((GB_user_type_f) p) ;
}

