//------------------------------------------------------------------------------
// GB_jit__user_op__0__my_rdiv.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.1.0, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

#define GB_USER_OP_FUNCTION my_rdiv
void my_rdiv (double *z, const double *x, const double *y)
;

void my_rdiv (double *z, const double *x, const double *y)
{
    // escape this quote: "
    /* escape this backslash \ */
    /* revised for GrB 10.1.0 */
    (*z) = (*y) / (*x) ;
}
#define GB_my_rdiv_USER_DEFN \
"void my_rdiv (double *z, const double *x, const double *y)\n" \
"{\n" \
"    // escape this quote: \"\n" \
"    /* escape this backslash \\ */\n" \
"    /* revised for GrB 10.1.0 */\n" \
"    (*z) = (*y) / (*x) ;\n" \
"}"
#define GB_USER_OP_DEFN GB_my_rdiv_USER_DEFN
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__user_op__0__my_rdiv
#define GB_jit_query  GB_jit__user_op__0__my_rdiv_query
#endif
#include "template/GB_jit_kernel_user_op.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x8c5b014b88b8adfb ;
    v [0] = 10 ; v [1] = 1 ; v [2] = 0 ;
    defn [0] = GB_my_rdiv_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
