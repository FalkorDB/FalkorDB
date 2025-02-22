//------------------------------------------------------------------------------
// GraphBLAS/Config/GB_prejit.c: return list of PreJIT kernels
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This file is configured by cmake from Config/GB_prejit.c.in, which has
// indexed the following 9 kernels in GraphBLAS/PreJIT:

#include "GB.h"
#include "jitifyer/GB_jitifyer.h"
#include "jit_kernels/include/GB_jit_kernel_proto.h"
#include "include/GB_pedantic_disable.h"

//------------------------------------------------------------------------------
// prototypes for all PreJIT kernels
//------------------------------------------------------------------------------

JIT_DOT2 (GB_jit__AxB_dot2__0004000bba0bbac7__plus_my_rdiv2)
JIT_DOT2 (GB_jit__AxB_dot2__0004000bba0bbacf__plus_my_rdiv2)
JIT_DOT2 (GB_jit__AxB_dot2__0004000bbb0bbbcd__plus_my_rdiv)
JIT_DOT2 (GB_jit__AxB_dot2__0004015bbb0bbbcd)
JIT_DOT2 (GB_jit__AxB_dot2__0004100bba0baacf__plus_my_rdiv2)
JIT_DOT2 (GB_jit__AxB_dot2__0004100bba0babcd__plus_my_rdiv2)
JIT_DOT2 (GB_jit__AxB_dot2__0004100bba0babcf__plus_my_rdiv2)
JIT_DOT2 (GB_jit__AxB_dot2__0004100bba0bbac7__plus_my_rdiv2)
JIT_UOP  (GB_jit__user_op__0__my_rdiv)


//------------------------------------------------------------------------------
// prototypes for all PreJIT query kernels
//------------------------------------------------------------------------------

JIT_Q (GB_jit__AxB_dot2__0004000bba0bbac7__plus_my_rdiv2_query)
JIT_Q (GB_jit__AxB_dot2__0004000bba0bbacf__plus_my_rdiv2_query)
JIT_Q (GB_jit__AxB_dot2__0004000bbb0bbbcd__plus_my_rdiv_query)
JIT_Q (GB_jit__AxB_dot2__0004015bbb0bbbcd_query)
JIT_Q (GB_jit__AxB_dot2__0004100bba0baacf__plus_my_rdiv2_query)
JIT_Q (GB_jit__AxB_dot2__0004100bba0babcd__plus_my_rdiv2_query)
JIT_Q (GB_jit__AxB_dot2__0004100bba0babcf__plus_my_rdiv2_query)
JIT_Q (GB_jit__AxB_dot2__0004100bba0bbac7__plus_my_rdiv2_query)
JIT_Q (GB_jit__user_op__0__my_rdiv_query)


//------------------------------------------------------------------------------
// GB_prejit_kernels: a list of function pointers to PreJIT kernels
//------------------------------------------------------------------------------

#if ( 9 > 0 )
static void *GB_prejit_kernels [9] =
{
GB_jit__AxB_dot2__0004000bba0bbac7__plus_my_rdiv2,
GB_jit__AxB_dot2__0004000bba0bbacf__plus_my_rdiv2,
GB_jit__AxB_dot2__0004000bbb0bbbcd__plus_my_rdiv,
GB_jit__AxB_dot2__0004015bbb0bbbcd,
GB_jit__AxB_dot2__0004100bba0baacf__plus_my_rdiv2,
GB_jit__AxB_dot2__0004100bba0babcd__plus_my_rdiv2,
GB_jit__AxB_dot2__0004100bba0babcf__plus_my_rdiv2,
GB_jit__AxB_dot2__0004100bba0bbac7__plus_my_rdiv2,
GB_jit__user_op__0__my_rdiv
} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit_queries: a list of function pointers to PreJIT query kernels
//------------------------------------------------------------------------------

#if ( 9 > 0 )
static void *GB_prejit_queries [9] =
{
GB_jit__AxB_dot2__0004000bba0bbac7__plus_my_rdiv2_query,
GB_jit__AxB_dot2__0004000bba0bbacf__plus_my_rdiv2_query,
GB_jit__AxB_dot2__0004000bbb0bbbcd__plus_my_rdiv_query,
GB_jit__AxB_dot2__0004015bbb0bbbcd_query,
GB_jit__AxB_dot2__0004100bba0baacf__plus_my_rdiv2_query,
GB_jit__AxB_dot2__0004100bba0babcd__plus_my_rdiv2_query,
GB_jit__AxB_dot2__0004100bba0babcf__plus_my_rdiv2_query,
GB_jit__AxB_dot2__0004100bba0bbac7__plus_my_rdiv2_query,
GB_jit__user_op__0__my_rdiv_query
} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit_names: a list of names of PreJIT kernels
//------------------------------------------------------------------------------

#if ( 9 > 0 )
static char *GB_prejit_names [9] =
{
"GB_jit__AxB_dot2__0004000bba0bbac7__plus_my_rdiv2",
"GB_jit__AxB_dot2__0004000bba0bbacf__plus_my_rdiv2",
"GB_jit__AxB_dot2__0004000bbb0bbbcd__plus_my_rdiv",
"GB_jit__AxB_dot2__0004015bbb0bbbcd",
"GB_jit__AxB_dot2__0004100bba0baacf__plus_my_rdiv2",
"GB_jit__AxB_dot2__0004100bba0babcd__plus_my_rdiv2",
"GB_jit__AxB_dot2__0004100bba0babcf__plus_my_rdiv2",
"GB_jit__AxB_dot2__0004100bba0bbac7__plus_my_rdiv2",
"GB_jit__user_op__0__my_rdiv"
} ;
#endif

//------------------------------------------------------------------------------
// GB_prejit: return list of PreJIT function pointers and function names
//------------------------------------------------------------------------------

void GB_prejit
(
    int32_t *nkernels,      // return # of kernels
    void ***Kernel_handle,  // return list of function pointers to kernels
    void ***Query_handle,   // return list of function pointers to queries
    char ***Name_handle     // return list of kernel names
)
{
    (*nkernels) = 9 ;
    #if ( 9 == 0 )
    (*Kernel_handle) = NULL ;
    (*Query_handle) = NULL ;
    (*Name_handle) = NULL ;
    #else
    (*Kernel_handle) = GB_prejit_kernels ;
    (*Query_handle) = GB_prejit_queries ;
    (*Name_handle) = GB_prejit_names ;
    #endif
}

