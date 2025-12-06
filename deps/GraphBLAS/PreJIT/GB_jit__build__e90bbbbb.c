//------------------------------------------------------------------------------
// GB_jit__build__e90bbbbb.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.0, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: (second, double)

// binary dup operator types:
#define GB_Z_TYPE double
#define GB_X_TYPE double
#define GB_Y_TYPE double

// Sx and Tx data types:
#define GB_Tx_TYPE double
#define GB_Sx_TYPE double

// binary dup operator:
#define GB_DUP(z,x,y) z = y
#define GB_UPDATE(z,y) z = y

// build copy/dup methods:
#define GB_BLD_COPY(Tx,p,Sx,k) Tx [p] = Sx [k]
#define GB_BLD_DUP(Tx,p,Sx,k) GB_UPDATE (Tx [p], Sx [k])

// 32/64 integer types:
#define GB_Ti_TYPE int32_t
#define GB_Ti_BITS 32
#define GB_I_TYPE  uint32_t
#define GB_K_TYPE  uint32_t
#define GB_K_WORK(k) K_work [k]
#define GB_K_IS_NULL 0
#define GB_NO_DUPLICATES 1

#include "include/GB_kernel_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__build__e90bbbbb
#define GB_jit_query  GB_jit__build__e90bbbbb_query
#endif
#include "template/GB_jit_kernel_build.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x2a1fb0f5bd39a6f1 ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 0 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
