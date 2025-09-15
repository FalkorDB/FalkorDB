//------------------------------------------------------------------------------
// GB_jit__build__e9077777.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.0.2, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: (second, uint32_t)

// binary dup operator types:
#define GB_Z_TYPE uint32_t
#define GB_X_TYPE uint32_t
#define GB_Y_TYPE uint32_t

// Sx and Tx data types:
#define GB_Tx_TYPE uint32_t
#define GB_Sx_TYPE uint32_t

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
#define GB_jit_kernel GB_jit__build__e9077777
#define GB_jit_query  GB_jit__build__e9077777_query
#endif
#include "template/GB_jit_kernel_build.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x49b2d031b37103e9 ;
    v [0] = 10 ; v [1] = 0 ; v [2] = 2 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
