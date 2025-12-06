//------------------------------------------------------------------------------
// GB_jit__build__f9066668.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.0, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: (second, int32_t)

// binary dup operator types:
#define GB_Z_TYPE int32_t
#define GB_X_TYPE int32_t
#define GB_Y_TYPE int32_t

// Sx and Tx data types:
#define GB_Tx_TYPE int32_t
#define GB_Sx_TYPE int64_t

// binary dup operator:
#define GB_DUP(z,x,y) z = y
#define GB_UPDATE(z,y) z = y

// build copy/dup methods:
#define GB_BLD_COPY(Tx,p,Sx,k) Tx [p] = (int32_t) Sx [k]
#define GB_BLD_DUP(Tx,p,Sx,k) \
    int32_t y = (int32_t) Sx [k] ; \
    int32_t x = Tx [p] ; \
    int32_t z ; \
    GB_DUP (z, x, y) ; \
    Tx [p] = z ;

// 32/64 integer types:
#define GB_Ti_TYPE int32_t
#define GB_Ti_BITS 32
#define GB_I_TYPE  uint32_t
#define GB_K_TYPE  uint32_t
#define GB_K_WORK(k) k
#define GB_K_IS_NULL 1
#define GB_NO_DUPLICATES 1

#include "include/GB_kernel_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__build__f9066668
#define GB_jit_query  GB_jit__build__f9066668_query
#endif
#include "template/GB_jit_kernel_build.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x9fad276fbe37a3a3 ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 0 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
