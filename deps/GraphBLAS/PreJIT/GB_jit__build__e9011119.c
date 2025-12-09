//------------------------------------------------------------------------------
// GB_jit__build__e9011119.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.0, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: (second, bool)

// binary dup operator types:
#define GB_Z_TYPE bool
#define GB_X_TYPE bool
#define GB_Y_TYPE bool

// Sx and Tx data types:
#define GB_Tx_TYPE bool
#define GB_Sx_TYPE uint64_t

// binary dup operator:
#define GB_DUP(z,x,y) z = y
#define GB_UPDATE(z,y) z = y

// build copy/dup methods:
#define GB_BLD_COPY(Tx,p,Sx,k) Tx [p] = ((Sx [k]) != 0)
#define GB_BLD_DUP(Tx,p,Sx,k) \
    bool y = ((Sx [k]) != 0) ; \
    bool x = Tx [p] ; \
    bool z ; \
    GB_DUP (z, x, y) ; \
    Tx [p] = z ;

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
#define GB_jit_kernel GB_jit__build__e9011119
#define GB_jit_query  GB_jit__build__e9011119_query
#endif
#include "template/GB_jit_kernel_build.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xea6d0f11cfd8d892 ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 0 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
