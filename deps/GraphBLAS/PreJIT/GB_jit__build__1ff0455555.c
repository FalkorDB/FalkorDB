//------------------------------------------------------------------------------
// GB_jit__build__1ff0455555.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.4.0, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: (plus, uint16_t)

// binary dup operator types:
#define GB_Z_TYPE uint16_t
#define GB_X_TYPE uint16_t
#define GB_Y_TYPE uint16_t

// Sx and Tx data types:
#define GB_Tx_TYPE uint16_t
#define GB_Sx_TYPE uint16_t

// binary dup operator:
#define GB_DUP(z,x,y) z = (x) + (y)
#define GB_UPDATE(z,y) z += y

// build copy/dup methods:
#define GB_BLD_SXTYPE_IS_TXTYPE 1
#define GB_BLD_NO_CASTING 1
#define GB_BLD_COPY(Tx,p,Sx,k) Tx [p] = Sx [k]
#define GB_BLD_DUP(Tx,p,Sx,k) GB_UPDATE (Tx [p], Sx [k])

// type of build:
#define GB_MTX_BUILD 1
#define GB_ISO_BUILD 0
#define GB_KNOWN_NO_DUPLICATES 0
#define GB_KNOWN_SORTED 0

// 32/64 integer types:
#define GB_Tp_TYPE int32_t
#define GB_Tp_BITS 32
#define GB_Tj_TYPE int32_t
#define GB_Tj_BITS 32
#define GB_Ti_TYPE int32_t
#define GB_Ti_BITS 32
#define GB_I_TYPE  uint32_t
#define GB_J_TYPE  uint32_t
#define GB_K_TYPE  uint32_t
#define GB_K_WORK(k) k
#define GB_K_IS_NULL 1
#define GB_KEY_PRELOADED 0
#define GB_KEY_TYPE uint32_t
#define GB_KEY_BITS 32

#include "include/GB_kernel_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__build__1ff0455555
#define GB_jit_query  GB_jit__build__1ff0455555_query
#endif
#include "template/GB_jit_kernel_build.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x3735730f67eb6a5c ;
    v [0] = 10 ; v [1] = 4 ; v [2] = 0 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
