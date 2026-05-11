//------------------------------------------------------------------------------
// GB_jit__build__f909999b.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.1, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: (second, uint64_t)

// binary dup operator types:
#define GB_Z_TYPE uint64_t
#define GB_X_TYPE uint64_t
#define GB_Y_TYPE uint64_t

// Sx and Tx data types:
#define GB_Tx_TYPE uint64_t
#define GB_Sx_TYPE double

// binary dup operator:
#define GB_DUP(z,x,y) z = y
#define GB_UPDATE(z,y) z = y

// build copy/dup methods:
#ifndef GB_GUARD_GJ_cast_to_uint64_DEFINED
#define GB_GUARD_GJ_cast_to_uint64_DEFINED
GB_STATIC_INLINE
uint64_t GJ_cast_to_uint64 (double x)                      
{                                                          
    if (isnan (x) || x <= 0) return (0) ;                  
    if (x >= (double) UINT64_MAX) return (UINT64_MAX) ;    
    return ((uint64_t) x) ;                                
}
#endif
#ifndef GB_GUARD_GJ_cast_to_uint64_DEFINED
#define GB_GUARD_GJ_cast_to_uint64_DEFINED
GB_STATIC_INLINE
uint64_t GJ_cast_to_uint64 (double x)                      
{                                                          
    if (isnan (x) || x <= 0) return (0) ;                  
    if (x >= (double) UINT64_MAX) return (UINT64_MAX) ;    
    return ((uint64_t) x) ;                                
}
#endif
#define GB_BLD_COPY(Tx,p,Sx,k) Tx [p] = GJ_cast_to_uint64 (Sx [k])
#define GB_BLD_DUP(Tx,p,Sx,k) \
    uint64_t y = GJ_cast_to_uint64 (Sx [k]) ; \
    uint64_t x = Tx [p] ; \
    uint64_t z ; \
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
#define GB_jit_kernel GB_jit__build__f909999b
#define GB_jit_query  GB_jit__build__f909999b_query
#endif
#include "template/GB_jit_kernel_build.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x62760a5ed930225f ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 1 ;
    defn [0] = NULL ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
