//------------------------------------------------------------------------------
// GB_jit__apply_unop__0046fe99999f__LG_rand_init_func.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.0.2, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: LG_rand_init_func, ztype: uint64_t, xtype: uint64_t, ytype: uint64_t

// unary operator types:
#define GB_Z_TYPE uint64_t
#define GB_X_TYPE uint64_t
#define GB_Y_TYPE uint64_t
#define GB_DECLAREZ(zwork) uint64_t zwork
#define GB_DECLAREX(xwork) uint64_t xwork
#define GB_DECLAREY(ywork) uint64_t ywork

// unary operator:
#ifndef GB_GUARD_LG_rand_init_func_DEFINED
#define GB_GUARD_LG_rand_init_func_DEFINED
GB_STATIC_INLINE
void LG_rand_init_func (uint64_t *z, const void *x,            
    GrB_Index i, GrB_Index j, const uint64_t *seed)            
{                                                              
   uint64_t state = i + (*seed) ;                              
   uint64_t result = (state += 0x9E3779B97F4A7C15LL) ;         
   result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9LL ; 
   result = (result ^ (result >> 27)) * 0x94D049BB133111EBLL ; 
   result = (result ^ (result >> 31)) ;                        
   (*z) = result ;                                             
}
#define GB_LG_rand_init_func_USER_DEFN \
"void LG_rand_init_func (uint64_t *z, const void *x,            \n" \
"    GrB_Index i, GrB_Index j, const uint64_t *seed)            \n" \
"{                                                              \n" \
"   uint64_t state = i + (*seed) ;                              \n" \
"   uint64_t result = (state += 0x9E3779B97F4A7C15LL) ;         \n" \
"   result = (result ^ (result >> 30)) * 0xBF58476D1CE4E5B9LL ; \n" \
"   result = (result ^ (result >> 27)) * 0x94D049BB133111EBLL ; \n" \
"   result = (result ^ (result >> 31)) ;                        \n" \
"   (*z) = result ;                                             \n" \
"}"
#endif
#define GB_UNARYOP(z,x,i,j,y) LG_rand_init_func (&(z), &(x), i, j, &(y))
#define GB_DEPENDS_ON_X 1
#define GB_DEPENDS_ON_Y 1
#define GB_DEPENDS_ON_I 1
#define GB_DEPENDS_ON_J 1
#define GB_UNOP(Cx,pC,Ax,pA,A_iso,i,j,y) GB_UNARYOP (Cx [pC], Ax [pA], i, j, y)

// C type:
#define GB_C_TYPE uint64_t
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64

// A matrix: full
#define GB_A_IS_HYPER  0
#define GB_A_IS_SPARSE 0
#define GB_A_IS_BITMAP 0
#define GB_A_IS_FULL   1
#define GBp_A(Ap,k,vlen) ((k) * (vlen))
#define GBh_A(Ah,k)      (k)
#define GBi_A(Ai,p,vlen) ((p) % (vlen))
#define GBb_A(Ab,p)      1
#define GB_A_NVALS(e) int64_t e = (A->vlen * A->vdim)
#define GB_A_NHELD(e) GB_A_NVALS(e)
#define GB_A_HAS_ZOMBIES 0
#define GB_A_ISO 0
#define GB_A_TYPE uint64_t
#define GB_A2TYPE uint64_t
#define GB_DECLAREA(a) uint64_t a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Ai_TYPE uint64_t
#define GB_Ai_SIGNED_TYPE int64_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 64
#define GB_Ai_BITS 64

#include "include/GB_kernel_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__apply_unop__0046fe99999f__LG_rand_init_func
#define GB_jit_query  GB_jit__apply_unop__0046fe99999f__LG_rand_init_func_query
#endif
#include "template/GB_jit_kernel_apply_unop.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xbb0a575dd2681dbf ;
    v [0] = 10 ; v [1] = 0 ; v [2] = 2 ;
    defn [0] = GB_LG_rand_init_func_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
