//------------------------------------------------------------------------------
// GB_jit__apply_unop__00400099099f__LG_rand_next_f2.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.0.2, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: LG_rand_next_f2, ztype: uint64_t, xtype: uint64_t, ytype: void

// unary operator types:
#define GB_Z_TYPE uint64_t
#define GB_X_TYPE uint64_t
#define GB_Y_TYPE void
#define GB_DECLAREZ(zwork) uint64_t zwork
#define GB_DECLAREX(xwork) uint64_t xwork
#define GB_DECLAREY(ywork) void ywork

// unary operator:
#ifndef GB_GUARD_LG_rand_next_f2_DEFINED
#define GB_GUARD_LG_rand_next_f2_DEFINED
GB_STATIC_INLINE
void LG_rand_next_f2 (uint64_t *z, const uint64_t *x)  
{                                                      
    uint64_t state = (*x) ;                            
    state ^= state << 13 ;                             
    state ^= state >> 7 ;                              
    state ^= state << 17 ;                             
    (*z) = state ;                                     
}
#define GB_LG_rand_next_f2_USER_DEFN \
"void LG_rand_next_f2 (uint64_t *z, const uint64_t *x)  \n" \
"{                                                      \n" \
"    uint64_t state = (*x) ;                            \n" \
"    state ^= state << 13 ;                             \n" \
"    state ^= state >> 7 ;                              \n" \
"    state ^= state << 17 ;                             \n" \
"    (*z) = state ;                                     \n" \
"}"
#endif
#define GB_UNARYOP(z,x,i,j,y)  LG_rand_next_f2 (&(z), &(x))
#define GB_DEPENDS_ON_X 1
#define GB_DEPENDS_ON_Y 0
#define GB_DEPENDS_ON_I 0
#define GB_DEPENDS_ON_J 0
#define GB_UNOP(Cx,pC,Ax,pA,A_iso,i,j,y) GB_UNARYOP (Cx [pC], Ax [pA],  ,  ,  )

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
#define GB_jit_kernel GB_jit__apply_unop__00400099099f__LG_rand_next_f2
#define GB_jit_query  GB_jit__apply_unop__00400099099f__LG_rand_next_f2_query
#endif
#include "template/GB_jit_kernel_apply_unop.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x0360f85b78f6d941 ;
    v [0] = 10 ; v [1] = 0 ; v [2] = 2 ;
    defn [0] = GB_LG_rand_next_f2_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
