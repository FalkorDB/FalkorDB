//------------------------------------------------------------------------------
// GB_jit__apply_bind2nd__000000999099fcc__LG_randmat_mod.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.0.2, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: LG_randmat_mod, ztype: uint64_t, xtype: uint64_t, ytype: uint64_t

// binary operator types:
#define GB_Z_TYPE uint64_t
#define GB_X_TYPE uint64_t
#define GB_Y_TYPE uint64_t

// binary operator:
#ifndef GB_GUARD_LG_randmat_mod_DEFINED
#define GB_GUARD_LG_randmat_mod_DEFINED
GB_STATIC_INLINE
void LG_randmat_mod (void *z, const void *x, const void *y)    
{                                                              
    uint64_t a = (*((uint64_t *) x)) ;                         
    uint64_t b = (*((uint64_t *) y)) ;                         
    (*((uint64_t *) z)) = a % b ;                              
}
#define GB_LG_randmat_mod_USER_DEFN \
"void LG_randmat_mod (void *z, const void *x, const void *y)    \n" \
"{                                                              \n" \
"    uint64_t a = (*((uint64_t *) x)) ;                         \n" \
"    uint64_t b = (*((uint64_t *) y)) ;                         \n" \
"    (*((uint64_t *) z)) = a % b ;                              \n" \
"}"
#endif
#define GB_BINOP(z,x,y,i,j)  LG_randmat_mod (&(z), &(x), &(y))
#define GB_COPY_A_to_C(Cx,pC,Ax,pA,A_iso)
#define GB_COPY_B_to_C(Cx,pC,Bx,pB,B_iso)

// C matrix: full
#define GB_C_IS_HYPER  0
#define GB_C_IS_SPARSE 0
#define GB_C_IS_BITMAP 0
#define GB_C_IS_FULL   1
#define GBp_C(Cp,k,vlen) ((k) * (vlen))
#define GBh_C(Ch,k)      (k)
#define GBi_C(Ci,p,vlen) ((p) % (vlen))
#define GBb_C(Cb,p)      1
#define GB_C_NVALS(e) int64_t e = (C->vlen * C->vdim)
#define GB_C_NHELD(e) GB_C_NVALS(e)
#define GB_C_ISO 0
#define GB_C_IN_ISO 0
#define GB_C_TYPE uint64_t
#define GB_PUTC(c,Cx,p) Cx [p] = c
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64
#define GB_EWISEOP(Cx,p,aij,bij,i,j) GB_BINOP (Cx [p], aij, bij, i, j)

// M matrix: none
#define GB_M_TYPE void
#define GB_MCAST(Mx,p,msize) 1
#define GB_MASK_STRUCT 1
#define GB_MASK_COMP   0
#define GB_NO_MASK     1
#define GB_Mp_TYPE uint64_t
#define GB_Mj_TYPE uint64_t
#define GB_Mi_TYPE uint64_t
#define GB_Mi_SIGNED_TYPE int64_t
#define GB_Mp_BITS 64
#define GB_Mj_BITS 64
#define GB_Mi_BITS 64

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

#include "include/GB_ewise_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__apply_bind2nd__000000999099fcc__LG_randmat_mod
#define GB_jit_query  GB_jit__apply_bind2nd__000000999099fcc__LG_randmat_mod_query
#endif
#include "template/GB_jit_kernel_apply_bind2nd.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xcec874706b3458f1 ;
    v [0] = 10 ; v [1] = 0 ; v [2] = 2 ;
    defn [0] = GB_LG_randmat_mod_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
