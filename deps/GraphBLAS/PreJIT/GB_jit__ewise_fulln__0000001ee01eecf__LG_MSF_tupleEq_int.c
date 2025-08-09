//------------------------------------------------------------------------------
// GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_int.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.0.2, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: LG_MSF_tupleEq_int, ztype: bool, xtype: LG_MSF_tuple_int, ytype: LG_MSF_tuple_int

typedef struct    
{                 
    int64_t wInt; 
    uint64_t idx; 
} LG_MSF_tuple_int;
#define GB_LG_MSF_tuple_int_USER_DEFN \
"typedef struct    \n" \
"{                 \n" \
"    int64_t wInt; \n" \
"    uint64_t idx; \n" \
"} LG_MSF_tuple_int;"

// binary operator types:
#define GB_Z_TYPE bool
#define GB_X_TYPE LG_MSF_tuple_int
#define GB_Y_TYPE LG_MSF_tuple_int

// binary operator:
#ifndef GB_GUARD_LG_MSF_tupleEq_int_DEFINED
#define GB_GUARD_LG_MSF_tupleEq_int_DEFINED
GB_STATIC_INLINE
void LG_MSF_tupleEq_int                             
(                                                   
    bool *z,                                        
    const LG_MSF_tuple_int *x,                      
    const LG_MSF_tuple_int *y                       
)                                                   
{                                                   
    *z = (x->wInt == y->wInt) && (x->idx == y->idx);
}
#define GB_LG_MSF_tupleEq_int_USER_DEFN \
"void LG_MSF_tupleEq_int                             \n" \
"(                                                   \n" \
"    bool *z,                                        \n" \
"    const LG_MSF_tuple_int *x,                      \n" \
"    const LG_MSF_tuple_int *y                       \n" \
")                                                   \n" \
"{                                                   \n" \
"    *z = (x->wInt == y->wInt) && (x->idx == y->idx);\n" \
"}"
#endif
#define GB_BINOP(z,x,y,i,j)  LG_MSF_tupleEq_int (&(z), &(x), &(y))
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
#define GB_C_TYPE bool
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
#define GB_A_TYPE LG_MSF_tuple_int
#define GB_A2TYPE LG_MSF_tuple_int
#define GB_DECLAREA(a) LG_MSF_tuple_int a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Ai_TYPE uint64_t
#define GB_Ai_SIGNED_TYPE int64_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 64
#define GB_Ai_BITS 64

// B matrix: full
#define GB_B_IS_HYPER  0
#define GB_B_IS_SPARSE 0
#define GB_B_IS_BITMAP 0
#define GB_B_IS_FULL   1
#define GBp_B(Bp,k,vlen) ((k) * (vlen))
#define GBh_B(Bh,k)      (k)
#define GBi_B(Bi,p,vlen) ((p) % (vlen))
#define GBb_B(Bb,p)      1
#define GB_B_NVALS(e) int64_t e = (B->vlen * B->vdim)
#define GB_B_NHELD(e) GB_B_NVALS(e)
#define GB_B_ISO 0
#define GB_B_TYPE LG_MSF_tuple_int
#define GB_B2TYPE LG_MSF_tuple_int
#define GB_DECLAREB(b) LG_MSF_tuple_int b
#define GB_GETB(b,Bx,p,iso) b = Bx [p]
#define GB_Bp_TYPE uint64_t
#define GB_Bj_TYPE uint64_t
#define GB_Bi_TYPE uint64_t
#define GB_Bi_SIGNED_TYPE int64_t
#define GB_Bp_BITS 64
#define GB_Bj_BITS 64
#define GB_Bi_BITS 64

#include "include/GB_ewise_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_int
#define GB_jit_query  GB_jit__ewise_fulln__0000001ee01eecf__LG_MSF_tupleEq_int_query
#endif
#include "template/GB_jit_kernel_ewise_fulln.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xbd23e39b140f8393 ;
    v [0] = 10 ; v [1] = 0 ; v [2] = 2 ;
    defn [0] = GB_LG_MSF_tupleEq_int_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = GB_LG_MSF_tuple_int_USER_DEFN ;
    defn [4] = defn [3] ;
    return (true) ;
}
