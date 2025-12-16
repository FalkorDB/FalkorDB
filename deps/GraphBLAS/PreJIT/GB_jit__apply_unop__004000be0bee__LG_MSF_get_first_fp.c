//------------------------------------------------------------------------------
// GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.0, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: LG_MSF_get_first_fp, ztype: double, xtype: LG_MSF_tuple_fp, ytype: void

typedef struct    
{                 
    double wFp;   
    uint64_t idx; 
} LG_MSF_tuple_fp;
#define GB_LG_MSF_tuple_fp_USER_DEFN \
"typedef struct    \n" \
"{                 \n" \
"    double wFp;   \n" \
"    uint64_t idx; \n" \
"} LG_MSF_tuple_fp;"

// unary operator types:
#define GB_Z_TYPE double
#define GB_X_TYPE LG_MSF_tuple_fp
#define GB_Y_TYPE void
#define GB_DECLAREZ(zwork) double zwork
#define GB_DECLAREX(xwork) LG_MSF_tuple_fp xwork
#define GB_DECLAREY(ywork) void ywork

// unary operator:
#ifndef GB_GUARD_LG_MSF_get_first_fp_DEFINED
#define GB_GUARD_LG_MSF_get_first_fp_DEFINED
GB_STATIC_INLINE
void LG_MSF_get_first_fp (double *y, const LG_MSF_tuple_fp *x)   
{                                                                
    *y = x->wFp;                                                 
}
#define GB_LG_MSF_get_first_fp_USER_DEFN \
"void LG_MSF_get_first_fp (double *y, const LG_MSF_tuple_fp *x)   \n" \
"{                                                                \n" \
"    *y = x->wFp;                                                 \n" \
"}"
#endif
#define GB_UNARYOP(z,x,i,j,y)  LG_MSF_get_first_fp (&(z), &(x))
#define GB_DEPENDS_ON_X 1
#define GB_DEPENDS_ON_Y 0
#define GB_DEPENDS_ON_I 0
#define GB_DEPENDS_ON_J 0
#define GB_UNOP(Cx,pC,Ax,pA,A_iso,i,j,y) GB_UNARYOP (Cx [pC], Ax [pA],  ,  ,  )

// C type:
#define GB_C_TYPE double
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Cj_SIGNED_TYPE int64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64

// A matrix: bitmap
#define GB_A_IS_HYPER  0
#define GB_A_IS_SPARSE 0
#define GB_A_IS_BITMAP 1
#define GB_A_IS_FULL   0
#define GBp_A(Ap,k,vlen) ((k) * (vlen))
#define GBh_A(Ah,k)      (k)
#define GBi_A(Ai,p,vlen) ((p) % (vlen))
#define GBb_A(Ab,p)      Ab [p]
#define GB_A_NVALS(e) int64_t e = A->nvals
#define GB_A_NHELD(e) int64_t e = (A->vlen * A->vdim)
#define GB_A_HAS_ZOMBIES 0
#define GB_A_ISO 0
#define GB_A_TYPE LG_MSF_tuple_fp
#define GB_A2TYPE LG_MSF_tuple_fp
#define GB_DECLAREA(a) LG_MSF_tuple_fp a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Aj_SIGNED_TYPE int64_t
#define GB_Ai_TYPE uint64_t
#define GB_Ai_SIGNED_TYPE int64_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 64
#define GB_Ai_BITS 64

#include "include/GB_kernel_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp
#define GB_jit_query  GB_jit__apply_unop__004000be0bee__LG_MSF_get_first_fp_query
#endif
#include "template/GB_jit_kernel_apply_unop.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x01c665219b49b989 ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 0 ;
    defn [0] = GB_LG_MSF_get_first_fp_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = GB_LG_MSF_tuple_fp_USER_DEFN ;
    defn [4] = NULL ;
    return (true) ;
}
