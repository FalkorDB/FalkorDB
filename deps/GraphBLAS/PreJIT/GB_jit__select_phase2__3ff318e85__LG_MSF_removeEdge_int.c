//------------------------------------------------------------------------------
// GB_jit__select_phase2__3ff318e85__LG_MSF_removeEdge_int.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.0.2, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: LG_MSF_removeEdge_int (flipped ij), ztype: bool, xtype: int64_t, ytype: LG_MSF_context_int

typedef struct           
{                        
    uint64_t    *parent; 
    struct               
    {                    
        int64_t wInt;    
        uint64_t idx;    
    } *w_partner;        
} LG_MSF_context_int;
#define GB_LG_MSF_context_int_USER_DEFN \
"typedef struct           \n" \
"{                        \n" \
"    uint64_t    *parent; \n" \
"    struct               \n" \
"    {                    \n" \
"        int64_t wInt;    \n" \
"        uint64_t idx;    \n" \
"    } *w_partner;        \n" \
"} LG_MSF_context_int;"

// unary operator types:
#define GB_Z_TYPE bool
#define GB_X_TYPE int64_t
#define GB_Y_TYPE LG_MSF_context_int

// index unary operator (flipped ij):
#ifndef GB_GUARD_LG_MSF_removeEdge_int_DEFINED
#define GB_GUARD_LG_MSF_removeEdge_int_DEFINED
GB_STATIC_INLINE
void LG_MSF_removeEdge_int                        
(                                                 
    bool *z,                                      
    const int64_t *x,                             
    GrB_Index i,                                  
    GrB_Index j,                                  
    const LG_MSF_context_int *theta               
)                                                 
{                                                 
    (*z) = (theta->parent[i] != theta->parent[j]);
}
#define GB_LG_MSF_removeEdge_int_USER_DEFN \
"void LG_MSF_removeEdge_int                        \n" \
"(                                                 \n" \
"    bool *z,                                      \n" \
"    const int64_t *x,                             \n" \
"    GrB_Index i,                                  \n" \
"    GrB_Index j,                                  \n" \
"    const LG_MSF_context_int *theta               \n" \
")                                                 \n" \
"{                                                 \n" \
"    (*z) = (theta->parent[i] != theta->parent[j]);\n" \
"}"
#endif
#define GB_IDXUNOP(z,x,j,i,y) LG_MSF_removeEdge_int (&(z), &(x), i, j, &(y))
#define GB_DEPENDS_ON_X 1
#define GB_DEPENDS_ON_I 1
#define GB_DEPENDS_ON_J 1
#define GB_DEPENDS_ON_Y 1
#define GB_ENTRY_SELECTOR

// test if A(i,j) is to be kept:
#define GB_TEST_VALUE_OF_ENTRY(keep,p) \
    bool keep ;                        \
    GB_IDXUNOP (keep, Ax [0], i, j, y) ;

// copy A(i,j) to C(i,j):
#define GB_SELECT_ENTRY(Cx,pC,Ax,pA)
#define GB_ISO_SELECT 1

// C matrix: sparse
#define GB_C_IS_HYPER  0
#define GB_C_IS_SPARSE 1
#define GB_C_IS_BITMAP 0
#define GB_C_IS_FULL   0
#define GBp_C(Cp,k,vlen) Cp [k]
#define GBh_C(Ch,k)      (k)
#define GBi_C(Ci,p,vlen) Ci [p]
#define GBb_C(Cb,p)      1
#define GB_C_NVALS(e) int64_t e = C->nvals
#define GB_C_NHELD(e) GB_C_NVALS(e)
#define GB_C_ISO 1
#define GB_C_IN_ISO 1
#define GB_C_TYPE void
#define GB_PUTC(c,Cx,p)
#define GB_Cp_TYPE uint32_t
#define GB_Cj_TYPE uint32_t
#define GB_Ci_TYPE uint32_t
#define GB_Ci_SIGNED_TYPE int32_t
#define GB_Cp_BITS 32
#define GB_Cj_BITS 32
#define GB_Ci_BITS 32

// A matrix: sparse
#define GB_A_IS_HYPER  0
#define GB_A_IS_SPARSE 1
#define GB_A_IS_BITMAP 0
#define GB_A_IS_FULL   0
#define GBp_A(Ap,k,vlen) Ap [k]
#define GBh_A(Ah,k)      (k)
#define GBi_A(Ai,p,vlen) Ai [p]
#define GBb_A(Ab,p)      1
#define GB_A_NVALS(e) int64_t e = A->nvals
#define GB_A_NHELD(e) GB_A_NVALS(e)
#define GB_A_ISO 1
#define GB_A_TYPE int64_t
#define GB_A2TYPE int64_t
#define GB_DECLAREA(a) int64_t a
#define GB_GETA(a,Ax,p,iso) a = Ax [0]
#define GB_Ap_TYPE uint32_t
#define GB_Aj_TYPE uint32_t
#define GB_Ai_TYPE uint32_t
#define GB_Ai_SIGNED_TYPE int32_t
#define GB_Ap_BITS 32
#define GB_Aj_BITS 32
#define GB_Ai_BITS 32

#include "include/GB_select_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__select_phase2__3ff318e85__LG_MSF_removeEdge_int
#define GB_jit_query  GB_jit__select_phase2__3ff318e85__LG_MSF_removeEdge_int_query
#endif
#include "template/GB_jit_kernel_select_phase2.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x2890c43be09fe483 ;
    v [0] = 10 ; v [1] = 0 ; v [2] = 2 ;
    defn [0] = GB_LG_MSF_removeEdge_int_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
