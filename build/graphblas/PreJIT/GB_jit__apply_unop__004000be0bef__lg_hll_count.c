//------------------------------------------------------------------------------
// GB_jit__apply_unop__004000be0bef__lg_hll_count.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.1, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: lg_hll_count, ztype: double, xtype: HLL, ytype: void

typedef struct { uint8_t registers[(1 << 10)]; } HLL;
#define GB_HLL_USER_DEFN \
"typedef struct { uint8_t registers[(1 << 10)]; } HLL;"

// unary operator types:
#define GB_Z_TYPE double
#define GB_X_TYPE HLL
#define GB_Y_TYPE void
#define GB_DECLAREZ(zwork) double zwork
#define GB_DECLAREX(xwork) HLL xwork
#define GB_DECLAREY(ywork) void ywork

// unary operator:
#ifndef GB_GUARD_lg_hll_count_DEFINED
#define GB_GUARD_lg_hll_count_DEFINED
GB_STATIC_INLINE
void lg_hll_count(double *z, const HLL *x) { const HLL *hll = x; double alpha_mm = 0.7213 / (1.0 + 1.079 / (double)(1 << 10)); alpha_mm *= ((double)(1 << 10) * (double)(1 << 10)); double sum = 0; for (uint32_t i = 0; i < (1 << 10); i++) { sum += 1.0 / (1 << hll->registers[i]); } double estimate = alpha_mm / sum; if (estimate <= 5.0 / 2.0 * (double)(1 << 10)) { int zeros = 0; for (uint32_t i = 0; i < (1 << 10); i++) zeros += (hll->registers[i] == 0); if (zeros) estimate = (double)(1 << 10) * log((double)(1 << 10) / zeros); } else if (estimate > (1.0 / 30.0) * 4294967296.0) { estimate = -4294967296.0 * log(1.0 - (estimate / 4294967296.0)) ; } *z = estimate; }
#define GB_lg_hll_count_USER_DEFN \
"void lg_hll_count(double *z, const HLL *x) { const HLL *hll = x; double alpha_mm = 0.7213 / (1.0 + 1.079 / (double)(1 << 10)); alpha_mm *= ((double)(1 << 10) * (double)(1 << 10)); double sum = 0; for (uint32_t i = 0; i < (1 << 10); i++) { sum += 1.0 / (1 << hll->registers[i]); } double estimate = alpha_mm / sum; if (estimate <= 5.0 / 2.0 * (double)(1 << 10)) { int zeros = 0; for (uint32_t i = 0; i < (1 << 10); i++) zeros += (hll->registers[i] == 0); if (zeros) estimate = (double)(1 << 10) * log((double)(1 << 10) / zeros); } else if (estimate > (1.0 / 30.0) * 4294967296.0) { estimate = -4294967296.0 * log(1.0 - (estimate / 4294967296.0)) ; } *z = estimate; }"
#endif
#define GB_UNARYOP(z,x,i,j,y)  lg_hll_count (&(z), &(x))
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
#define GB_A_TYPE HLL
#define GB_A2TYPE HLL
#define GB_DECLAREA(a) HLL a
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
#define GB_jit_kernel GB_jit__apply_unop__004000be0bef__lg_hll_count
#define GB_jit_query  GB_jit__apply_unop__004000be0bef__lg_hll_count_query
#endif
#include "template/GB_jit_kernel_apply_unop.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xc79aa66cc559fc42 ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 1 ;
    defn [0] = GB_lg_hll_count_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = GB_HLL_USER_DEFN ;
    defn [4] = NULL ;
    return (true) ;
}
