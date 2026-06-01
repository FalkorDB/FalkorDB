//------------------------------------------------------------------------------
// GB_jit__AxB_dot3__ff80034ee62ee653__LG_MF_Rxd_Add32_LG_MF_Rxd_Mult32.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.1, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// semiring: (LG_MF_Rxd_Add32, LG_MF_Rxd_Mult32, LG_MF_flowEdge)
typedef struct{ double residual; int32_t d; int32_t j; } LG_MF_resultTuple32;
#define GB_LG_MF_resultTuple32_USER_DEFN \
"typedef struct{ double residual; int32_t d; int32_t j; } LG_MF_resultTuple32;"

typedef struct{ double capacity; double flow; } LG_MF_flowEdge;
#define GB_LG_MF_flowEdge_USER_DEFN \
"typedef struct{ double capacity; double flow; } LG_MF_flowEdge;"


// monoid:
#define GB_Z_TYPE LG_MF_resultTuple32
#ifndef GB_GUARD_LG_MF_Rxd_Add32_DEFINED
#define GB_GUARD_LG_MF_Rxd_Add32_DEFINED
GB_STATIC_INLINE
void LG_MF_Rxd_Add32(LG_MF_resultTuple32 * z, const LG_MF_resultTuple32 * x, const LG_MF_resultTuple32 * y) { if(x->d < y->d){ (*z) = (*x) ; } else if(x->d > y->d){ (*z) = (*y) ; } else{ if(x->residual > y->residual){ (*z) = (*x) ; } else if(x->residual < y->residual){ (*z) = (*y) ; } else{ if(x->j > y->j){ (*z) = (*x); } else{ (*z) = (*y) ; } } } }
#define GB_LG_MF_Rxd_Add32_USER_DEFN \
"void LG_MF_Rxd_Add32(LG_MF_resultTuple32 * z, const LG_MF_resultTuple32 * x, const LG_MF_resultTuple32 * y) { if(x->d < y->d){ (*z) = (*x) ; } else if(x->d > y->d){ (*z) = (*y) ; } else{ if(x->residual > y->residual){ (*z) = (*x) ; } else if(x->residual < y->residual){ (*z) = (*y) ; } else{ if(x->j > y->j){ (*z) = (*x); } else{ (*z) = (*y) ; } } } }"
#endif
#define GB_ADD(z,x,y)  LG_MF_Rxd_Add32 (&(z), &(x), &(y))
#define GB_UPDATE(z,y) GB_ADD(z,z,y)
#define GB_DECLARE_IDENTITY(z) LG_MF_resultTuple32 z ; \
{ \
    const uint8_t bytes [16] = \
    { \
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, \
        0xff, 0xff, 0xff, 0x7f, 0xff, 0xff, 0xff, 0xff  \
    } ; \
    memcpy (&z, bytes, 16) ; \
}
#define GB_DECLARE_IDENTITY_CONST(z) GB_DECLARE_IDENTITY(z)
#define GB_Z_SIZE  16
#define GB_Z_NBITS 128

// multiplicative operator:
#define GB_X_TYPE LG_MF_flowEdge
#define GB_Y_TYPE int32_t
#define GB_THETA_TYPE bool
#ifndef GB_GUARD_LG_MF_Rxd_Mult32_DEFINED
#define GB_GUARD_LG_MF_Rxd_Mult32_DEFINED
GB_STATIC_INLINE
void LG_MF_Rxd_Mult32(LG_MF_resultTuple32 *z, const LG_MF_flowEdge *x, GrB_Index i, GrB_Index j, const int32_t *y, GrB_Index iy, GrB_Index jy, const bool* theta) { double r = x->capacity - x->flow; if(r > 0){ z->residual = r; z->d = (*y); z->j = j; } else{ z->residual = 0; z->d = INT32_MAX; z->j = -1; } }
#define GB_LG_MF_Rxd_Mult32_USER_DEFN \
"void LG_MF_Rxd_Mult32(LG_MF_resultTuple32 *z, const LG_MF_flowEdge *x, GrB_Index i, GrB_Index j, const int32_t *y, GrB_Index iy, GrB_Index jy, const bool* theta) { double r = x->capacity - x->flow; if(r > 0){ z->residual = r; z->d = (*y); z->j = j; } else{ z->residual = 0; z->d = INT32_MAX; z->j = -1; } }"
#endif
#define GB_MULT(z,x,y,i,k,j)  LG_MF_Rxd_Mult32 (&(z), &(x),i,k, &(y),k,j, (const GB_THETA_TYPE *) theta)

// multiply-add operator:
#define GB_MULTADD(z,x,y,i,k,j)    \
{                                  \
   GB_Z_TYPE x_op_y ;              \
   GB_MULT (x_op_y, x,y,i,k,j) ;   \
   GB_UPDATE (z, x_op_y) ;         \
}

// special cases:

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
#define GB_C_ISO 0
#define GB_C_IN_ISO 0
#define GB_C_TYPE LG_MF_resultTuple32
#define GB_PUTC(c,Cx,p) Cx [p] = c
#define GB_Cp_TYPE uint32_t
#define GB_Cj_TYPE uint32_t
#define GB_Cj_SIGNED_TYPE int32_t
#define GB_Ci_TYPE uint32_t
#define GB_Ci_SIGNED_TYPE int32_t
#define GB_Cp_BITS 32
#define GB_Cj_BITS 32
#define GB_Ci_BITS 32

// M matrix: sparse
#define GB_M_IS_HYPER  0
#define GB_M_IS_SPARSE 1
#define GB_M_IS_BITMAP 0
#define GB_M_IS_FULL   0
#define GBp_M(Mp,k,vlen) Mp [k]
#define GBh_M(Mh,k)      (k)
#define GBi_M(Mi,p,vlen) Mi [p]
#define GBb_M(Mb,p)      1
// structural mask:
#define GB_M_TYPE void
#define GB_MCAST(Mx,p,msize) 1
#define GB_MASK_STRUCT 1
#define GB_MASK_COMP   0
#define GB_NO_MASK     0
#define GB_MASK_SPARSE_STRUCTURAL_AND_NOT_COMPLEMENTED
#define GB_M_NVALS(e) int64_t e = M->nvals
#define GB_M_NHELD(e) GB_M_NVALS(e)
#define GB_Mp_TYPE uint32_t
#define GB_Mj_TYPE uint32_t
#define GB_Mj_SIGNED_TYPE int32_t
#define GB_Mi_TYPE uint32_t
#define GB_Mi_SIGNED_TYPE int32_t
#define GB_Mp_BITS 32
#define GB_Mj_BITS 32
#define GB_Mi_BITS 32

// A matrix: hypersparse
#define GB_A_IS_HYPER  1
#define GB_A_IS_SPARSE 0
#define GB_A_IS_BITMAP 0
#define GB_A_IS_FULL   0
#define GBp_A(Ap,k,vlen) Ap [k]
#define GBh_A(Ah,k)      Ah [k]
#define GBi_A(Ai,p,vlen) Ai [p]
#define GBb_A(Ab,p)      1
#define GB_A_NVALS(e) int64_t e = A->nvals
#define GB_A_NHELD(e) GB_A_NVALS(e)
#define GB_A_ISO 0
#define GB_A_TYPE LG_MF_flowEdge
#define GB_A2TYPE LG_MF_flowEdge
#define GB_DECLAREA(a) LG_MF_flowEdge a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint32_t
#define GB_Aj_TYPE uint32_t
#define GB_Aj_SIGNED_TYPE int32_t
#define GB_Ai_TYPE uint32_t
#define GB_Ai_SIGNED_TYPE int32_t
#define GB_Ap_BITS 32
#define GB_Aj_BITS 32
#define GB_Ai_BITS 32

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
#define GB_B_TYPE int32_t
#define GB_B2TYPE int32_t
#define GB_DECLAREB(b) int32_t b
#define GB_GETB(b,Bx,p,iso) b = Bx [p]
#define GB_Bp_TYPE uint64_t
#define GB_Bj_TYPE uint64_t
#define GB_Bj_SIGNED_TYPE int64_t
#define GB_Bi_TYPE uint64_t
#define GB_Bi_SIGNED_TYPE int64_t
#define GB_Bp_BITS 64
#define GB_Bj_BITS 64
#define GB_Bi_BITS 64

#include "include/GB_mxm_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__AxB_dot3__ff80034ee62ee653__LG_MF_Rxd_Add32_LG_MF_Rxd_Mult32
#define GB_jit_query  GB_jit__AxB_dot3__ff80034ee62ee653__LG_MF_Rxd_Add32_LG_MF_Rxd_Mult32_query
#endif
#include "template/GB_jit_kernel_AxB_dot3.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xc1acea62559cef5d ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 1 ;
    defn [0] = GB_LG_MF_Rxd_Add32_USER_DEFN ;
    defn [1] = GB_LG_MF_Rxd_Mult32_USER_DEFN ;
    defn [2] = GB_LG_MF_resultTuple32_USER_DEFN ;
    defn [3] = GB_LG_MF_flowEdge_USER_DEFN ;
    defn [4] = NULL ;
    if (id_size != 16 || term_size != 0) return (false) ;
    GB_DECLARE_IDENTITY_CONST (zidentity) ;
    if (id == NULL || memcmp (id, &zidentity, 16) != 0) return (false) ;
    return (true) ;
}
