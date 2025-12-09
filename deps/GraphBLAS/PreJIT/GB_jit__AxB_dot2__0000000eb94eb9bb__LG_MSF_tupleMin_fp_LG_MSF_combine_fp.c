//------------------------------------------------------------------------------
// GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.3.0, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// semiring: (LG_MSF_tupleMin_fp, LG_MSF_combine_fp, double)
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


// monoid:
#define GB_Z_TYPE LG_MSF_tuple_fp
#ifndef GB_GUARD_LG_MSF_tupleMin_fp_DEFINED
#define GB_GUARD_LG_MSF_tupleMin_fp_DEFINED
GB_STATIC_INLINE
void LG_MSF_tupleMin_fp                                                      
(                                                                            
    LG_MSF_tuple_fp *z,                                                      
    const LG_MSF_tuple_fp *x,                                                
    const LG_MSF_tuple_fp *y                                                 
)                                                                            
{                                                                            
    bool xSmaller = x->wFp < y->wFp || (x->wFp == y->wFp && x->idx < y->idx);
    z->wFp = (xSmaller)? x->wFp: y->wFp;                                     
    z->idx = (xSmaller)? x->idx: y->idx;                                     
}
#define GB_LG_MSF_tupleMin_fp_USER_DEFN \
"void LG_MSF_tupleMin_fp                                                      \n" \
"(                                                                            \n" \
"    LG_MSF_tuple_fp *z,                                                      \n" \
"    const LG_MSF_tuple_fp *x,                                                \n" \
"    const LG_MSF_tuple_fp *y                                                 \n" \
")                                                                            \n" \
"{                                                                            \n" \
"    bool xSmaller = x->wFp < y->wFp || (x->wFp == y->wFp && x->idx < y->idx);\n" \
"    z->wFp = (xSmaller)? x->wFp: y->wFp;                                     \n" \
"    z->idx = (xSmaller)? x->idx: y->idx;                                     \n" \
"}"
#endif
#define GB_ADD(z,x,y)  LG_MSF_tupleMin_fp (&(z), &(x), &(y))
#define GB_UPDATE(z,y) GB_ADD(z,z,y)
#define GB_DECLARE_IDENTITY(z) LG_MSF_tuple_fp z ; \
{ \
    const uint8_t bytes [16] = \
    { \
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x7f, \
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff  \
    } ; \
    memcpy (&z, bytes, 16) ; \
}
#define GB_DECLARE_IDENTITY_CONST(z) GB_DECLARE_IDENTITY(z)
#define GB_Z_SIZE  16
#define GB_Z_NBITS 128

// multiplicative operator:
#define GB_X_TYPE double
#define GB_Y_TYPE uint64_t
#ifndef GB_GUARD_LG_MSF_combine_fp_DEFINED
#define GB_GUARD_LG_MSF_combine_fp_DEFINED
GB_STATIC_INLINE
void LG_MSF_combine_fp  
(                       
    LG_MSF_tuple_fp *z, 
    const double *x,    
    const uint64_t *y   
)                       
{                       
    z->wFp = *x;        
    z->idx = *y;        
}
#define GB_LG_MSF_combine_fp_USER_DEFN \
"void LG_MSF_combine_fp  \n" \
"(                       \n" \
"    LG_MSF_tuple_fp *z, \n" \
"    const double *x,    \n" \
"    const uint64_t *y   \n" \
")                       \n" \
"{                       \n" \
"    z->wFp = *x;        \n" \
"    z->idx = *y;        \n" \
"}"
#endif
#define GB_MULT(z,x,y,i,k,j)  LG_MSF_combine_fp (&(z), &(x), &(y))

// multiply-add operator:
#define GB_MULTADD(z,x,y,i,k,j)    \
{                                  \
   GB_Z_TYPE x_op_y ;              \
   GB_MULT (x_op_y, x,y,i,k,j) ;   \
   GB_UPDATE (z, x_op_y) ;         \
}

// special cases:

// C matrix: bitmap
#define GB_C_IS_HYPER  0
#define GB_C_IS_SPARSE 0
#define GB_C_IS_BITMAP 1
#define GB_C_IS_FULL   0
#define GBp_C(Cp,k,vlen) ((k) * (vlen))
#define GBh_C(Ch,k)      (k)
#define GBi_C(Ci,p,vlen) ((p) % (vlen))
#define GBb_C(Cb,p)      Cb [p]
#define GB_C_NVALS(e) int64_t e = C->nvals
#define GB_C_NHELD(e) int64_t e = (C->vlen * C->vdim)
#define GB_C_ISO 0
#define GB_C_IN_ISO 0
#define GB_C_TYPE LG_MSF_tuple_fp
#define GB_PUTC(c,Cx,p) Cx [p] = c
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Cj_SIGNED_TYPE int64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64

// M matrix: full
#define GB_M_IS_HYPER  0
#define GB_M_IS_SPARSE 0
#define GB_M_IS_BITMAP 0
#define GB_M_IS_FULL   1
#define GBp_M(Mp,k,vlen) ((k) * (vlen))
#define GBh_M(Mh,k)      (k)
#define GBi_M(Mi,p,vlen) ((p) % (vlen))
#define GBb_M(Mb,p)      1
// valued mask (1 byte):
#define GB_M_TYPE uint8_t
#define GB_MCAST(Mx,p,msize) (Mx [p] != 0)
#define GB_MASK_STRUCT 0
#define GB_MASK_COMP   0
#define GB_NO_MASK     0
#define GB_M_NVALS(e) int64_t e = (M->vlen * M->vdim)
#define GB_M_NHELD(e) GB_M_NVALS(e)
#define GB_Mp_TYPE uint64_t
#define GB_Mj_TYPE uint64_t
#define GB_Mj_SIGNED_TYPE int64_t
#define GB_Mi_TYPE uint64_t
#define GB_Mi_SIGNED_TYPE int64_t
#define GB_Mp_BITS 64
#define GB_Mj_BITS 64
#define GB_Mi_BITS 64

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
#define GB_A_ISO 0
#define GB_A_TYPE double
#define GB_A2TYPE double
#define GB_DECLAREA(a) double a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint64_t
#define GB_Aj_SIGNED_TYPE int64_t
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
#define GB_B_TYPE uint64_t
#define GB_B2TYPE uint64_t
#define GB_DECLAREB(b) uint64_t b
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
#define GB_jit_kernel GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp
#define GB_jit_query  GB_jit__AxB_dot2__0000000eb94eb9bb__LG_MSF_tupleMin_fp_LG_MSF_combine_fp_query
#endif
#include "template/GB_jit_kernel_AxB_dot2.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0xe160d6883041e19f ;
    v [0] = 10 ; v [1] = 3 ; v [2] = 0 ;
    defn [0] = GB_LG_MSF_tupleMin_fp_USER_DEFN ;
    defn [1] = GB_LG_MSF_combine_fp_USER_DEFN ;
    defn [2] = GB_LG_MSF_tuple_fp_USER_DEFN ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    if (id_size != 16 || term_size != 0) return (false) ;
    GB_DECLARE_IDENTITY_CONST (zidentity) ;
    if (id == NULL || memcmp (id, &zidentity, 16) != 0) return (false) ;
    return (true) ;
}
