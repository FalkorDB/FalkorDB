//------------------------------------------------------------------------------
// GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_int_LG_MSF_tuple2nd_int.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.0.2, Timothy A. Davis, (c) 2017-2025,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// semiring: (LG_MSF_tupleMin_int, LG_MSF_tuple2nd_int, bool)
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


// monoid:
#define GB_Z_TYPE LG_MSF_tuple_int
#ifndef GB_GUARD_LG_MSF_tupleMin_int_DEFINED
#define GB_GUARD_LG_MSF_tupleMin_int_DEFINED
GB_STATIC_INLINE
void LG_MSF_tupleMin_int                        
(                                               
    LG_MSF_tuple_int *z,                        
    const LG_MSF_tuple_int *x,                  
    const LG_MSF_tuple_int *y                   
)                                               
{                                               
    bool xSmaller = x->wInt < y->wInt ||        
        (x->wInt == y->wInt && x->idx < y->idx);
    z->wInt = (xSmaller)? x->wInt: y->wInt;     
    z->idx = (xSmaller)? x->idx: y->idx;        
}
#define GB_LG_MSF_tupleMin_int_USER_DEFN \
"void LG_MSF_tupleMin_int                        \n" \
"(                                               \n" \
"    LG_MSF_tuple_int *z,                        \n" \
"    const LG_MSF_tuple_int *x,                  \n" \
"    const LG_MSF_tuple_int *y                   \n" \
")                                               \n" \
"{                                               \n" \
"    bool xSmaller = x->wInt < y->wInt ||        \n" \
"        (x->wInt == y->wInt && x->idx < y->idx);\n" \
"    z->wInt = (xSmaller)? x->wInt: y->wInt;     \n" \
"    z->idx = (xSmaller)? x->idx: y->idx;        \n" \
"}"
#endif
#define GB_ADD(z,x,y)  LG_MSF_tupleMin_int (&(z), &(x), &(y))
#define GB_UPDATE(z,y) GB_ADD(z,z,y)
#define GB_DECLARE_IDENTITY(z) LG_MSF_tuple_int z ; \
{ \
    const uint8_t bytes [16] = \
    { \
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f, \
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff  \
    } ; \
    memcpy (&z, bytes, 16) ; \
}
#define GB_DECLARE_IDENTITY_CONST(z) GB_DECLARE_IDENTITY(z)
#define GB_Z_SIZE  16
#define GB_Z_NBITS 128

// multiplicative operator:
#define GB_X_TYPE bool
#define GB_Y_TYPE LG_MSF_tuple_int
#ifndef GB_GUARD_LG_MSF_tuple2nd_int_DEFINED
#define GB_GUARD_LG_MSF_tuple2nd_int_DEFINED
GB_STATIC_INLINE
void LG_MSF_tuple2nd_int     
(                            
    LG_MSF_tuple_int *z,     
    const void *x,           
    const LG_MSF_tuple_int *y
)                            
{                            
    z->wInt = y->wInt;       
    z->idx = y->idx;         
}
#define GB_LG_MSF_tuple2nd_int_USER_DEFN \
"void LG_MSF_tuple2nd_int     \n" \
"(                            \n" \
"    LG_MSF_tuple_int *z,     \n" \
"    const void *x,           \n" \
"    const LG_MSF_tuple_int *y\n" \
")                            \n" \
"{                            \n" \
"    z->wInt = y->wInt;       \n" \
"    z->idx = y->idx;         \n" \
"}"
#endif
#define GB_MULT(z,x,y,i,k,j)  LG_MSF_tuple2nd_int (&(z), &(x), &(y))

// multiply-add operator:
#define GB_MULTADD(z,x,y,i,k,j)    \
{                                  \
   GB_Z_TYPE x_op_y ;              \
   GB_MULT (x_op_y, x,y,i,k,j) ;   \
   GB_UPDATE (z, x_op_y) ;         \
}

// special cases:

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
#define GB_C_TYPE LG_MSF_tuple_int
#define GB_PUTC(c,Cx,p) Cx [p] = c
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64

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
#define GB_A_TYPE bool
#define GB_A2TYPE bool
#define GB_DECLAREA(a) bool a
#define GB_GETA(a,Ax,p,iso) a = Ax [0]
#define GB_Ap_TYPE uint64_t
#define GB_Aj_TYPE uint32_t
#define GB_Ai_TYPE uint64_t
#define GB_Ai_SIGNED_TYPE int64_t
#define GB_Ap_BITS 64
#define GB_Aj_BITS 32
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

#include "include/GB_mxm_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_int_LG_MSF_tuple2nd_int
#define GB_jit_query  GB_jit__AxB_saxpy4__0100400e1e0e1ec7__LG_MSF_tupleMin_int_LG_MSF_tuple2nd_int_query
#endif
#include "template/GB_jit_kernel_AxB_saxpy4.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x36890dbdb4efe069 ;
    v [0] = 10 ; v [1] = 0 ; v [2] = 2 ;
    defn [0] = GB_LG_MSF_tupleMin_int_USER_DEFN ;
    defn [1] = GB_LG_MSF_tuple2nd_int_USER_DEFN ;
    defn [2] = GB_LG_MSF_tuple_int_USER_DEFN ;
    defn [3] = NULL ;
    defn [4] = defn [2] ;
    if (id_size != 16 || term_size != 0) return (false) ;
    GB_DECLARE_IDENTITY_CONST (zidentity) ;
    if (id == NULL || memcmp (id, &zidentity, 16) != 0) return (false) ;
    return (true) ;
}
