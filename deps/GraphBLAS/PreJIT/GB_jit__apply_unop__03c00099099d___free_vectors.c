//------------------------------------------------------------------------------
// GB_jit__apply_unop__03c00099099d___free_vectors.c
//------------------------------------------------------------------------------
// SuiteSparse:GraphBLAS v10.4.0, Timothy A. Davis, (c) 2017-2026,
// All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// The above copyright and license do not apply to any
// user-defined types and operators defined below.
//------------------------------------------------------------------------------

#include "include/GB_jit_kernel.h"

// op: _free_vectors, ztype: uint64_t, xtype: uint64_t, ytype: void

// unary operator types:
#define GB_Z_TYPE uint64_t
#define GB_X_TYPE uint64_t
#define GB_Y_TYPE void
#define GB_DECLAREZ(zwork) uint64_t zwork
#define GB_DECLAREX(xwork) uint64_t xwork
#define GB_DECLAREY(ywork) void ywork

// unary operator:
#ifndef GB_GUARD__free_vectors_DEFINED
#define GB_GUARD__free_vectors_DEFINED
GB_STATIC_INLINE
void _free_vectors ( void *z, const uint64_t *x ) { if(!!((*x) & (1UL << (sizeof(uint64_t) * 8 - 1)))) { GrB_Vector V = ((GrB_Vector)(((*x) & ~(1UL << (sizeof(uint64_t) * 8 - 1))))); _Generic ((&V), GrB_Type *: GrB_Type_free , GrB_UnaryOp *: GrB_UnaryOp_free , GrB_BinaryOp *: GrB_BinaryOp_free , GrB_IndexUnaryOp *: GrB_IndexUnaryOp_free , GxB_IndexBinaryOp*: GxB_IndexBinaryOp_free, GrB_Monoid *: GrB_Monoid_free , GrB_Semiring *: GrB_Semiring_free , GrB_Scalar *: GrB_Scalar_free , GrB_Vector *: GrB_Vector_free , GrB_Matrix *: GrB_Matrix_free , GrB_Descriptor *: GrB_Descriptor_free , GxB_Context *: GxB_Context_free , GxB_Container *: GxB_Container_free , GxB_Iterator *: GxB_Iterator_free) (&V); } }
#define GB__free_vectors_USER_DEFN \
"void _free_vectors ( void *z, const uint64_t *x ) { if(!!((*x) & (1UL << (sizeof(uint64_t) * 8 - 1)))) { GrB_Vector V = ((GrB_Vector)(((*x) & ~(1UL << (sizeof(uint64_t) * 8 - 1))))); _Generic ((&V), GrB_Type *: GrB_Type_free , GrB_UnaryOp *: GrB_UnaryOp_free , GrB_BinaryOp *: GrB_BinaryOp_free , GrB_IndexUnaryOp *: GrB_IndexUnaryOp_free , GxB_IndexBinaryOp*: GxB_IndexBinaryOp_free, GrB_Monoid *: GrB_Monoid_free , GrB_Semiring *: GrB_Semiring_free , GrB_Scalar *: GrB_Scalar_free , GrB_Vector *: GrB_Vector_free , GrB_Matrix *: GrB_Matrix_free , GrB_Descriptor *: GrB_Descriptor_free , GxB_Context *: GxB_Context_free , GxB_Container *: GxB_Container_free , GxB_Iterator *: GxB_Iterator_free) (&V); } }"
#endif
#define GB_UNARYOP(z,x,i,j,y)  _free_vectors (&(z), &(x))
#define GB_DEPENDS_ON_X 1
#define GB_DEPENDS_ON_Y 0
#define GB_DEPENDS_ON_I 0
#define GB_DEPENDS_ON_J 0
#define GB_UNOP(Cx,pC,Ax,pA,A_iso,i,j,y) GB_UNARYOP (Cx [pC], Ax [pA],  ,  ,  )

// C type:
#define GB_C_TYPE uint64_t
#define GB_Cp_TYPE uint64_t
#define GB_Cj_TYPE uint64_t
#define GB_Cj_SIGNED_TYPE int64_t
#define GB_Ci_TYPE uint64_t
#define GB_Ci_SIGNED_TYPE int64_t
#define GB_Cp_BITS 64
#define GB_Cj_BITS 64
#define GB_Ci_BITS 64

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
#define GB_A_HAS_ZOMBIES 0
#define GB_A_ISO 0
#define GB_A_TYPE uint64_t
#define GB_A2TYPE uint64_t
#define GB_DECLAREA(a) uint64_t a
#define GB_GETA(a,Ax,p,iso) a = Ax [p]
#define GB_Ap_TYPE uint32_t
#define GB_Aj_TYPE uint32_t
#define GB_Aj_SIGNED_TYPE int32_t
#define GB_Ai_TYPE uint32_t
#define GB_Ai_SIGNED_TYPE int32_t
#define GB_Ap_BITS 32
#define GB_Aj_BITS 32
#define GB_Ai_BITS 32

#include "include/GB_kernel_shared_definitions.h"
#ifndef GB_JIT_RUNTIME
#define GB_jit_kernel GB_jit__apply_unop__03c00099099d___free_vectors
#define GB_jit_query  GB_jit__apply_unop__03c00099099d___free_vectors_query
#endif
#include "template/GB_jit_kernel_apply_unop.c"
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query) ;
GB_JIT_GLOBAL GB_JIT_QUERY_PROTO (GB_jit_query)
{
    (*hash) = 0x07a5ffac02bd1763 ;
    v [0] = 10 ; v [1] = 4 ; v [2] = 0 ;
    defn [0] = GB__free_vectors_USER_DEFN ;
    defn [1] = NULL ;
    defn [2] = NULL ;
    defn [3] = NULL ;
    defn [4] = NULL ;
    return (true) ;
}
