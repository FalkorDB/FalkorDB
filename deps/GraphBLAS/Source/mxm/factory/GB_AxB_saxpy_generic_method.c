//------------------------------------------------------------------------------
// GB_AxB_saxpy_generic_method: C=A*B, C<M>=A*B, or C<!M>=A*B
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_AxB_saxpy_generic_method computes C=A*B, C<M>=A*B, or C<!M>=A*B.  with
// arbitrary types and operators.  C can be hyper, sparse, or bitmap, but not
// full.  For all cases, the four matrices C, M (if present), A, and B have the
// same format (by-row or by-column), or they represent implicitly transposed
// matrices with the same effect.  This method does not handle the dot-product
// methods, which compute C=A'*B if A and B are held by column, or equivalently
// A*B' if both are held by row.

// This method uses GB_AxB_saxpy_generic_template.c to implement two
// meta-methods, each of which can contain further specialized methods (such as
// the fine/coarse x Gustavson/Hash, mask/no-mask methods in saxpy3):

// saxpy3: general purpose method, where C is sparse or hypersparse,
//          via GB_AxB_saxpy3_template.c.  SaxpyTasks holds the (fine/coarse x
//          Gustavson/Hash) tasks constructed by GB_AxB_saxpy3_slice*.

// saxbit: general purpose method, where C is bitmap, via
//          GB_AxB_saxbit_template.c.  The method constructs its own
//          tasks in workspace defined and freed in that template.

// C is not iso, nor is it full.

// This template is used to construct the following methods, all of which
// are called by GB_AxB_saxpy_generic:

//      GB_AxB_saxpy3_generic_first
//      GB_AxB_saxpy3_generic_second
//      GB_AxB_saxpy3_generic_flipped
//      GB_AxB_saxpy3_generic_unflipped

//      GB_AxB_saxbit_generic_first
//      GB_AxB_saxbit_generic_second
//      GB_AxB_saxbit_generic_flipped
//      GB_AxB_saxbit_generic_unflipped

//------------------------------------------------------------------------------

#include "mxm/GB_AxB_saxpy.h"
#include "slice/GB_ek_slice.h"
#include "binaryop/GB_binop.h"
#include "assign/GB_bitmap_assign_methods.h"
#include "mxm/include/GB_mxm_shared_definitions.h"
#include "mxm/GB_AxB_saxpy_generic.h"
#include "generic/GB_generic.h"

GrB_Info GB_AXB_SAXPY_GENERIC_METHOD
(
    GrB_Matrix C,                   // any sparsity except full
    const GrB_Matrix M,
    bool Mask_comp,
    const bool Mask_struct,
    const bool M_in_place,          // ignored if C is bitmap
    const GrB_Matrix A,
    bool A_is_pattern,
    const GrB_Matrix B,
    bool B_is_pattern,
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const int ntasks,
    const int nthreads,

    #if GB_GENERIC_C_IS_SPARSE_OR_HYPERSPARSE
    // for saxpy3 only:
    GB_saxpy3task_struct *restrict SaxpyTasks,
    const int nfine,
    const int do_sort,              // if true, sort in saxpy3
    GB_Werk Werk
    #else
    // for saxbit only:
    const int nfine_tasks_per_vector,
    const bool use_coarse_tasks,
    const bool use_atomics,
    const int64_t *restrict M_ek_slicing,
    const int M_nthreads,
    const int M_ntasks,
    const int64_t *restrict A_slice,
    const int64_t *restrict H_slice,
    GB_void *restrict Wcx,
    int8_t *restrict Wf
    #endif
)
{

    //--------------------------------------------------------------------------
    // get operators, functions, workspace, contents of A, B, and C
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    GrB_Monoid add = semiring->add ;
    ASSERT (mult->ztype == add->op->ztype) ;
    ASSERT (mult->ztype == C->type) ;

    GxB_binary_function fmult = mult->binop_function ;    // NULL if positional
    GxB_index_binary_function fmult_idx = mult->idxbinop_function ;
    GxB_binary_function fadd  = add->op->binop_function ;
    GB_Opcode opcode = mult->opcode ;

    size_t csize = C->type->size ;
    size_t asize = A_is_pattern ? 0 : A->type->size ;
    size_t bsize = B_is_pattern ? 0 : B->type->size ;

    size_t xsize = mult->xtype->size ;
    size_t ysize = mult->ytype->size ;

    // scalar workspace: because of typecasting, the x/y types need not
    // be the same as the size of the A and B types.
    // GB_GENERIC_FLIPXY false: aik = (xtype) A(i,k) and bkj = (ytype) B(k,j)
    // GB_GENERIC_FLIPXY true:  aik = (ytype) A(i,k) and bkj = (xtype) B(k,j)
    size_t aik_size = GB_GENERIC_FLIPXY ? ysize : xsize ;
    size_t bkj_size = GB_GENERIC_FLIPXY ? xsize : ysize ;

    GB_cast_function cast_A, cast_B ;
    #if GB_GENERIC_FLIPXY
    { 
        // A is typecasted to y, and B is typecasted to x
        cast_A = A_is_pattern ? NULL : 
                 GB_cast_factory (mult->ytype->code, A->type->code) ;
        cast_B = B_is_pattern ? NULL : 
                 GB_cast_factory (mult->xtype->code, B->type->code) ;
    }
    #else
    { 
        // A is typecasted to x, and B is typecasted to y
        cast_A = A_is_pattern ? NULL :
                 GB_cast_factory (mult->xtype->code, A->type->code) ;
        cast_B = B_is_pattern ? NULL :
                 GB_cast_factory (mult->ytype->code, B->type->code) ;
    }
    #endif

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // C = A*B via saxpy3 or bitmap method, function pointers, and typecasting
    //--------------------------------------------------------------------------

    // This is before typecast to GB_B2TYPE, so it is the size of the
    // entries in the B matrix, not as typecasted to GB_B2TYPE.
    #define GB_B_SIZE bsize

    // definitions for GB_AxB_saxpy_generic_template.c
    #include "mxm/include/GB_AxB_saxpy3_template.h"

    // aik = A(i,k), located in Ax [A_iso ? 0:pA]
    #undef  GB_A_IS_PATTERN
    #define GB_A_IS_PATTERN 0
    #undef  GB_DECLAREA
    #define GB_DECLAREA(aik)                                            \
        GB_void aik [GB_VLA(aik_size)] ;
    #undef  GB_GETA
    #define GB_GETA(aik,Ax,pA,A_iso)                                    \
        if (!A_is_pattern)                                              \
        {                                                               \
            cast_A (aik, Ax +((A_iso) ? 0:((pA)*asize)), asize) ;       \
        }

    // bkj = B(k,j), located in Bx [B_iso ? 0:pB]
    #undef  GB_B_IS_PATTERN
    #define GB_B_IS_PATTERN 0
    #undef  GB_DECLAREB
    #define GB_DECLAREB(bkj)                                            \
        GB_void bkj [GB_VLA(bkj_size)] ;
    #undef  GB_GETB
    #define GB_GETB(bkj,Bx,pB,B_iso)                                    \
        if (!B_is_pattern)                                              \
        {                                                               \
            cast_B (bkj, Bx +((B_iso) ? 0:((pB)*bsize)), bsize) ;       \
        }

    // define t for each task
    #undef  GB_CIJ_DECLARE
    #define GB_CIJ_DECLARE(t) GB_void t [GB_VLA(csize)]

    // address of Cx [p]
    #undef  GB_CX
    #define GB_CX(p) (Cx +((p)*csize))

    // Cx [p] = t
    #undef  GB_CIJ_WRITE
    #define GB_CIJ_WRITE(p,t) memcpy (GB_CX (p), t, csize)

    // address of Hx [i]
    #undef  GB_HX
    #define GB_HX(i) (Hx +((i)*csize))

    // Hx [i] = t
    #undef  GB_HX_WRITE
    #define GB_HX_WRITE(i,t) memcpy (GB_HX (i), t, csize)

    // Cx [p] = Hx [i]
    #undef  GB_CIJ_GATHER
    #define GB_CIJ_GATHER(p,i) memcpy (GB_CX (p), GB_HX(i), csize)

    // Cx [p:p+len=-1] = Hx [i:i+len-1]
    // via memcpy (&(Cx [p]), &(Hx [i]), len*csize)
    #undef  GB_CIJ_MEMCPY
    #define GB_CIJ_MEMCPY(p,i,len) memcpy (GB_CX (p), GB_HX (i), (len)*csize)

    // Cx [p] += Hx [i]
    #undef  GB_CIJ_GATHER_UPDATE
    #define GB_CIJ_GATHER_UPDATE(p,i) fadd (GB_CX (p), GB_CX (p), GB_HX (i))

    // Cx [p] += t
    #undef  GB_CIJ_UPDATE
    #define GB_CIJ_UPDATE(p,t) fadd (GB_CX (p), GB_CX (p), t)

    // Hx [i] += t
    #undef  GB_HX_UPDATE
    #define GB_HX_UPDATE(i,t) fadd (GB_HX (i), GB_HX (i), t)

    // generic types for C and Z
    #undef  GB_C_TYPE
    #define GB_C_TYPE GB_void

    #undef  GB_Z_TYPE
    #define GB_Z_TYPE GB_void

    #undef  GB_C_SIZE
    #define GB_C_SIZE csize

    #if GB_GENERIC_OP_IS_FIRST
    { 
        // t = A(i,k)
        ASSERT (B_is_pattern) ;
        #undef  GB_MULT
        #define GB_MULT(t, aik, bkj, i, k, j) memcpy (t, aik, csize)
        #include "mxm/factory/GB_AxB_saxpy_generic_template.c"
    }
    #elif GB_GENERIC_OP_IS_SECOND
    { 
        // t = B(i,k)
        ASSERT (A_is_pattern) ;
        #undef  GB_MULT
        #define GB_MULT(t, aik, bkj, i, k, j) memcpy (t, bkj, csize)
        #include "mxm/factory/GB_AxB_saxpy_generic_template.c"
    }
    #elif GB_GENERIC_FLIPXY
    { 
        // t = B(k,j) * A(i,k)
        ASSERT (fmult != NULL) ;
        #undef  GB_MULT
        #define GB_MULT(t, aik, bkj, i, k, j) fmult (t, bkj, aik)
        #include "mxm/factory/GB_AxB_saxpy_generic_template.c"
    }
    #elif GB_GENERIC_NOFLIPXY
    { 
        // t = A(i,k) * B(k,j)
        ASSERT (fmult != NULL) ;
        #undef  GB_MULT
        #define GB_MULT(t, aik, bkj, i, k, j) fmult (t, aik, bkj)
        #include "mxm/factory/GB_AxB_saxpy_generic_template.c"
    }
    #elif GB_GENERIC_IDX_FLIPXY
    { 
        // t = B(k,j) * A(i,k)
        ASSERT (fmult_idx != NULL) ;
        const void *theta = mult->theta ;
        #undef  GB_MULT
        #define GB_MULT(t, aik, bkj, i, k, j) \
            fmult_idx (t, bkj, j, k, aik, k, i, theta)
        #include "mxm/factory/GB_AxB_saxpy_generic_template.c"
    }
    #elif GB_GENERIC_IDX_NOFLIPXY
    { 
        // t = A(i,k) * B(k,j)
        ASSERT (fmult_idx != NULL) ;
        const void *theta = mult->theta ;
        #undef  GB_MULT
        #define GB_MULT(t, aik, bkj, i, k, j) \
            fmult_idx (t, aik, i, k, bkj, k, j, theta)
        #include "mxm/factory/GB_AxB_saxpy_generic_template.c"
    }
    #endif

    return (GrB_SUCCESS) ;
}

