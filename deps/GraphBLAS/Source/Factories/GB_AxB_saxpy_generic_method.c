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

//      GB_AxB_saxpy3_generic_firsti64
//      GB_AxB_saxpy3_generic_firstj64
//      GB_AxB_saxpy3_generic_secondj64
//      GB_AxB_saxpy3_generic_firsti32
//      GB_AxB_saxpy3_generic_firstj32
//      GB_AxB_saxpy3_generic_secondj32
//      GB_AxB_saxpy3_generic_first
//      GB_AxB_saxpy3_generic_second
//      GB_AxB_saxpy3_generic_flipped
//      GB_AxB_saxpy3_generic_unflipped

//      GB_AxB_saxbit_generic_firsti64
//      GB_AxB_saxbit_generic_firstj64
//      GB_AxB_saxbit_generic_secondj64
//      GB_AxB_saxbit_generic_firsti32
//      GB_AxB_saxbit_generic_firstj32
//      GB_AxB_saxbit_generic_secondj32
//      GB_AxB_saxbit_generic_first
//      GB_AxB_saxbit_generic_second
//      GB_AxB_saxbit_generic_flipped
//      GB_AxB_saxbit_generic_unflipped

//------------------------------------------------------------------------------

#include "GB_AxB_saxpy.h"
#include "GB_ek_slice.h"
#include "GB_binop.h"
#include "GB_sort.h"
#include "GB_ek_slice_search.c"
#include "GB_bitmap_assign_methods.h"
#include "GB_mxm_shared_definitions.h"
#include "GB_AxB_saxpy_generic.h"
#include "GB_generic.h"

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
    #include "GB_AxB_saxpy3_template.h"

    #if GB_GENERIC_OP_IS_POSITIONAL
    { 

        //----------------------------------------------------------------------
        // generic semirings with positional mulitiply operators
        //----------------------------------------------------------------------

        // C and Z type become int32_t or int64_t

        // C always has type int64_t or int32_t.  The monoid must be used via
        // its function pointer.  The positional multiply operator must be
        // hard-coded since it has no function pointer.  The numerical values
        // and types of A and B are not accessed.

        ASSERT (A_is_pattern) ;
        ASSERT (B_is_pattern) ;

        // aik = A(i,k), located in Ax [A_iso ? 0:pA], value not used
        #undef  GB_A_IS_PATTERN
        #define GB_A_IS_PATTERN 1
        #define GB_DECLAREA(aik)
        #define GB_GETA(aik,Ax,pA,A_iso)

        // bkj = B(k,j), located in Bx [B_iso ? 0:pB], value not used
        #undef  GB_B_IS_PATTERN
        #define GB_B_IS_PATTERN 1
        #define GB_DECLAREB(bkj)
        #define GB_GETB(bkj,Bx,pB,B_iso)

        // define t for each task
        #undef  GB_CIJ_DECLARE
        #define GB_CIJ_DECLARE(t) GB_C_TYPE t

        // address of Cx [p]
        #define GB_CX(p) (&Cx [p])

        // Cx [p] = t
        #undef  GB_CIJ_WRITE
        #define GB_CIJ_WRITE(p,t) Cx [p] = t

        // address of Hx [i]
        #define GB_HX(i) (&Hx [i])

        // Hx [i] = t
        #undef  GB_HX_WRITE
        #define GB_HX_WRITE(i,t) Hx [i] = t

        // Cx [p] = Hx [i]
        #undef  GB_CIJ_GATHER
        #define GB_CIJ_GATHER(p,i) Cx [p] = Hx [i]

        // Cx [p:p+len=-1] = Hx [i:i+len-1]
        // via memcpy (&(Cx [p]), &(Hx [i]), len*csize)
        #undef  GB_CIJ_MEMCPY
        #define GB_CIJ_MEMCPY(p,i,len) memcpy (GB_CX (p), GB_HX (i), (len)*csize)

        // Cx [p] += Hx [i]
        #undef  GB_CIJ_GATHER_UPDATE
        #define GB_CIJ_GATHER_UPDATE(p,i) fadd (GB_CX (p), GB_CX (p), GB_HX (i))

        // Cx [p] += t
        #undef  GB_CIJ_UPDATE
        #define GB_CIJ_UPDATE(p,t) fadd (GB_CX (p), GB_CX (p), &t)

        // Hx [i] += t
        #undef  GB_HX_UPDATE
        #define GB_HX_UPDATE(i,t) fadd (GB_HX (i), GB_HX (i), &t)

        // the original multiplier op may have been flipped, but the offset
        // is unchanged
        int64_t offset = GB_positional_offset (mult->opcode, NULL, NULL) ;

        #if GB_GENERIC_OP_IS_INT64
        {
            #undef  GB_C_TYPE
            #define GB_C_TYPE int64_t
            #undef  GB_Z_TYPE
            #define GB_Z_TYPE int64_t
            // future:: rename GB_C_SIZE to GB_Z_SIZE
            #undef  GB_C_SIZE
            #define GB_C_SIZE (sizeof (int64_t))
            ASSERT (C->type == GrB_INT64) ;
            ASSERT (csize == sizeof (int64_t)) ;
            #if GB_GENERIC_OP_IS_FIRSTI
            { 
                // GB_FIRSTI_binop_code   :   // z = first_i(A(i,k),y) == i
                // GB_FIRSTI1_binop_code  :   // z = first_i1(A(i,k),y) == i+1
                #undef  GB_MULT
                #define GB_MULT(t, aik, bkj, i, k, j) t = i + offset
                #include "GB_AxB_saxpy_generic_template.c"
            }
            #elif GB_GENERIC_OP_IS_FIRSTJ
            { 
                // GB_FIRSTJ_binop_code   :   // z = first_j(A(i,k),y) == k
                // GB_FIRSTJ1_binop_code  :   // z = first_j1(A(i,k),y) == k+1
                // GB_SECONDI_binop_code  :   // z = second_i(x,B(k,j)) == k
                // GB_SECONDI1_binop_code :   // z = second_i1(x,B(k,j))== k+1
                #undef  GB_MULT
                #define GB_MULT(t, aik, bkj, i, k, j) t = k + offset
                #include "GB_AxB_saxpy_generic_template.c"
            }
            #else
            { 
                // GB_SECONDJ_binop_code  :   // z = second_j(x,B(k,j)) == j
                // GB_SECONDJ1_binop_code :   // z = second_j1(x,B(k,j))== j+1
                #undef  GB_MULT
                #define GB_MULT(t, aik, bkj, i, k, j) t = j + offset
                #include "GB_AxB_saxpy_generic_template.c"
            }
            #endif
        }
        #else
        {
            #undef  GB_C_TYPE
            #define GB_C_TYPE int32_t
            #undef  GB_Z_TYPE
            #define GB_Z_TYPE int32_t
            #undef  GB_C_SIZE
            #define GB_C_SIZE (sizeof (int32_t))
            ASSERT (C->type == GrB_INT32) ;
            ASSERT (csize == sizeof (int32_t)) ;
            #if GB_GENERIC_OP_IS_FIRSTI
            { 
                // GB_FIRSTI_binop_code   :   // z = first_i(A(i,k),y) == i
                // GB_FIRSTI1_binop_code  :   // z = first_i1(A(i,k),y) == i+1
                #undef  GB_MULT
                #define GB_MULT(t,aik,bkj,i,k,j) t = (int32_t) (i + offset)
                #include "GB_AxB_saxpy_generic_template.c"
            }
            #elif GB_GENERIC_OP_IS_FIRSTJ
            { 
                // GB_FIRSTJ_binop_code   :   // z = first_j(A(i,k),y) == k
                // GB_FIRSTJ1_binop_code  :   // z = first_j1(A(i,k),y) == k+1
                // GB_SECONDI_binop_code  :   // z = second_i(x,B(k,j)) == k
                // GB_SECONDI1_binop_code :   // z = second_i1(x,B(k,j))== k+1
                #undef  GB_MULT
                #define GB_MULT(t,aik,bkj,i,k,j) t = (int32_t) (k + offset)
                #include "GB_AxB_saxpy_generic_template.c"
            }
            #else
            { 
                // GB_SECONDJ_binop_code  :   // z = second_j(x,B(k,j)) == j
                // GB_SECONDJ1_binop_code :   // z = second_j1(x,B(k,j))== j+1
                #undef  GB_MULT
                #define GB_MULT(t,aik,bkj,i,k,j) t = (int32_t) (j + offset)
                #include "GB_AxB_saxpy_generic_template.c"
            }
            #endif
        }
        #endif

    }
    #else
    {

        //----------------------------------------------------------------------
        // generic semirings with standard multiply operators
        //----------------------------------------------------------------------

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
            #include "GB_AxB_saxpy_generic_template.c"
        }
        #elif GB_GENERIC_OP_IS_SECOND
        { 
            // t = B(i,k)
            ASSERT (A_is_pattern) ;
            #undef  GB_MULT
            #define GB_MULT(t, aik, bkj, i, k, j) memcpy (t, bkj, csize)
            #include "GB_AxB_saxpy_generic_template.c"
        }
        #elif GB_GENERIC_FLIPXY
        { 
            // t = B(k,j) * A(i,k)
            ASSERT (fmult != NULL) ;
            #undef  GB_MULT
            #define GB_MULT(t, aik, bkj, i, k, j) fmult (t, bkj, aik)
            #include "GB_AxB_saxpy_generic_template.c"
        }
        #else
        { 
            // t = A(i,k) * B(k,j)
            ASSERT (fmult != NULL) ;
            #undef  GB_MULT
            #define GB_MULT(t, aik, bkj, i, k, j) fmult (t, aik, bkj)
            #include "GB_AxB_saxpy_generic_template.c"
        }
        #endif
    }
    #endif

    return (GrB_SUCCESS) ;
}

