//------------------------------------------------------------------------------
// GB_unjumble: unjumble the vectors of a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "sort/GB_sort.h"
#include "unaryop/GB_unop.h"
#include "jitifyer/GB_stringify.h"

#define GB_FREE_ALL GB_WERK_POP (A_slice, int64_t) ;

GrB_Info GB_unjumble        // unjumble a matrix
(
    GrB_Matrix A,           // matrix to unjumble
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (A, "A to unjumble", GB0) ;
    ASSERT (!GB_ZOMBIES (A)) ;      // zombies must be killed first
    ASSERT (GB_PENDING_OK (A)) ;    // pending tuples are not modified

    if (A->nvec_nonempty < 0)
    { 
        A->nvec_nonempty = GB_nvec_nonempty (A) ;
    }

    if (!A->jumbled)
    { 
        // nothing to do
        return (GrB_SUCCESS) ;
    }

    // full and bitmap matrices are never jumbled
    ASSERT (!GB_IS_FULL (A)) ;
    ASSERT (!GB_IS_BITMAP (A)) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const int64_t anvec = A->nvec ;
    const int64_t anz = GB_nnz (A) ;
    const int64_t *restrict Ap = A->p ;
    int64_t *restrict Ai = A->i ;
    const size_t asize = (A->iso) ? 0 : A->type->size ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (anz + anvec, chunk, nthreads_max) ;
    int ntasks = (nthreads == 1) ? 1 : (32 * nthreads) ;
    ntasks = GB_IMIN (ntasks, anvec) ;
    ntasks = GB_IMAX (ntasks, 1) ;

    //--------------------------------------------------------------------------
    // slice the work
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (A_slice, int64_t) ;
    GB_WERK_PUSH (A_slice, ntasks + 1, int64_t) ;
    if (A_slice == NULL)
    { 
        // out of memory
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_p_slice (A_slice, Ap, anvec, ntasks, false) ;

    //--------------------------------------------------------------------------
    // sort the vectors
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_NO_VALUE ;

    if (asize == 0)
    { 

        //----------------------------------------------------------------------
        // iso matrices of any type; only sort the pattern
        //----------------------------------------------------------------------

        #define GB_QSORT GB_qsort_1 (Ai+pA_start, aknz) ;
        #include "wait/template/GB_unjumbled_template.c"
        info = GrB_SUCCESS ;
    }
    else
    { 

        //----------------------------------------------------------------------
        // factory kernels for non-iso matrices
        //----------------------------------------------------------------------

        #ifndef GBCOMPACT
        GB_IF_FACTORY_KERNELS_ENABLED
        {
            switch (asize)
            {

                case GB_1BYTE : // bool, uint8, int8, and user types of size 1
                {
                    uint8_t *Ax = (uint8_t *) A->x ;
                    #define GB_QSORT \
                        GB_qsort_1b_size1 (Ai+pA_start, Ax+pA_start, aknz) ;
                    #include "wait/template/GB_unjumbled_template.c"
                    info = GrB_SUCCESS ;
                }
                break ;

                case GB_2BYTE : // uint16, int16, and user types of size 2
                {
                    uint16_t *Ax = (uint16_t *) A->x ;
                    #define GB_QSORT \
                        GB_qsort_1b_size2 (Ai+pA_start, Ax+pA_start, aknz) ;
                    #include "wait/template/GB_unjumbled_template.c"
                    info = GrB_SUCCESS ;
                }
                break ;

                case GB_4BYTE : // uint32, int32, float, and 4-byte user
                {
                    uint32_t *Ax = (uint32_t *) A->x ;
                    #define GB_QSORT \
                        GB_qsort_1b_size4 (Ai+pA_start, Ax+pA_start, aknz) ;
                    #include "wait/template/GB_unjumbled_template.c"
                    info = GrB_SUCCESS ;
                }
                break ;

                case GB_8BYTE : // uint64, int64, double, float complex,
                                // and 8-byte user-defined types
                {
                    uint64_t *Ax = (uint64_t *) A->x ;
                    #define GB_QSORT \
                        GB_qsort_1b_size8 (Ai+pA_start, Ax+pA_start, aknz) ;
                    #include "wait/template/GB_unjumbled_template.c"
                    info = GrB_SUCCESS ;
                }
                break ;

                case GB_16BYTE : // double complex, and user types of size 16
                {
                    GB_blob16 *Ax = (GB_blob16 *) A->x ;
                    #define GB_QSORT \
                        GB_qsort_1b_size16 (Ai+pA_start, Ax+pA_start, aknz) ;
                    #include "wait/template/GB_unjumbled_template.c"
                    info = GrB_SUCCESS ;
                }
                break ;

                default:;
            }
        }
        #endif
    }

    //--------------------------------------------------------------------------
    // via the JIT kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        GBURBLE ("(unjumble: jit kernel) ") ;
        struct GB_UnaryOp_opaque op_header ;
        GB_Operator op = GB_unop_identity (A->type, &op_header) ;
        info = GB_unjumble_jit (A, op, A_slice, ntasks, nthreads) ;
    }

    //--------------------------------------------------------------------------
    // via the generic kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        GBURBLE ("(unjumble: generic kernel) ") ;
        GB_void *Ax = (GB_void *) A->x ;
        #define GB_QSORT \
            GB_qsort_1b (Ai+pA_start, Ax+pA_start*asize, asize, aknz) ;
        #include "wait/template/GB_unjumbled_template.c"
        info = GrB_SUCCESS ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    if (info == GrB_SUCCESS)
    { 
        A->jumbled = false ;        // A has been unjumbled
        ASSERT_MATRIX_OK (A, "A unjumbled", GB0) ;
        ASSERT (A->nvec_nonempty >= 0)
    }
    return (info) ;
}

