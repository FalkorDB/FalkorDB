//------------------------------------------------------------------------------
// GB_AxB_dot3: compute C<M> = A'*B in parallel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This function only computes C<M>=A'*B.  The mask must be present, and not
// complemented, and can be either valued or structural.  The mask is always
// applied.  C and M are both sparse or hypersparse, and have the same sparsity
// structure.

// JIT: done.

#include "GB_mxm.h"
#include "GB_binop.h"
#include "GB_stringify.h"
#include "GB_AxB__include1.h"
#ifndef GBCOMPACT
#include "GB_AxB__include2.h"
#endif
#include "GB_unused.h"

#define GB_FREE_WORKSPACE                       \
{                                               \
    GB_FREE_WORK (&TaskList, TaskList_size) ;   \
}

#define GB_FREE_ALL                             \
{                                               \
    GB_FREE_WORKSPACE ;                         \
    GB_phybix_free (C) ;                        \
}

GrB_Info GB_AxB_dot3                // C<M> = A'*B using dot product method
(
    GrB_Matrix C,                   // output matrix, static header
    const bool C_iso,               // true if C is iso
    const GB_void *cscalar,         // iso value of C
    const GrB_Matrix M,             // mask matrix
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;

    ASSERT_MATRIX_OK (M, "M for dot3 A'*B", GB0) ;
    ASSERT_MATRIX_OK (A, "A for dot3 A'*B", GB0) ;
    ASSERT_MATRIX_OK (B, "B for dot3 A'*B", GB0) ;

    ASSERT (!GB_ZOMBIES (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;    // C is jumbled if M is jumbled
    ASSERT (!GB_PENDING (M)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (!GB_JUMBLED (B)) ;
    ASSERT (!GB_PENDING (B)) ;

    ASSERT (!GB_IS_BITMAP (M)) ;
    ASSERT (!GB_IS_FULL (M)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for numeric A'*B", GB0) ;

    int ntasks, nthreads ;
    GB_task_struct *TaskList = NULL ; size_t TaskList_size = 0 ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    GrB_Monoid add = semiring->add ;
    ASSERT (mult->ztype == add->op->ztype) ;

    bool op_is_first  = mult->opcode == GB_FIRST_binop_code ;
    bool op_is_second = mult->opcode == GB_SECOND_binop_code ;
    bool op_is_pair   = mult->opcode == GB_PAIR_binop_code ;
    bool A_is_pattern = false ;
    bool B_is_pattern = false ;

    if (flipxy)
    { 
        // z = fmult (b,a) will be computed
        A_is_pattern = op_is_first  || op_is_pair ;
        B_is_pattern = op_is_second || op_is_pair ;
        ASSERT (GB_IMPLIES (!A_is_pattern,
            GB_Type_compatible (A->type, mult->ytype))) ;
        ASSERT (GB_IMPLIES (!B_is_pattern,
            GB_Type_compatible (B->type, mult->xtype))) ;
    }
    else
    { 
        // z = fmult (a,b) will be computed
        A_is_pattern = op_is_second || op_is_pair ;
        B_is_pattern = op_is_first  || op_is_pair ;
        ASSERT (GB_IMPLIES (!A_is_pattern,
            GB_Type_compatible (A->type, mult->xtype))) ;
        ASSERT (GB_IMPLIES (!B_is_pattern,
            GB_Type_compatible (B->type, mult->ytype))) ;
    }

    //--------------------------------------------------------------------------
    // get M, A, and B
    //--------------------------------------------------------------------------

    const int64_t *restrict Mp = M->p ;
    const int64_t *restrict Mh = M->h ;
    const int64_t *restrict Mi = M->i ;
    const GB_M_TYPE *restrict Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
    const size_t msize = M->type->size ;
    const int64_t mvlen = M->vlen ;
    const int64_t mvdim = M->vdim ;
    const int64_t mnz = GB_nnz (M) ;
    const int64_t mnvec = M->nvec ;
    const bool M_is_hyper = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_sparse = GB_IS_SPARSE (M) ;

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t vlen = A->vlen ;
    const int64_t anvec = A->nvec ;
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_sparse = GB_IS_SPARSE (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int64_t bnvec = B->nvec ;
    const bool B_is_hyper = GB_IS_HYPERSPARSE (B) ;
    const bool B_is_sparse = GB_IS_SPARSE (B) ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    ASSERT (A->vlen == B->vlen) ;
    ASSERT (vlen > 0) ;

    const int64_t *restrict A_Yp = (A->Y == NULL) ? NULL : A->Y->p ;
    const int64_t *restrict A_Yi = (A->Y == NULL) ? NULL : A->Y->i ;
    const int64_t *restrict A_Yx = (A->Y == NULL) ? NULL : A->Y->x ;
    const int64_t A_hash_bits = (A->Y == NULL) ? 0 : (A->Y->vdim - 1) ;

    const int64_t *restrict B_Yp = (B->Y == NULL) ? NULL : B->Y->p ;
    const int64_t *restrict B_Yi = (B->Y == NULL) ? NULL : B->Y->i ;
    const int64_t *restrict B_Yx = (B->Y == NULL) ? NULL : B->Y->x ;
    const int64_t B_hash_bits = (B->Y == NULL) ? 0 : (B->Y->vdim - 1) ;

    //--------------------------------------------------------------------------
    // allocate C, the same size and # of entries as M
    //--------------------------------------------------------------------------

    GrB_Type ctype = add->op->ztype ;
    int64_t cvlen = mvlen ;
    int64_t cvdim = mvdim ;
    int64_t cnz = mnz ;
    int64_t cnvec = mnvec ;
    int C_sparsity = (M_is_hyper) ? GxB_HYPERSPARSE : GxB_SPARSE ;

    // C is sparse or hypersparse, not full or bitmap
    // set C->iso = C_iso   OK
    GB_OK (GB_new_bix (&C, // sparse or hyper (from M), existing header
        ctype, cvlen, cvdim, GB_Ap_malloc, true,
        C_sparsity, true, M->hyper_switch, cnvec,
        cnz+1,  // add one to cnz for GB_cumsum of Cwork in GB_AxB_dot3_slice
        true, C_iso)) ;

    int64_t *restrict Cp = C->p ;
    int64_t *restrict Ch = C->h ;
    int64_t *restrict Cwork = C->i ;    // use C->i as workspace

    //--------------------------------------------------------------------------
    // determine the # of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // copy Mp and Mh into C
    //--------------------------------------------------------------------------

    // M is sparse or hypersparse; C is the same as M
    nthreads = GB_nthreads (cnvec, chunk, nthreads_max) ;
    // TODO: try this with Cp and Ch shallow
    GB_memcpy (Cp, Mp, (cnvec+1) * sizeof (int64_t), nthreads) ;
    if (M_is_hyper)
    { 
        GB_memcpy (Ch, Mh, cnvec * sizeof (int64_t), nthreads) ;
    }
    C->nvec_nonempty = M->nvec_nonempty ;
    C->nvec = M->nvec ;
    C->nvals = M->nvals ;
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // construct the tasks for the first phase
    //--------------------------------------------------------------------------

    nthreads = GB_nthreads (cnz, chunk, nthreads_max) ;
    GB_OK (GB_AxB_dot3_one_slice (&TaskList, &TaskList_size, &ntasks, &nthreads,
        M, Werk)) ;

    //--------------------------------------------------------------------------
    // phase1: estimate the work to compute each entry in C
    //--------------------------------------------------------------------------

    // The work to compute C(i,j) is held in Cwork [p], if C(i,j) appears in
    // as the pth entry in C.  This phase is purely symbolic and does not
    // depend on the data types or semiring.

    #include "GB_mxm_shared_definitions.h"
    #define GB_DOT3
    #define GB_DOT3_PHASE1

    if (M_is_sparse && Mask_struct)
    { 
        // special case: M is present, sparse, structural, and not complemented
        #define GB_MASK_SPARSE_STRUCTURAL_AND_NOT_COMPLEMENTED
        #include "GB_meta16_factory.c"
        #undef  GB_MASK_SPARSE_STRUCTURAL_AND_NOT_COMPLEMENTED
        // TODO: skip phase1 if A and B are both bitmap/full.
    }
    else
    { 
        // general case: M sparse/hyper, structural/valued
        #include "GB_meta16_factory.c"
    }

    #undef GB_DOT3
    #undef GB_DOT3_PHASE1

    //--------------------------------------------------------------------------
    // free the current tasks and construct the tasks for the second phase
    //--------------------------------------------------------------------------

    GB_FREE_WORK (&TaskList, TaskList_size) ;
    GB_OK (GB_AxB_dot3_slice (&TaskList, &TaskList_size, &ntasks, &nthreads,
        C, Werk)) ;

    GBURBLE ("nthreads %d ntasks %d ", nthreads, ntasks) ;

    //--------------------------------------------------------------------------
    // C<M> = A'*B, via masked dot product method and built-in semiring
    //--------------------------------------------------------------------------

    if (C_iso)
    { 

        //----------------------------------------------------------------------
        // via the iso kernel
        //----------------------------------------------------------------------

        memcpy (C->x, cscalar, ctype->size) ;
        info = GB (_Adot3B__any_pair_iso) (C, M, Mask_struct, A, B,
            TaskList, ntasks, nthreads) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // via the factory kernel
        //----------------------------------------------------------------------

        info = GrB_NO_VALUE ;
        #ifndef GBCOMPACT
        GB_IF_FACTORY_KERNELS_ENABLED
        { 

            //------------------------------------------------------------------
            // define the worker for the switch factory
            //------------------------------------------------------------------

            #define GB_Adot3B(add,mult,xname) \
                GB (_Adot3B_ ## add ## mult ## xname)

            #define GB_AxB_WORKER(add,mult,xname)                           \
            {                                                               \
                info = GB_Adot3B (add,mult,xname) (C, M, Mask_struct, A, B, \
                    TaskList, ntasks, nthreads) ;                           \
            }                                                               \
            break ;

            //------------------------------------------------------------------
            // launch the switch factory
            //------------------------------------------------------------------

            GB_Opcode mult_binop_code, add_binop_code ;
            GB_Type_code xcode, ycode, zcode ;
            if (GB_AxB_semiring_builtin (A, A_is_pattern, B, B_is_pattern,
                semiring, flipxy, &mult_binop_code, &add_binop_code, &xcode,
                &ycode, &zcode))
            { 
                #include "GB_AxB_factory.c"
            }
        }
        #endif

        //----------------------------------------------------------------------
        // via the JIT or PreJIT kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            info = GB_AxB_dot3_jit (C, M, Mask_struct, A, B,
                semiring, flipxy, TaskList, ntasks, nthreads) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            #define GB_DOT3_GENERIC
            GB_BURBLE_MATRIX (C, "(generic C<M>=A'*B) ") ;
            #include "GB_AxB_dot_generic.c"
            info = GrB_SUCCESS ;
        }
    }

    if (info != GrB_SUCCESS)
    { 
        // out of memory, or other error
        GB_FREE_ALL ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    C->jumbled = GB_JUMBLED (M) ;   // C is jumbled if M is jumbled
    ASSERT_MATRIX_OK (C, "dot3: C<M> = A'*B output", GB0) ;
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    return (GrB_SUCCESS) ;
}

