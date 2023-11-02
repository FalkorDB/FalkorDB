//------------------------------------------------------------------------------
// GB_AxB_saxbit: compute C=A*B, C<M>=A*B, or C<!M>=A*B; C bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_FREE_WORK (&Wf, Wf_size) ;           \
    GB_FREE_WORK (&Wcx, Wcx_size) ;         \
    GB_WERK_POP (H_slice, int64_t) ;        \
    GB_WERK_POP (A_slice, int64_t) ;        \
    GB_WERK_POP (M_ek_slicing, int64_t) ;   \
}

#define GB_FREE_ALL                         \
{                                           \
    GB_FREE_WORKSPACE ;                     \
    GB_phybix_free (C) ;                    \
}

#include "GB_mxm.h"
#include "GB_unused.h"
#include "GB_stringify.h"
#include "GB_AxB_saxpy.h"
#include "GB_binop.h"
#include "GB_ek_slice.h"
#include "GB_AxB_saxpy_generic.h"
#include "GB_AxB__include1.h"
#ifndef GBCOMPACT
#include "GB_AxB__include2.h"
#endif

//------------------------------------------------------------------------------
// GB_AxB_saxbit: compute C=A*B, C<M>=A*B, or C<!M>=A*B
//------------------------------------------------------------------------------

// TODO: also pass in the user's C and the accum operator, and done_in_place,
// like GB_AxB_dot4.

GrB_Info GB_AxB_saxbit        // C = A*B where C is bitmap
(
    GrB_Matrix C,                   // output matrix, static header
    const bool C_iso,               // true if C is iso
    const GB_void *cscalar,         // iso value of C
    const GrB_Matrix M,             // optional mask matrix
    const bool Mask_comp,           // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
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

    ASSERT_MATRIX_OK_OR_NULL (M, "M for bitmap saxpy A*B", GB0) ;
    ASSERT (!GB_PENDING (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_ZOMBIES (M)) ;

    ASSERT_MATRIX_OK (A, "A for bitmap saxpy A*B", GB0) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    ASSERT_MATRIX_OK (B, "B for bitmap saxpy A*B", GB0) ;
    ASSERT (!GB_PENDING (B)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!GB_ZOMBIES (B)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for bitmap saxpy A*B", GB0) ;
    ASSERT (A->vdim == B->vlen) ;

    //--------------------------------------------------------------------------
    // declare workspace
    //--------------------------------------------------------------------------

    int8_t  *restrict Wf  = NULL ; size_t Wf_size = 0 ;
    GB_void *restrict Wcx = NULL ; size_t Wcx_size = 0 ;
    GB_WERK_DECLARE (H_slice, int64_t) ;
    GB_WERK_DECLARE (A_slice, int64_t) ;
    GB_WERK_DECLARE (M_ek_slicing, int64_t) ;

    int M_nthreads = 0 ;
    int M_ntasks = 0 ;

    int nthreads = 0 ;
    int ntasks = 0 ;
    int nfine_tasks_per_vector  = 0 ;
    bool use_coarse_tasks = false ;
    bool use_atomics = false ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // construct C
    //--------------------------------------------------------------------------

    // TODO: If C is the right type on input, and accum is the same as the
    // monoid, then do not create C, but compute in-place instead.

    // Cb is set to all zero.  C->x is malloc'd unless C is iso, in which case
    // it is calloc'ed.

    GrB_Type ctype = semiring->add->op->ztype ;
    int64_t cnzmax = 1 ;
    (void) GB_int64_multiply ((GrB_Index *) (&cnzmax), A->vlen, B->vdim) ;
    // set C->iso = C_iso   OK
    GB_OK (GB_new_bix (&C, // existing header
        ctype, A->vlen, B->vdim, GB_Ap_null, true, GxB_BITMAP, true,
        GB_HYPER_SWITCH_DEFAULT, -1, cnzmax, true, C_iso)) ;
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    ASSERT (mult->ztype == semiring->add->op->ztype) ;
    bool A_is_pattern, B_is_pattern ;
    GB_binop_pattern (&A_is_pattern, &B_is_pattern, flipxy, mult->opcode) ;

    //--------------------------------------------------------------------------
    // slice the M matrix
    //--------------------------------------------------------------------------

    if (M != NULL)
    { 
        GB_SLICE_MATRIX (M, 8) ;
    }

    //--------------------------------------------------------------------------
    // slice the A matrix (if sparse or hyper) and construct the tasks
    //--------------------------------------------------------------------------

    if (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A))
    {

        //----------------------------------------------------------------------
        // slice A if it is sparse or hypersparse
        //----------------------------------------------------------------------

        GB_AxB_saxpy4_tasks (&ntasks, &nthreads, &nfine_tasks_per_vector,
            &use_coarse_tasks, &use_atomics, GB_nnz_held (A), GB_nnz_held (B),
            B->vdim, C->vlen) ;
        if (!use_coarse_tasks)
        {
            // slice the matrix A for each team of fine tasks
            GB_WERK_PUSH (A_slice, nfine_tasks_per_vector + 1, int64_t) ;
            if (A_slice == NULL)
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }
            GB_pslice (A_slice, A->p, A->nvec, nfine_tasks_per_vector, true) ;
        }

        //----------------------------------------------------------------------
        // allocate workspace
        //----------------------------------------------------------------------

        size_t wspace = 0 ;

        if (use_coarse_tasks)
        {

            //------------------------------------------------------------------
            // C<#M> = A*B using coarse tasks where A is sparse/hyper
            //------------------------------------------------------------------

            GB_WERK_PUSH (H_slice, ntasks, int64_t) ;
            if (H_slice == NULL)
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }

            int64_t hwork = 0 ;
            for (int tid = 0 ; tid < ntasks ; tid++)
            { 
                int64_t jstart, jend ;
                GB_PARTITION (jstart, jend, B->vdim, tid, ntasks) ;
                int64_t jtask = jend - jstart ;
                int64_t jpanel = GB_IMIN (jtask, GB_SAXBIT_PANEL_SIZE) ;
                H_slice [tid] = hwork ;
                // bitmap case always needs Hx workspace
                hwork += jpanel ;
            }

            wspace = hwork * C->vlen ;

        }
        else if (!use_atomics)
        { 

            //------------------------------------------------------------------
            // C<#M> = A*B using fine tasks and workspace, with no atomics
            //------------------------------------------------------------------

            // Each fine task is given size-(C->vlen) workspace to compute its
            // result in the first phase, W(:,tid) = A(:,k1:k2) * B(k1:k2,j),
            // where k1:k2 is defined by the fine_tid of the task.  The
            // workspaces are then summed into C in the second phase.

            wspace = (C->vlen) * ntasks ;
        }

        if (wspace > 0)
        {

            //------------------------------------------------------------------
            // allocate Wf and Wcx workspaces
            //------------------------------------------------------------------

            size_t csize = (C_iso) ? 0 : C->type->size ;
            Wf  = GB_MALLOC_WORK (wspace, int8_t, &Wf_size) ;
            Wcx = GB_MALLOC_WORK (wspace * csize, GB_void, &Wcx_size) ;
            if (Wf == NULL || Wcx == NULL)
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // C<#M>=A*B
    //--------------------------------------------------------------------------

    if (C_iso)
    { 

        //----------------------------------------------------------------------
        // via the iso kernel
        //----------------------------------------------------------------------

        GBURBLE ("(iso bitmap saxpy) ") ;
        memcpy (C->x, cscalar, ctype->size) ;
        info = GB (_AsaxbitB__any_pair_iso) (C, M, Mask_comp, Mask_struct,
            A, B, ntasks, nthreads,
            nfine_tasks_per_vector, use_coarse_tasks, use_atomics,
            M_ek_slicing, M_nthreads, M_ntasks, A_slice, H_slice,
            Wcx, Wf) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // via the factory kernel
        //----------------------------------------------------------------------

        info = GrB_NO_VALUE ;
        GBURBLE ("(bitmap saxpy) ") ;

        #ifndef GBCOMPACT
        GB_IF_FACTORY_KERNELS_ENABLED
        { 

            //------------------------------------------------------------------
            // define the worker for the switch factory
            //------------------------------------------------------------------

            #define GB_AsaxbitB(add,mult,xname)  \
                GB (_AsaxbitB_ ## add ## mult ## xname)

            #define GB_AxB_WORKER(add,mult,xname)                           \
            {                                                               \
                info = GB_AsaxbitB (add,mult,xname) (C, M, Mask_comp,       \
                    Mask_struct, A, B, ntasks, nthreads,                    \
                    nfine_tasks_per_vector, use_coarse_tasks, use_atomics,  \
                    M_ek_slicing, M_nthreads, M_ntasks,                     \
                    A_slice, H_slice, Wcx, Wf) ;                            \
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
            info = GB_AxB_saxbit_jit (C, M, Mask_comp,
                Mask_struct, A, B, semiring, flipxy, ntasks, nthreads,
                nfine_tasks_per_vector, use_coarse_tasks, use_atomics,
                M_ek_slicing, M_nthreads, M_ntasks, A_slice, H_slice, Wcx, Wf) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            info = GB_AxB_saxpy_generic (C, M, Mask_comp, Mask_struct,
                true, A, A_is_pattern, B, B_is_pattern, semiring,
                flipxy, GB_SAXPY_METHOD_BITMAP, ntasks, nthreads,
                /* unused: */ NULL, 0, 0, NULL,
                nfine_tasks_per_vector, use_coarse_tasks, use_atomics,
                M_ek_slicing, M_nthreads, M_ntasks,
                A_slice, H_slice, Wcx, Wf) ;
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
    ASSERT_MATRIX_OK (C, "C bitmap saxpy output", GB0) ;
    return (GrB_SUCCESS) ;
}

