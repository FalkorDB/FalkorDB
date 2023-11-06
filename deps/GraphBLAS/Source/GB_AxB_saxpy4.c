//------------------------------------------------------------------------------
// GB_AxB_saxpy4: compute C+=A*B: C full, A sparse/hyper, B bitmap/full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_AxB_saxpy4 computes C+=A*B where C is as-if-full, A is
// sparse/hypersparse, and B is bitmap/full (or as-if-full).  No mask is
// present, C_replace is false, the accum matches the monoid, no typecasting is
// needed, and no user-defined types or operators are used.

// The ANY monoid is not supported, since its use as accum would be unusual.
// The monoid must have an atomic implementation, so the TIMES monoid for
// complex types is not supported.

// JIT: done.

//------------------------------------------------------------------------------

#include "GB_mxm.h"
#include "GB_control.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_AxB__include2.h"
#endif

#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_WERK_POP (A_slice, int64_t) ;        \
    GB_WERK_POP (H_slice, int64_t) ;        \
    GB_FREE_WORK (&Wf, Wf_size) ;           \
    GB_FREE_WORK (&Wcx, Wcx_size) ;         \
}

#define GB_FREE_ALL                         \
{                                           \
    GB_FREE_WORKSPACE ;                     \
    GB_phybix_free (C) ;                    \
}

//------------------------------------------------------------------------------
// GB_AxB_saxpy4: compute C+=A*B: C full, A sparse/hyper, B bitmap/full
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxpy4              // C += A*B
(
    GrB_Matrix C,                   // users input/output matrix
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B and accum
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *done_in_place,            // if true, saxpy4 has computed the result
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_WERK_DECLARE (A_slice, int64_t) ;
    GB_WERK_DECLARE (H_slice, int64_t) ;
    GB_void *restrict Wcx= NULL ; size_t Wcx_size = 0 ;
    int8_t  *restrict Wf = NULL ; size_t Wf_size  = 0 ;

    ASSERT_MATRIX_OK (C, "C for saxpy4 C+=A*B", GB0) ;
    ASSERT (GB_IS_FULL (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_ZOMBIES (C)) ;

    ASSERT_MATRIX_OK (A, "A for saxpy4 C+=A*B", GB0) ;
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    ASSERT_MATRIX_OK (B, "B for saxpy4 C+=A*B", GB0) ;
    ASSERT (GB_IS_BITMAP (B) || GB_IS_FULL (B)) ;
    ASSERT (!GB_PENDING (B)) ;
    ASSERT (!GB_JUMBLED (B)) ;
    ASSERT (!GB_ZOMBIES (B)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for saxpy4 C+=A*B", GB0) ;
    ASSERT (A->vdim == B->vlen) ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    ASSERT (mult->ztype == semiring->add->op->ztype) ;
    bool A_is_pattern, B_is_pattern ;
    GB_binop_pattern (&A_is_pattern, &B_is_pattern, flipxy, mult->opcode) ;

    GB_Opcode mult_binop_code, add_binop_code ;
    GB_Type_code xcode, ycode, zcode ;
    bool builtin_semiring = GB_AxB_semiring_builtin (A, A_is_pattern, B,
        B_is_pattern, semiring, flipxy, &mult_binop_code, &add_binop_code,
        &xcode, &ycode, &zcode) ;

    if (add_binop_code == GB_ANY_binop_code)
    { 
        // The semiring cannot use the ANY monoid.
        // The semiring must be builtin, or use the JIT (no generic method).
        GBURBLE ("(punt) ") ;
        return (GrB_NO_VALUE) ;
    }

    // the complex TIMES and ANY monoids do not have an atomic update
    bool z_has_no_atomic_update = (zcode >= GB_FC32_code) &&
        (add_binop_code == GB_TIMES_binop_code) ;

    GBURBLE ("(saxpy4: %s += %s*%s) ",
            GB_sparsity_char_matrix (C),
            GB_sparsity_char_matrix (A),
            GB_sparsity_char_matrix (B)) ;

    //--------------------------------------------------------------------------
    // ensure C is non-iso
    //--------------------------------------------------------------------------

    GB_OK (GB_convert_any_to_non_iso (C, true)) ;

    //--------------------------------------------------------------------------
    // determine the # of threads to use and the parallel tasks
    //--------------------------------------------------------------------------

    int nthreads, ntasks, nfine_tasks_per_vector ;
    bool use_coarse_tasks, use_atomics ;
    GB_AxB_saxpy4_tasks (&ntasks, &nthreads, &nfine_tasks_per_vector,
        &use_coarse_tasks, &use_atomics, GB_nnz (A), GB_nnz_held (B),
        B->vdim, C->vlen) ;

    //--------------------------------------------------------------------------
    // allocate workspace and slice A
    //--------------------------------------------------------------------------

    size_t wspace = 0 ;

    if (use_coarse_tasks)
    {

        //----------------------------------------------------------------------
        // allocate workspace for coarse tasks
        //----------------------------------------------------------------------

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
            int64_t jpanel = GB_IMIN (jtask, GB_SAXPY4_PANEL_SIZE) ;
            H_slice [tid] = hwork ;
            // full case needs Hx workspace only if jpanel > 1
            if (jpanel > 1)
            { 
                hwork += jpanel ;
            }
        }

        wspace = hwork * C->vlen * (C->type->size) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // allocate workspace for fine tasks (both atomic and non-atomic)
        //----------------------------------------------------------------------

        // slice A for each team of fine tasks (atomic and non-atomic)
        GB_WERK_PUSH (A_slice, nfine_tasks_per_vector + 1, int64_t) ;
        if (A_slice == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        GB_pslice (A_slice, A->p, A->nvec, nfine_tasks_per_vector, true) ;

        if (!use_atomics)
        { 
            // Each non-atomic fine task is given size-cvlen workspace to
            // compute its result in the first phase, W(:,tid) = A(:,k1:k2) *
            // B(k1:k2,j), where k1:k2 is defined by the fine_tid of the task.
            // The workspaces are then summed into C in the second phase.
            // Atomic fine takes do not require any Wcx workspace; they
            // use just A_slice.
            wspace = (C->vlen) * ntasks * (C->type->size) ;
        }
        else if (z_has_no_atomic_update)
        { 
            // The atomic fine tasks use the monoid's atomic update, which is
            // available for most factory kernels.  The TIMES monoid for the
            // complex types (FC32 and FC64) requires a critical section for
            // each C(i,j) scalar. User-defined monoids for JIT kernels also
            // require this mutex.
            Wf = GB_CALLOC_WORK (C->vlen * C->vdim, int8_t, &Wf_size) ;
            if (Wf == NULL)
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }
        }
    }

    if (wspace > 0)
    {
        Wcx = GB_MALLOC_WORK (wspace, GB_void, &Wcx_size) ;
        if (Wcx == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
    }

    //--------------------------------------------------------------------------
    // via the factory kernel
    //--------------------------------------------------------------------------

    info = GrB_NO_VALUE ;
    #ifndef GBCOMPACT
    GB_IF_FACTORY_KERNELS_ENABLED
    { 

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_Asaxpy4B(add,mult,xname) \
            GB (_Asaxpy4B_ ## add ## mult ## xname)
        #define GB_AxB_WORKER(add,mult,xname)                               \
        {                                                                   \
            info = GB_Asaxpy4B (add,mult,xname) (C, A, B, ntasks, nthreads, \
                nfine_tasks_per_vector, use_coarse_tasks, use_atomics,      \
                A_slice, H_slice, Wcx, Wf) ;                                \
        }                                                                   \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        // disabled the ANY monoid
        #define GB_NO_ANY_MONOID
        if (builtin_semiring)
        {
            #include "GB_AxB_factory.c"
        }

    }
    #endif

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        info = GB_AxB_saxpy4_jit (C, A, B, semiring, flipxy,
            ntasks, nthreads, nfine_tasks_per_vector, use_coarse_tasks,
            use_atomics, A_slice, H_slice, Wcx, Wf) ;
    }


    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    if (info == GrB_NO_VALUE)
    { 
        // saxpy4 doesn't handle this case; punt to saxpy3, bitmap saxpy, etc
        GBURBLE ("(punt) ") ;
    }
    else if (info == GrB_SUCCESS)
    { 
        ASSERT_MATRIX_OK (C, "saxpy4: output", GB0) ;
        (*done_in_place) = true ;
    }
    else
    { 
        // out of memory, or other error
        GB_FREE_ALL ;
    }
    return (info) ;
}

