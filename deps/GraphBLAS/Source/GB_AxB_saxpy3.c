//------------------------------------------------------------------------------
// GB_AxB_saxpy3: compute C=A*B, C<M>=A*B, or C<!M>=A*B in parallel
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: done.

// GB_AxB_saxpy3 computes C=A*B, C<M>=A*B, or C<!M>=A*B in parallel.  If the
// mask matrix M has too many entries compared to the work to compute A*B, then
// it is not applied.  Instead, M is ignored and C=A*B is computed.  The mask
// is applied later, in GB_mxm.

// C is sparse or hypersparse.  M, A, and B can have any format.
// The accum operator is not handled, and C is not modified in-place.  Instead,
// C is constructed in a static header.

// For simplicity, this discussion and all comments in this code assume that
// all matrices are in CSC format, but the algorithm is CSR/CSC agnostic.

// The matrix B is split into two kinds of tasks: coarse and fine.  A coarse
// task computes C(:,j1:j2) = A*B(:,j1:j2), for a unique set of vectors j1:j2.
// Those vectors are not shared with any other tasks.  A fine task works with a
// team of other fine tasks to compute C(:,j) for a single vector j.  Each fine
// task computes A*B(k1:k2,j) for a unique range k1:k2, and sums its results
// into C(:,j) via atomic operations.

// Each coarse or fine task uses either Gustavson's method [1] or the Hash
// method [2].  There are 4 kinds of tasks:

//      fine Gustavson task
//      fine hash task
//      coarse Gustason task
//      coarse hash task

// Each of the 4 kinds tasks are then subdivided into 3 variants, for C=A*B,
// C<M>=A*B, and C<!M>=A*B, giving a total of 12 different types of tasks.

// Fine tasks are used when there would otherwise be too much work for a single
// task to compute the single vector C(:,j).  Fine tasks share all of their
// workspace with the team of fine tasks computing C(:,j).  Coarse tasks are
// prefered since they require less synchronization, but fine tasks allow for
// better parallelization when B has only a few vectors.  If B consists of a
// single vector (for GrB_mxv if A is in CSC format and not transposed, or
// for GrB_vxm if A is in CSR format and not transpose), then the only way to
// get parallelism is via fine tasks.  If a single thread is used for this
// case, a single-vector coarse task is used.

// To select between the Hash method or Gustavson's method for each task, the
// hash table size is first found.  The hash table size for a hash task depends
// on the maximum flop count for any vector in that task (which is just one
// vector for the fine tasks).  It is set to twice the smallest power of 2 that
// is greater than the flop count to compute that vector (plus the # of entries
// in M(:,j) for tasks that compute C<M>=A*B or C<!M>=A*B).  This size ensures
// the results will fit in the hash table, and with ideally only a modest
// number of collisions.  If the hash table size exceeds a threshold (currently
// m/16 if C is m-by-n), then Gustavson's method is used instead, and the hash
// table size is set to m, to serve as the gather/scatter workspace for
// Gustavson's method.

// The workspace allocated depends on the type of task.  Let s be the hash
// table size for the task, and C is m-by-n (assuming all matrices are CSC; if
// CSR, then m is replaced with n).
//
//      fine Gustavson task (shared):   int8_t  Hf [m] ; ctype Hx [m] ;
//      fine hash task (shared):        int64_t Hf [s] ; ctype Hx [s] ;
//      coarse Gustavson task:          int64_t Hf [m] ; ctype Hx [m] ;
//      coarse hash task:               int64_t Hf [s] ; ctype Hx [s] ;
//                                      int64_t Hi [s] ; 
//
// Note that the Hi array is needed only for the coarse hash task.  Additional
// workspace is allocated to construct the list of tasks, but this is freed
// before C is constructed.

// References:

// [1] Fred G. Gustavson. 1978. Two Fast Algorithms for Sparse Matrices:
// Multiplication and Permuted Transposition. ACM Trans. Math. Softw.  4, 3
// (Sept. 1978), 250–269. DOI:https://doi.org/10.1145/355791.355796

// [2] Yusuke Nagasaka, Satoshi Matsuoka, Ariful Azad, and Aydin Buluc. 2018.
// High-Performance Sparse Matrix-Matrix Products on Intel KNL and Multicore
// Architectures. In Proc. 47th Intl. Conf. on Parallel Processing (ICPP '18).
// Association for Computing Machinery, New York, NY, USA, Article 34, 1–10.
// DOI:https://doi.org/10.1145/3229710.3229720

//------------------------------------------------------------------------------

#include "GB_mxm.h"
#include "GB_stringify.h"
#include "GB_AxB_saxpy_generic.h"
#include "GB_control.h"
#include "GB_AxB__include1.h"
#ifndef GBCOMPACT
#include "GB_AxB__include2.h"
#endif
#include "GB_unused.h"

#define GB_FREE_WORKSPACE                           \
{                                                   \
    GB_FREE_WORK (&SaxpyTasks, SaxpyTasks_size) ;   \
    GB_FREE_WORK (&Hi_all, Hi_all_size) ;           \
    GB_FREE_WORK (&Hf_all, Hf_all_size) ;           \
    GB_FREE_WORK (&Hx_all, Hx_all_size) ;           \
}

#define GB_FREE_ALL             \
{                               \
    GB_FREE_WORKSPACE ;         \
    GB_phybix_free (C) ;        \
}

//------------------------------------------------------------------------------
// GB_AxB_saxpy3: compute C=A*B, C<M>=A*B, or C<!M>=A*B in parallel
//------------------------------------------------------------------------------

GrB_Info GB_AxB_saxpy3              // C = A*B using Gustavson+Hash
(
    GrB_Matrix C,                   // output, static header, not in-place
    const bool C_iso,               // true if C is iso
    const GB_void *cscalar,         // iso value of C
    int C_sparsity,                 // construct C as sparse or hypersparse
    const GrB_Matrix M_input,       // optional mask matrix
    const bool Mask_comp_input,     // if true, use !M
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_Matrix A,             // input matrix A
    const GrB_Matrix B,             // input matrix B
    const GrB_Semiring semiring,    // semiring that defines C=A*B
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *mask_applied,             // if true, then mask was applied
    GrB_Desc_Value AxB_method,      // Default, Gustavson, or Hash
    const int do_sort,              // if nonzero, try to sort in saxpy3
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    GrB_Matrix M = M_input ;        // use the mask M, until deciding otherwise
    bool Mask_comp = Mask_comp_input ;

    (*mask_applied) = false ;
    bool apply_mask = false ;

    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;

    ASSERT_MATRIX_OK_OR_NULL (M, "M for saxpy3 A*B", GB0) ;
    ASSERT (!GB_PENDING (M)) ;
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (!GB_ZOMBIES (M)) ;

    ASSERT_MATRIX_OK (A, "A for saxpy3 A*B", GB0) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;

    ASSERT_MATRIX_OK (B, "B for saxpy3 A*B", GB0) ;
    ASSERT (!GB_PENDING (B)) ;
    ASSERT (GB_JUMBLED_OK (B)) ;
    ASSERT (!GB_ZOMBIES (B)) ;

    ASSERT_SEMIRING_OK (semiring, "semiring for saxpy3 A*B", GB0) ;
    ASSERT (A->vdim == B->vlen) ;

    ASSERT (C_sparsity == GxB_HYPERSPARSE || C_sparsity == GxB_SPARSE) ;

    //--------------------------------------------------------------------------
    // determine the # of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // define workspace
    //--------------------------------------------------------------------------

    int64_t *restrict Hi_all = NULL ; size_t Hi_all_size = 0 ;
    int64_t *restrict Hf_all = NULL ; size_t Hf_all_size = 0 ;
    GB_void *restrict Hx_all = NULL ; size_t Hx_all_size = 0 ;
    GB_saxpy3task_struct *SaxpyTasks = NULL ; size_t SaxpyTasks_size = 0 ;

    //--------------------------------------------------------------------------
    // construct the hyper hashes for M and A
    //--------------------------------------------------------------------------

    GB_OK (GB_hyper_hash_build (M, Werk)) ;    // does nothing if M is NULL
    GB_OK (GB_hyper_hash_build (A, Werk)) ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    GrB_Monoid add = semiring->add ;
    ASSERT (mult->ztype == add->op->ztype) ;
    bool A_is_pattern, B_is_pattern ;
    GB_binop_pattern (&A_is_pattern, &B_is_pattern, flipxy, mult->opcode) ;

    GB_Opcode mult_binop_code, add_binop_code ;
    GB_Type_code xcode, ycode, zcode ;
    bool builtin_semiring = GB_AxB_semiring_builtin (A, A_is_pattern, B,
        B_is_pattern, semiring, flipxy, &mult_binop_code, &add_binop_code,
        &xcode, &ycode, &zcode) ;

    //--------------------------------------------------------------------------
    // get A, and B
    //--------------------------------------------------------------------------

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int64_t avlen = A->vlen ;
    const int64_t anvec = A->nvec ;
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;

    const int64_t *restrict Bp = B->p ;
    const int64_t *restrict Bh = B->h ;
    const int8_t  *restrict Bb = B->b ;
    const int64_t *restrict Bi = B->i ;
    const int64_t bvdim = B->vdim ;
    const int64_t bnz = GB_nnz_held (B) ;
    const int64_t bnvec = B->nvec ;
    const int64_t bvlen = B->vlen ;
    const bool B_is_hyper = GB_IS_HYPERSPARSE (B) ;

    //--------------------------------------------------------------------------
    // allocate C (just C->p and C->h, but not C->i or C->x)
    //--------------------------------------------------------------------------

    GrB_Type ctype = add->op->ztype ;
    size_t csize = ctype->size ;
    int64_t cvlen = avlen ;
    int64_t cvdim = bvdim ;
    int64_t cnvec = bnvec ;

    info = GB_new (&C, // sparse or hyper, existing header
        ctype, cvlen, cvdim, GB_Ap_malloc, true,
        C_sparsity, B->hyper_switch, cnvec) ;
    if (info != GrB_SUCCESS)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (info) ;
    }

    C->iso = C_iso ;    // OK

    int64_t *restrict Cp = C->p ;
    int64_t *restrict Ch = C->h ;
    if (B_is_hyper)
    { 
        // B and C are both hypersparse
        ASSERT (C_sparsity == GxB_HYPERSPARSE) ;
        int nth = GB_nthreads (cnvec, chunk, nthreads_max) ;
        GB_memcpy (Ch, Bh, cnvec * sizeof (int64_t), nth) ;
        C->nvec = bnvec ;
    }
    else
    { 
        // B is sparse, bitmap, or full; C is sparse
        ASSERT (C_sparsity == GxB_SPARSE) ;
    }

    //==========================================================================
    // phase0: create parallel tasks and allocate workspace
    //==========================================================================

    int nthreads, ntasks, nfine ;
    bool M_in_place = false ;

    if (nthreads_max == 1 && M == NULL && (AxB_method != GxB_AxB_HASH) &&
        GB_IMIN (GB_nnz (A), GB_nnz (B)) > cvlen/16)
    { 
        // Skip the flopcount analysis if only a single thread is being used,
        // no mask is present, the Hash method is not explicitly selected, and
        // the problem is not extremely sparse.  In this case, use a single
        // coarse Gustavson task only.  In this case, the flop count analysis
        // is not needed.
        GBURBLE ("(single-threaded Gustavson) ") ;
        info = GB_AxB_saxpy3_slice_quick (C, A, B,
            &SaxpyTasks, &SaxpyTasks_size, &ntasks, &nfine, &nthreads,
            Werk) ;
    }
    else
    { 
        // Do the flopcount analysis and create a set of well-balanced tasks in
        // the general case.  This may select a single task for a single thread
        // anyway, but this decision would be based on the analysis.
        info = GB_AxB_saxpy3_slice_balanced (C, M, Mask_comp, A, B, AxB_method,
            builtin_semiring,
            &SaxpyTasks, &SaxpyTasks_size, &apply_mask, &M_in_place,
            &ntasks, &nfine, &nthreads, Werk) ;
    }

    if (info == GrB_NO_VALUE)
    { 
        // The mask is present but has been discarded; need to discard the
        // analysis so far and redo it without the mask.  This may result in
        // GB_AxB_saxbit being called instead of GB_AxB_saxpy3.
        ASSERT (M != NULL && !apply_mask) ;
        GB_FREE_ALL ;
        return (GrB_NO_VALUE) ;
    }
    else if (info != GrB_SUCCESS)
    { 
        // out of memory or other error
        GB_FREE_ALL ;
        return (info) ;
    }

    if (!apply_mask)
    { 
        // disable the mask, if present.
        M = NULL ;
        Mask_comp = false ;
    }

    if (do_sort) GBURBLE ("sort ") ;

    //--------------------------------------------------------------------------
    // allocate the hash tables
    //--------------------------------------------------------------------------

    // If Gustavson's method is used (coarse tasks):
    //
    //      hash_size is cvlen.
    //      Hi is not allocated.
    //      Hf and Hx are both of size hash_size.
    //
    //      (Hf [i] == mark) is true if i is in the hash table.
    //      Hx [i] is the value of C(i,j) during the numeric phase.
    //
    //      Gustavson's method is used if the hash_size for the Hash method
    //      is a significant fraction of cvlen. 
    //
    // If the Hash method is used (coarse tasks):
    //
    //      hash_size is 2 times the smallest power of 2 that is larger than
    //      the # of flops required for any column C(:,j) being computed.  This
    //      ensures that all entries have space in the hash table, and that the
    //      hash occupancy will never be more than 50%.  It is always smaller
    //      than cvlen (otherwise, Gustavson's method is used).
    //
    //      A hash function is used for the ith entry:
    //          hash = GB_HASHF (i, hash_bits) ; in range 0 to hash_size-1
    //      If a collision occurs, linear probing is used:
    //          GB_REHASH (hash, i, hash_bits)
    //      which is:
    //          hash = (hash + 1) & (hash_size-1)
    //      where hash_bits = hash_size - 1
    //
    //      (Hf [hash] == mark) is true if the position is occupied.
    //      i = Hi [hash] gives the row index i that occupies that position.
    //      Hx [hash] is the value of C(i,j) during the numeric phase.
    //
    // For both coarse methods:
    //
    //      Hf starts out all zero (via calloc), and mark starts out as 1.  To
    //      clear Hf, mark is incremented, so that all entries in Hf are not
    //      equal to mark.

    // add some padding to the end of each hash table, to avoid false
    // sharing of cache lines between the hash tables.  But only add the
    // padding if there is more than one team.
    size_t hx_pad = 0 ;
    size_t hi_pad = 0 ;
    for (int taskid = 1 ; taskid < ntasks ; taskid++)
    {
        if (taskid == SaxpyTasks [taskid].leader)
        { 
            hx_pad = GB_ICEIL (64, csize) ;
            hi_pad = 64 / sizeof (int64_t) ;
            break ;
        }
    }

    size_t Hi_size_total = 0 ;
    size_t Hf_size_total = 0 ;
    size_t Hx_size_total = 0 ;

    //--------------------------------------------------------------------------
    // determine the total size of all hash tables
    //--------------------------------------------------------------------------

    int nfine_hash = 0 ;
    int nfine_gus = 0 ;
    int ncoarse_hash = 0 ;
    int ncoarse_1hash = 0 ;
    int ncoarse_gus = 0 ;

    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {

        // get the task type and its hash size
        int64_t hash_size = SaxpyTasks [taskid].hsize ;
        int64_t k = SaxpyTasks [taskid].vector ;
        bool is_fine = (k >= 0) ;
        bool use_Gustavson = (hash_size == cvlen) ;

        if (is_fine)
        {
            // fine task
            if (use_Gustavson)
            { 
                // fine Gustavson task
                nfine_gus++ ;
            }
            else
            { 
                // fine hash task
                nfine_hash++ ;
            }
        }
        else
        {
            // coarse task
            if (use_Gustavson)
            { 
                // coarse Gustavson task
                ncoarse_gus++ ;
            }
            else
            { 
                // coarse hash task
                ncoarse_hash++ ;
            }
        }

        if (taskid != SaxpyTasks [taskid].leader)
        { 
            // allocate a single shared hash table for all fine
            // tasks that compute a single C(:,j)
            continue ;
        }

        int64_t hi_size = GB_IMAX (hash_size, 8) ;
        int64_t hx_size = hi_size ;
        if (!GB_IS_POWER_OF_TWO (hi_size))
        { 
            hi_size += hi_pad ;
            hx_size += hx_pad ;
        }
        if (is_fine && use_Gustavson)
        { 
            // Hf is int8_t for the fine Gustavson tasks, but round up
            // to the nearest number of int64_t values.
            int64_t hi_size2 = GB_IMAX (hi_size, 64) ;
            Hf_size_total += GB_ICEIL (hi_size2, sizeof (int64_t)) ;
        }
        else
        { 
            // Hf is int64_t for all other methods
            Hf_size_total += hi_size ;
        }
        if (!is_fine && !use_Gustavson)
        { 
            // only coarse hash tasks need Hi
            Hi_size_total += hi_size ;
        }
        // all tasks use an Hx array of size hash_size
        if (!C_iso)
        { 
            // except that the ANY_PAIR semiring does not use Hx
            Hx_size_total += hx_size ;
        }
    }

    GBURBLE ("(nthreads %d", nthreads) ;
    if (ncoarse_gus  > 0) GBURBLE (" coarse: %d",      ncoarse_gus) ;
    if (ncoarse_hash > 0) GBURBLE (" coarse hash: %d", ncoarse_hash) ;
    if (nfine_gus    > 0) GBURBLE (" fine: %d",        nfine_gus) ;
    if (nfine_hash   > 0) GBURBLE (" fine hash: %d",   nfine_hash) ;
    GBURBLE (") ") ;

    //--------------------------------------------------------------------------
    // allocate space for all hash tables
    //--------------------------------------------------------------------------

    if (Hi_size_total > 0)
    { 
        Hi_all = GB_MALLOC_WORK (Hi_size_total, int64_t, &Hi_all_size) ;
    }
    if (Hf_size_total > 0)
    { 
        // Hf must be calloc'd to initialize all entries as empty 
        Hf_all = GB_CALLOC_WORK (Hf_size_total, int64_t, &Hf_all_size) ;
    }
    if (Hx_size_total > 0)
    { 
        Hx_all = GB_MALLOC_WORK (Hx_size_total * csize, GB_void, &Hx_all_size) ;
    }

    if ((Hi_size_total > 0 && Hi_all == NULL) ||
        (Hf_size_total > 0 && Hf_all == NULL) || 
        (Hx_size_total > 0 && Hx_all == NULL))
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // split the space into separate hash tables
    //--------------------------------------------------------------------------

    int64_t *restrict Hi_part = Hi_all ;
    int64_t *restrict Hf_part = Hf_all ;
    GB_void *restrict Hx_part = Hx_all ;

    for (int taskid = 0 ; taskid < ntasks ; taskid++)
    {

        if (taskid != SaxpyTasks [taskid].leader)
        { 
            // allocate a single hash table for all fine
            // tasks that compute a single C(:,j)
            continue ;
        }

        int64_t hash_size = SaxpyTasks [taskid].hsize ;
        int64_t k = SaxpyTasks [taskid].vector ;
        bool is_fine = (k >= 0) ;
        bool use_Gustavson = (hash_size == cvlen) ;

        SaxpyTasks [taskid].Hi = Hi_part ;
        SaxpyTasks [taskid].Hf = (GB_void *) Hf_part ;
        SaxpyTasks [taskid].Hx = Hx_part ;

        int64_t hi_size = GB_IMAX (hash_size, 8) ;
        int64_t hx_size = hi_size ;
        if (!GB_IS_POWER_OF_TWO (hi_size))
        { 
            hi_size += hi_pad ;
            hx_size += hx_pad ;
        }
        if (is_fine && use_Gustavson)
        { 
            // Hf is int8_t for the fine Gustavson tasks, but round up
            // to the nearest number of int64_t values.
            int64_t hi_size2 = GB_IMAX (hi_size, 64) ;
            Hf_part += GB_ICEIL (hi_size2, sizeof (int64_t)) ;
        }
        else
        { 
            // Hf is int64_t for all other methods
            Hf_part += hi_size ;
        }
        if (!is_fine && !use_Gustavson)
        { 
            // only coarse hash tasks need Hi
            Hi_part += hi_size ;
        }
        // all tasks use an Hx array of size hash_size
        if (!C_iso)
        { 
            // except that the ANY_PAIR iso semiring does not use Hx
            Hx_part += hx_size * csize ;
        }
    }

    // assign shared hash tables to fine task teams
    for (int taskid = 0 ; taskid < nfine ; taskid++)
    {
        int leader = SaxpyTasks [taskid].leader ;
        ASSERT (SaxpyTasks [leader].vector >= 0) ;
        if (taskid != leader)
        { 
            // this fine task (Gustavson or hash) shares its hash table
            // with all other tasks in its team, for a single vector C(:,j).
            ASSERT (SaxpyTasks [taskid].vector == SaxpyTasks [leader].vector) ;
            SaxpyTasks [taskid].Hf = SaxpyTasks [leader].Hf ;
            SaxpyTasks [taskid].Hx = SaxpyTasks [leader].Hx ;
        }
    }

    //==========================================================================
    // phase1: symbolic analysis
    //==========================================================================

    // TODO constructing the tasks (the work above) can take a lot of time.
    // See the web graph, where it takes a total of 3.03 sec for 64 trials, vs
    // a total of 5.9 second for phase 7 (the numerical work below).
    // Figure out a faster method.

    GB_AxB_saxpy3_symbolic (C, M, Mask_comp, Mask_struct, M_in_place,
        A, B, SaxpyTasks, ntasks, nfine, nthreads) ;

    //==========================================================================
    // C = A*B, via saxpy3 method, phases 2 to 5
    //==========================================================================

    if (C_iso)
    {

        //----------------------------------------------------------------------
        // via the iso kernel
        //----------------------------------------------------------------------

        GBURBLE ("(iso sparse saxpy) ") ;
        info = GB (_Asaxpy3B__any_pair_iso) (C, M, Mask_comp, Mask_struct,
            M_in_place, A, B, SaxpyTasks, ntasks, nfine,
            nthreads, do_sort, Werk) ;
        if (info == GrB_SUCCESS)
        { 
            memcpy (C->x, cscalar, csize) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // via the factory kernel
        //----------------------------------------------------------------------

        info = GrB_NO_VALUE ;
        GBURBLE ("(sparse saxpy) ") ;

        #ifndef GBCOMPACT
        GB_IF_FACTORY_KERNELS_ENABLED
        { 

            //------------------------------------------------------------------
            // define the worker for the switch factory
            //------------------------------------------------------------------

            #define GB_Asaxpy3B(add,mult,xname) \
                GB (_Asaxpy3B_ ## add ## mult ## xname)

            #define GB_AxB_WORKER(add,mult,xname)                           \
            {                                                               \
                info = GB_Asaxpy3B (add,mult,xname) (C, M, Mask_comp,       \
                    Mask_struct, M_in_place, A, B,                          \
                    SaxpyTasks, ntasks, nfine, nthreads, do_sort, Werk) ;   \
            }                                                               \
            break ;

            //------------------------------------------------------------------
            // launch the switch factory
            //------------------------------------------------------------------

            if (builtin_semiring)
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
            info = GB_AxB_saxpy3_jit (C, M, Mask_comp,
                Mask_struct, M_in_place, A, B, semiring, flipxy,
                SaxpyTasks, ntasks, nfine, nthreads, do_sort, Werk) ;
        }

        //----------------------------------------------------------------------
        // via the generic kernel
        //----------------------------------------------------------------------

        if (info == GrB_NO_VALUE)
        { 
            info = GB_AxB_saxpy_generic (C, M, Mask_comp, Mask_struct,
                M_in_place, A, A_is_pattern, B, B_is_pattern, semiring,
                flipxy, GB_SAXPY_METHOD_3, ntasks, nthreads,
                SaxpyTasks, nfine, do_sort, Werk,
                // unused:
                0, 0, 0, NULL, 0, 0, NULL, NULL, NULL, NULL) ;
        }
    }

    if (info != GrB_SUCCESS)
    { 
        // out of memory, or other error
        GB_FREE_ALL ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // prune empty vectors, free workspace, and return result
    //--------------------------------------------------------------------------

    C->magic = GB_MAGIC ;
    GB_FREE_WORKSPACE ;
    GB_OK (GB_hypermatrix_prune (C, Werk)) ;
    ASSERT_MATRIX_OK (C, "saxpy3: output", GB0) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    (*mask_applied) = apply_mask ;
    return (info) ;
}

