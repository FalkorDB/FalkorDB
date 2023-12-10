//------------------------------------------------------------------------------
// GB_wait:  finish all pending computations on a single matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed.  Only one variant possible.

// CALLS:     GB_builder

// The matrix A has zombies and/or pending tuples placed there by
// GrB_setElement, GrB_*assign, or GB_mxm.  Zombies must now be deleted, and
// pending tuples must now be assembled together and added into the matrix.
// The indices in A might also be jumbled; if so, they are sorted now.

// When the function returns, and all pending tuples and zombies have been
// deleted.  This is true even the function fails due to lack of memory (in
// that case, the matrix is cleared as well).

// If A is hypersparse, the time taken is at most O(nnz(A) + t log t), where t
// is the number of pending tuples in A, and nnz(A) includes both zombies and
// live entries.  There is no O(m) or O(n) time component, if A is m-by-n.
// If the number of non-empty vectors of A grows too large, then A can be
// converted to non-hypersparse.

// If A is non-hypersparse, then O(n) is added in the worst case, to prune
// zombies and to update the vector pointers for A.

// If A->nvec_nonempty is unknown (-1) it is computed.

// The A->Y hyper_hash is freed if the A->h hyperlist has to be constructed.
// Instead, it is not computed and left pending (as NULL).  It is not modified
// if A->h doesn't change.

// If the method is successful, it does an OpenMP flush just before returning.

#define GB_FREE_WORKSPACE               \
{                                       \
    GB_Matrix_free (&Y) ;               \
    GB_Matrix_free (&T) ;               \
    GB_Matrix_free (&S) ;               \
    GB_Matrix_free (&A1) ;              \
}

#define GB_FREE_ALL                     \
{                                       \
    GB_FREE_WORKSPACE ;                 \
    GB_phybix_free (A) ;                \
}

#include "GB_select.h"
#include "GB_add.h"
#include "GB_binop.h"
#include "GB_Pending.h"
#include "GB_build.h"
#include "GB_jappend.h"
#include "GB_scalar_wrap.h"

GrB_Info GB_wait                // finish all pending computations
(
    GrB_Matrix A,               // matrix with pending computations
    const char *name,           // name of the matrix
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info = GrB_SUCCESS ;
    struct GB_Matrix_opaque T_header, A1_header, S_header ;
    GrB_Matrix T = NULL, A1 = NULL, S = NULL, Y = NULL ;

    ASSERT_MATRIX_OK (A, "A to wait", GB_FLIP (GB0)) ;

    if (GB_IS_FULL (A) || GB_IS_BITMAP (A))
    { 
        // full and bitmap matrices never have any pending work
        ASSERT (!GB_ZOMBIES (A)) ;
        ASSERT (!GB_JUMBLED (A)) ;
        ASSERT (!GB_PENDING (A)) ;
        ASSERT (A->nvec_nonempty >= 0) ;
        // ensure the matrix is written to memory
        #pragma omp flush
        return (GrB_SUCCESS) ;
    }

    // only sparse and hypersparse matrices can have pending work
    ASSERT (GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A)) ;
    ASSERT (GB_ZOMBIES_OK (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (GB_PENDING_OK (A)) ;

    //--------------------------------------------------------------------------
    // get the zombie and pending count, and burble if work needs to be done
    //--------------------------------------------------------------------------

    int64_t nzombies = A->nzombies ;
    int64_t npending = GB_Pending_n (A) ;
    const bool A_iso = A->iso ;
    if (nzombies > 0 || npending > 0 || A->jumbled || A->nvec_nonempty < 0)
    { 
        GB_BURBLE_MATRIX (A, "(%swait:%s " GBd " %s, " GBd " pending%s%s) ",
            A_iso ? "iso " : "", name, nzombies,
            (nzombies == 1) ? "zombie" : "zombies", npending,
            A->jumbled ? ", jumbled" : "",
            A->nvec_nonempty < 0 ? ", nvec" : "") ;
    }

    //--------------------------------------------------------------------------
    // determine the max # of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // check if only A->nvec_nonempty is needed
    //--------------------------------------------------------------------------

    if (npending == 0 && nzombies == 0 && !A->jumbled)
    {
        // A->Y is not modified.  If not NULL, it remains valid
        if (A->nvec_nonempty < 0)
        {
            A->nvec_nonempty = GB_nvec_nonempty (A) ;
        }
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // check if A only needs to be unjumbled
    //--------------------------------------------------------------------------

    if (npending == 0 && nzombies == 0)
    { 
        // A is not conformed, so the sparsity structure of A is not modified.
        // That is, if A has no pending tuples and no zombies, but is just
        // jumbled, then it stays sparse or hypersparse.  A->Y is not modified
        // nor accessed, and remains NULL if it is NULL on input.  If it is
        // present, it remains valid.
        GB_OK (GB_unjumble (A, Werk)) ;
        ASSERT (GB_IMPLIES (info == GrB_SUCCESS, A->nvec_nonempty >= 0)) ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // assemble the pending tuples into T
    //--------------------------------------------------------------------------

    int64_t anz_orig = GB_nnz (A) ;
    int64_t asize = A->type->size ;

    int64_t tnz = 0 ;
    if (npending > 0)
    {

        //----------------------------------------------------------------------
        // construct a new hypersparse matrix T with just the pending tuples
        //----------------------------------------------------------------------

        // T has the same type as A->type, which can differ from the type of
        // the pending tuples, A->Pending->type.  The Pending->op can be NULL
        // (an implicit SECOND function), or it can be any accum operator.  The
        // z=accum(x,y) operator can have any types, and it does not have to be
        // associative.  T is constructed as iso if A is iso.

        GB_void *S_input = (A_iso) ? ((GB_void *) A->x) : NULL ;
        GrB_Type stype = (A_iso) ? A->type : A->Pending->type ;

        GB_CLEAR_STATIC_HEADER (T, &T_header) ;
        info = GB_builder (
            T,                      // create T using a static header
            A->type,                // T->type = A->type
            A->vlen,                // T->vlen = A->vlen
            A->vdim,                // T->vdim = A->vdim
            A->is_csc,              // T->is_csc = A->is_csc
            &(A->Pending->i),       // iwork_handle, becomes T->i on output
            &(A->Pending->i_size),
            &(A->Pending->j),       // jwork_handle, free on output
            &(A->Pending->j_size),
            &(A->Pending->x),       // Swork_handle, free on output
            &(A->Pending->x_size),
            A->Pending->sorted,     // tuples may or may not be sorted
            false,                  // there might be duplicates; look for them
            A->Pending->nmax,       // size of Pending->[ijx] arrays
            true,                   // is_matrix: unused
            NULL, NULL, S_input,    // original I,J,S_input tuples
            A_iso,                  // pending tuples are iso if A is iso
            npending,               // # of tuples
            A->Pending->op,         // dup operator for assembling duplicates,
                                    // NULL if A is iso
            stype,                  // type of Pending->x
            true,                   // burble is allowed
            Werk
        ) ;

        //----------------------------------------------------------------------
        // free pending tuples
        //----------------------------------------------------------------------

        // The tuples have been converted to T, which is more compact, and
        // duplicates have been removed.  The following work needs to be done
        // even if the builder fails.

        // GB_builder frees A->Pending->j and A->Pending->x.  If successful,
        // A->Pending->i is now T->i.  Otherwise A->Pending->i is freed.  In
        // both cases, A->Pending->i is NULL.
        ASSERT (A->Pending->i == NULL) ;
        ASSERT (A->Pending->j == NULL) ;
        ASSERT (A->Pending->x == NULL) ;

        // free the list of pending tuples
        GB_Pending_free (&(A->Pending)) ;
        ASSERT (!GB_PENDING (A)) ;

        ASSERT_MATRIX_OK (A, "A after moving pending tuples to T", GB0) ;

        //----------------------------------------------------------------------
        // check the status of the builder
        //----------------------------------------------------------------------

        // Finally check the status of the builder.  The pending tuples, must
        // be freed (just above), whether or not the builder is successful.
        if (info != GrB_SUCCESS)
        { 
            // out of memory in GB_builder
            GB_FREE_ALL ;
            return (info) ;
        }

        ASSERT_MATRIX_OK (T, "T = hypersparse matrix of pending tuples", GB0) ;
        ASSERT (GB_IS_HYPERSPARSE (T)) ;
        ASSERT (!GB_ZOMBIES (T)) ;
        ASSERT (!GB_JUMBLED (T)) ;
        ASSERT (!GB_PENDING (T)) ;

        tnz = GB_nnz (T) ;
        ASSERT (tnz > 0) ;
    }

    //--------------------------------------------------------------------------
    // delete zombies
    //--------------------------------------------------------------------------

    // A zombie is an entry A(i,j) in the matrix that as been marked for
    // deletion, but hasn't been deleted yet.  It is marked by "negating"
    // replacing its index i with GB_FLIP(i).

    // TODO: pass tnz to GB_selector, to pad the reallocated A matrix
    ASSERT_MATRIX_OK (A, "A before zombies removed", GB0) ;

    if (nzombies > 0)
    { 
        // remove all zombies from A
        // GB_selector frees A->Y if it changes A->h, or leaves it
        // unmodified (and valid) otherwise.
        struct GB_Scalar_opaque Thunk_header ;
        int64_t k = 0 ;
        GrB_Scalar Thunk = GB_Scalar_wrap (&Thunk_header, GrB_INT64, &k) ;
        GB_OK (GB_selector (NULL, GxB_NONZOMBIE, false, A, Thunk, Werk)) ;
        ASSERT (A->nzombies == (anz_orig - GB_nnz (A))) ;
        A->nzombies = 0 ;
    }

    ASSERT_MATRIX_OK (A, "A after zombies removed", GB0) ;

    // all the zombies are gone, and pending tuples are now in T 
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // unjumble the matrix
    //--------------------------------------------------------------------------

    GB_OK (GB_unjumble (A, Werk)) ;

    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // check for pending tuples
    //--------------------------------------------------------------------------

    if (npending == 0)
    { 
        // conform A to its desired sparsity structure and return result
        info = GB_conform (A, Werk) ;
        ASSERT (GB_IMPLIES (info == GrB_SUCCESS, A->nvec_nonempty >= 0)) ;
        #pragma omp flush
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // check for quick transplant
    //--------------------------------------------------------------------------

    int64_t anz = GB_nnz (A) ;
    if (anz == 0)
    { 
        // A has no entries so just transplant T into A, then free T and
        // conform A to its desired hypersparsity.
        info = GB_transplant_conform (A, A->type, &T, Werk) ;
        ASSERT (GB_IMPLIES (info == GrB_SUCCESS, A->nvec_nonempty >= 0)) ;
        #pragma omp flush
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // create the SECOND_ATYPE binary operator
    //--------------------------------------------------------------------------

    struct GB_BinaryOp_opaque op_header ;
    GrB_BinaryOp op_2nd = GB_binop_second (A->type, &op_header) ;

    //--------------------------------------------------------------------------
    // determine the method for A = A+T
    //--------------------------------------------------------------------------

    // If anz > 0, T is hypersparse, even if A is a GrB_Vector
    ASSERT (GB_IS_HYPERSPARSE (T)) ;
    ASSERT (tnz > 0) ;
    ASSERT (T->nvec > 0) ;
    ASSERT (A->nvec > 0) ;

    // tjfirst = first vector in T
    int64_t tjfirst = T->h [0] ;
    int64_t anz0 = 0 ;
    int64_t kA = 0 ;
    int64_t jlast ;

    int64_t *restrict Ap = A->p ;
    int64_t *restrict Ah = A->h ;
    int64_t *restrict Ai = A->i ;
    GB_void *restrict Ax = (GB_void *) A->x ;

    int64_t anvec = A->nvec ;

    // anz0 = nnz (A0) = nnz (A (:, 0:tjfirst-1)), the region not modified by T
    if (A->h != NULL)
    { 
        // find tjfirst in A->h 
        int64_t pright = anvec - 1 ;
        bool found ;
        GB_SPLIT_BINARY_SEARCH (tjfirst, A->h, kA, pright, found) ;
        // A->h [0 ... kA-1] excludes vector tjfirst.  The list
        // A->h [kA ... anvec-1] includes tjfirst.
        ASSERT (kA >= 0 && kA <= anvec) ;
        ASSERT (GB_IMPLIES (kA > 0 && kA < anvec, A->h [kA-1] < tjfirst)) ;
        ASSERT (GB_IMPLIES (found, A->h [kA] == tjfirst)) ;
        jlast = (kA > 0) ? A->h [kA-1] : (-1) ;
    }
    else
    { 
        kA = tjfirst ;
        jlast = tjfirst - 1 ;
    }

    // anz1 = nnz (A1) = nnz (A (:, kA:end)), the region modified by T
    anz0 = A->p [kA] ;
    int64_t anz1 = anz - anz0 ;
    bool ignore ;

    // A + T will have anz_new entries
    int64_t anz_new = anz + tnz ;       // must have at least this space

    if (2 * anz1 < anz0)
    {

        //----------------------------------------------------------------------
        // append new tuples to A
        //----------------------------------------------------------------------

        // A is growing incrementally.  It splits into two parts: A = [A0 A1].
        // where A0 = A (:, 0:kA-1) and A1 = A (:, kA:end).  The
        // first part (A0 with anz0 = nnz (A0) entries) is not modified.  The
        // second part (A1, with anz1 = nnz (A1) entries) overlaps with T.
        // If anz1 is zero, or small compared to anz0, then it is faster to
        // leave A0 unmodified, and to update just A1.

        // TODO: if A also had zombies, GB_selector could pad A so that
        // GB_nnz_max (A) is equal to anz + tnz.

        // make sure A has enough space for the new tuples
        if (anz_new > GB_nnz_max (A))
        { 
            // double the size if not enough space
            GB_OK (GB_ix_realloc (A, 2 * anz_new)) ;
            Ai = A->i ;
            Ax = (GB_void *) A->x ;
        }

        //----------------------------------------------------------------------
        // T = A1 + T
        //----------------------------------------------------------------------

        if (anz1 > 0)
        {

            //------------------------------------------------------------------
            // extract A1 = A (:, kA:end) as a shallow copy
            //------------------------------------------------------------------

            // A1 = [0, A (:, kA:end)], hypersparse with same dimensions as A
            GB_CLEAR_STATIC_HEADER (A1, &A1_header) ;
            GB_OK (GB_new (&A1, // hyper, existing header
                A->type, A->vlen, A->vdim, GB_Ap_malloc, A->is_csc,
                GxB_HYPERSPARSE, GB_ALWAYS_HYPER, anvec - kA)) ;

            // the A1->i and A1->x content are shallow copies of A(:,kA:end).
            // They are not allocated pointers, but point to space inside
            // Ai and Ax.

            A1->x = (void *) (Ax + (A_iso ? 0 : (asize * anz0))) ;
            A1->x_size = (A_iso ? 1 : anz1) * asize  ;
            A1->x_shallow = true ;
            A1->i = Ai + anz0 ;
            A1->i_size = anz1 * sizeof (int64_t) ;
            A1->i_shallow = true ;
            A1->iso = A_iso ;       // OK

            // fill the column A1->h and A1->p with A->h and A->p, shifted
            int64_t *restrict A1p = A1->p ;
            int64_t *restrict A1h = A1->h ;
            int64_t a1nvec = 0 ;
            for (int64_t k = kA ; k < anvec ; k++)
            {
                // get A (:,k)
                int64_t pA_start = Ap [k] ;
                int64_t pA_end = Ap [k+1] ;
                if (pA_end > pA_start)
                { 
                    // add this column to A1 if A (:,k) is not empty
                    int64_t j = GBH (Ah, k) ;
                    A1p [a1nvec] = pA_start - anz0 ;
                    A1h [a1nvec] = j ;
                    a1nvec++ ;
                }
            }

            // finalize A1
            A1p [a1nvec] = anz1 ;
            A1->nvec = a1nvec ;
            A1->nvec_nonempty = a1nvec ;
            A1->nvals = anz1 ;
            A1->magic = GB_MAGIC ;

            ASSERT_MATRIX_OK (A1, "A1 slice for GB_wait", GB0) ;

            //------------------------------------------------------------------
            // S = A1 + T, with no operator or mask
            //------------------------------------------------------------------
    
            GB_CLEAR_STATIC_HEADER (S, &S_header) ;
            GB_OK (GB_add (S, A->type, A->is_csc, NULL, 0, 0, &ignore, A1, T,
                false, NULL, NULL, op_2nd, true, Werk)) ;

            ASSERT_MATRIX_OK (S, "S = A1+T", GB0) ;

            // free A1 and T
            GB_Matrix_free (&T) ;
            GB_Matrix_free (&A1) ;

            //------------------------------------------------------------------
            // replace T with S
            //------------------------------------------------------------------

            T = S ;
            S = NULL ;
            tnz = GB_nnz (T) ;

            //------------------------------------------------------------------
            // remove A1 from the vectors of A, if A is hypersparse
            //------------------------------------------------------------------

            if (A->h != NULL)
            { 
                A->nvec = kA ;
            }
        }

        //----------------------------------------------------------------------
        // append T to the end of A0
        //----------------------------------------------------------------------

        const int64_t *restrict Tp = T->p ;
        const int64_t *restrict Th = T->h ;
        const int64_t *restrict Ti = T->i ;
        int64_t tnvec = T->nvec ;

        anz = anz0 ;
        int64_t anz_last = anz ;
    
        int nthreads = GB_nthreads (tnz, chunk, nthreads_max) ;

        // append the indices and values of T to the end of A
        GB_memcpy (Ai + anz, Ti, tnz * sizeof (int64_t), nthreads) ;
        if (!A_iso)
        {
            const GB_void *restrict Tx = (GB_void *) T->x ;
            GB_memcpy (Ax + anz * asize, Tx, tnz * asize, nthreads) ;
        }

        // append the vectors of T to the end of A
        for (int64_t k = 0 ; k < tnvec ; k++)
        { 
            int64_t j = Th [k] ;
            ASSERT (j >= tjfirst) ;
            anz += (Tp [k+1] - Tp [k]) ;
            GB_OK (GB_jappend (A, j, &jlast, anz, &anz_last, Werk)) ;
        }

        GB_jwrapup (A, jlast, anz) ;
        ASSERT (anz == anz_new) ;

        // need to recompute the # of non-empty vectors in GB_conform
        A->nvec_nonempty = -1 ;     // recomputed just below

        // A->h has been modified so A->Y is now invalid
        GB_hyper_hash_free (A) ;

        ASSERT_MATRIX_OK (A, "A after GB_wait:append", GB0) ;

        GB_Matrix_free (&T) ;

        // conform A to its desired sparsity structure
        GB_OK (GB_conform (A, Werk)) ;
        ASSERT (A->nvec_nonempty >= 0) ;

    }
    else
    { 

        //----------------------------------------------------------------------
        // A = A+T
        //----------------------------------------------------------------------

        // The update is not incremental since most of A is changing.  Just do
        // a single parallel add: S=A+T, free T, and then transplant S back
        // into A.  The nzmax of A is tight, with no room for future
        // incremental growth.

        // FUTURE:: if GB_add could tolerate zombies in A, then the initial
        // prune of zombies can be skipped.

        // T->Y is not present (GB_builder does not create it).  The old A->Y
        // is still valid, if present, for the matrix A prior to added the
        // pending tuples in T.  GB_add may need A->Y to compute S, but it does
        // not compute S->Y.

        GB_CLEAR_STATIC_HEADER (S, &S_header) ;
        GB_OK (GB_add (S, A->type, A->is_csc, NULL, 0, 0, &ignore, A, T,
            false, NULL, NULL, op_2nd, true, Werk)) ;
        GB_Matrix_free (&T) ;
        ASSERT_MATRIX_OK (S, "S after GB_wait:add", GB0) ;

        if (GB_IS_HYPERSPARSE (A) && GB_IS_HYPERSPARSE (S) && A->Y != NULL
            && !A->Y_shallow && !GB_is_shallow (A->Y))
        {
            // A and S are both hypersparse, and the old A->Y exists and is not
            // shallow.  Check if S->h and A->h are identical.  If so, remove
            // A->Y from A and save it.  Then after the transplant of S into A,
            // below, if A is still hyperparse, transplant Y back into A->Y.
            if (S->nvec == anvec)
            {
                // A and S have the same number of vectors.  Compare Ah and Sh
                int64_t *restrict Ah = A->h ;
                int64_t *restrict Sh = S->h ;
                bool hsame = true ;
                int nthreads = GB_nthreads (anvec, chunk, nthreads_max) ;
                if (nthreads == 1)
                { 
                    // compare Ah and Sh with a single thread
                    hsame = (memcmp (Ah, Sh, anvec * sizeof (int64_t)) == 0) ;
                }
                else
                { 
                    // compare Ah and Sh with several threads
                    int ntasks = 64 * nthreads ;
                    int tid ;
                    #pragma omp parallel for num_threads(nthreads) \
                        schedule(dynamic)
                    for (tid = 0 ; tid < ntasks ; tid++)
                    {
                        int64_t kstart, kend ;
                        GB_PARTITION (kstart, kend, anvec, tid, ntasks) ;
                        bool my_hsame ;
                        GB_ATOMIC_READ
                        my_hsame = hsame ;
                        if (my_hsame)
                        {
                            // compare this task's region of Ah and Sh
                            my_hsame = (memcmp (Ah + kstart, Sh + kstart,
                                (kend - kstart) * sizeof (int64_t)) == 0) ;
                            if (!my_hsame)
                            {
                                // tell other tasks to exit early
                                GB_ATOMIC_WRITE
                                hsame = false ;
                            }
                        }
                    }
                }
                if (hsame)
                { 
                    // Ah and Sh are the same, so keep A->Y
                    Y = A->Y ;
                    A->Y = NULL ;
                    A->Y_shallow = false ;
                }
            }
        }

        // transplant S into A
        GB_OK (GB_transplant_conform (A, A->type, &S, Werk)) ;
        ASSERT (A->nvec_nonempty >= 0) ;

        if (Y != NULL && GB_IS_HYPERSPARSE (A) && A->Y == NULL)
        { 
            // The hyperlist of A has not changed.  A is still hypersparse, and
            // has no A->Y after the transplant/conform above.  The original
            // A->Y is valid, so transplant it back into A.
            A->Y = Y ;
            A->Y_shallow = false ;
            Y = NULL ;
        }

        ASSERT_MATRIX_OK (A, "A after GB_wait:add", GB0) ;
    }

    //--------------------------------------------------------------------------
    // flush the matrix and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    #pragma omp flush
    return (info) ;
}

