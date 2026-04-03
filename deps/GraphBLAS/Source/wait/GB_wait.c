//------------------------------------------------------------------------------
// GB_wait:  finish all pending computations on a single matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// CALLS:     GB_builder

// The matrix A has zombies and/or pending tuples placed there by
// GrB_setElement, GrB_*assign, GB_mxm, or any other GraphBLAS method with an
// accum operator and a small update.  Zombies must now be deleted, and pending
// tuples must now be assembled together and added into the matrix.  The
// indices in A might also be jumbled; if so, they are sorted now.

// When the function returns, and all pending tuples and zombies have been
// deleted.  This is true even the function fails due to lack of memory (in
// that case, the matrix is cleared as well).

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
    GB_Matrix_free (&W) ;               \
}

#define GB_FREE_ALL                     \
{                                       \
    GB_FREE_WORKSPACE ;                 \
    GB_phybix_free (A) ;                \
}

#include "select/GB_select.h"
#include "add/GB_add.h"
#include "binaryop/GB_binop.h"
#include "pending/GB_Pending.h"
#include "builder/GB_build.h"
#include "scalar/GB_Scalar_wrap.h"

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
    struct GB_Matrix_opaque T_header, W_header, S_header ;
    GrB_Matrix T = NULL, W = NULL, S = NULL, Y = NULL ;

    ASSERT_MATRIX_OK (A, "A to wait", GB0_Z) ;

    int64_t nvec_nonempty = GB_nvec_nonempty_get (A) ;

    if (GB_IS_FULL (A) || GB_IS_BITMAP (A))
    { 
        // full and bitmap matrices never have any pending work
        ASSERT (!GB_ZOMBIES (A)) ;
        ASSERT (!GB_JUMBLED (A)) ;
        ASSERT (!GB_PENDING (A)) ;
        ASSERT (nvec_nonempty >= 0) ;
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
    if (nzombies > 0 || npending > 0 || A->jumbled || nvec_nonempty < 0)
    { 
        GB_BURBLE_MATRIX (A, "(%swait:%s " GBd " %s, " GBd " pending%s%s) ",
            A_iso ? "iso " : "", name, nzombies,
            (nzombies == 1) ? "zombie" : "zombies", npending,
            A->jumbled ? ", jumbled" : "",
            nvec_nonempty < 0 ? ", nvec" : "") ;
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
        GB_nvec_nonempty_update (A) ;
        #pragma omp flush
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
        GB_BURBLE_MATRIX (A, "%s", "(wait: unjumble only) ") ;
        GB_RETURN_IF_OUTPUT_IS_READONLY (A) ;
        GB_OK (GB_unjumble (A, Werk)) ;
        ASSERT (GB_nvec_nonempty_get (A) >= 0) ;
        #pragma omp flush
        return (GrB_SUCCESS) ;
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

        GB_BURBLE_MATRIX (A, "%s", "(wait: build pending tuples) ") ;
        GB_RETURN_IF_OUTPUT_IS_READONLY (A) ;
        GB_void *S_input = (A_iso) ? ((GB_void *) A->x) : NULL ;
        GrB_Type stype = (A_iso) ? A->type : A->Pending->type ;

        GB_CLEAR_MATRIX_HEADER (T, &T_header) ;
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
            Werk,
            A->i_is_32, A->j_is_32, // true if Pending->[ij] are 32-bit,
                                    // false if 64-bit
            true, true, true        // create T with 32/64 bits
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
        GB_OK (info) ;

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
    // deletion, but hasn't been deleted yet.  It is marked by replacing its
    // index i with GB_ZOMBIE (i).

    ASSERT_MATRIX_OK (A, "A before zombies removed", GB0) ;

    if (nzombies > 0)
    { 
        // remove all zombies from A
        GB_BURBLE_MATRIX (A, "%s", "(wait: remove zombies) ") ;
        GB_RETURN_IF_OUTPUT_IS_READONLY (A) ;
        struct GB_Scalar_opaque scalar_header ;
        int64_t k = 0 ;
        GrB_Scalar scalar = GB_Scalar_wrap (&scalar_header, GrB_INT64, &k) ;
        GB_CLEAR_MATRIX_HEADER (W, &W_header) ;
        GB_OK (GB_selector (W, GxB_NONZOMBIE, false, A, scalar, Werk)) ;
        GB_OK (GB_transplant (A, A->type, &W, Werk)) ;
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
        GB_BURBLE_MATRIX (A, "%s", "(wait: no pending; conform and finish) ") ;
        GB_OK (GB_conform (A, Werk)) ;
        ASSERT (GB_nvec_nonempty_get (A) >= 0) ;
        #pragma omp flush
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // check for quick transplant
    //--------------------------------------------------------------------------

    int64_t anz = GB_nnz (A) ;
    if (anz == 0)
    { 
        // A has no entries so just transplant T into A, then free T and
        // conform A to its desired hypersparsity.
        GB_BURBLE_MATRIX (A, "%s", "(wait: no prior entries; transplant) ") ;
        GB_OK (GB_transplant_conform (A, A->type, &T, Werk)) ;
        ASSERT (GB_nvec_nonempty_get (A) >= 0) ;
        #pragma omp flush
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // S = A+T using the SECOND_ATYPE binary operator
    //--------------------------------------------------------------------------

    // A single parallel add: S=A+T, free T, and then transplant S back into A.
    GB_BURBLE_MATRIX (A, "%s", "(wait: add pending tuples into existing A) ") ;

    // FUTURE:: if GB_add could tolerate zombies in A, then the initial
    // prune of zombies can be skipped.

    // T->Y is not present (GB_builder does not create it).  The old A->Y
    // is still valid, if present, for the matrix A prior to added the
    // pending tuples in T.  GB_add may need A->Y to compute S, but it does
    // not compute S->Y.

    struct GB_BinaryOp_opaque op_header ;
    GrB_BinaryOp op_2nd = GB_binop_second (A->type, &op_header) ;

    // If anz > 0, T is hypersparse, even if A is a GrB_Vector
    ASSERT (GB_IS_HYPERSPARSE (T)) ;
    ASSERT (tnz > 0) ;
    ASSERT (T->nvec > 0) ;
    ASSERT (A->nvec > 0) ;
    int64_t anvec = A->nvec ;
    bool ignore ;

    GB_CLEAR_MATRIX_HEADER (S, &S_header) ;
    GB_OK (GB_add (S, A->type, A->is_csc, NULL, 0, 0, &ignore, A, T,
        false, NULL, NULL, op_2nd, false, true, Werk)) ;
    GB_Matrix_free (&T) ;
    ASSERT_MATRIX_OK (S, "S after GB_wait:add", GB0) ;

    //--------------------------------------------------------------------------
    // check if the A->Y hyper_hash can be kept
    //--------------------------------------------------------------------------

    if (A->no_hyper_hash)
    { 
        // A does not want the hyper_hash, so free A->Y and S->Y if present
        GB_hyper_hash_free (A) ;
        GB_hyper_hash_free (S) ;
    }

    bool Ai_is_32 = A->i_is_32 ;
    bool Aj_is_32 = A->j_is_32 ;

    if (GB_IS_HYPERSPARSE (A) && GB_IS_HYPERSPARSE (S) && A->Y != NULL
        && S->Y == NULL && !A->Y_shallow && !GB_is_shallow (A->Y)
        && Aj_is_32 == S->j_is_32 && S->nvec == anvec)
    { 
        // A and S are both hypersparse, and the old A->Y exists and is not
        // shallow.  Check if S->h and A->h are identical.  If so, remove A->Y
        // from A and save it.  Then after the transplant of S into A, below,
        // if A is still hyperparse, transplant Y back into A->Y.

        GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
        GB_Sh_DECLARE (Sh, const) ; GB_Sh_PTR (Sh, S) ;

        bool hsame = true ;
        int nthreads = GB_nthreads (anvec, chunk, nthreads_max) ;
        if (nthreads == 1)
        { 
            // compare Ah and Sh with a single thread
            if (Aj_is_32)
            { 
                hsame = (memcmp (Ah32, Sh32, anvec * sizeof (uint32_t)) == 0) ;
            }
            else
            { 
                hsame = (memcmp (Ah64, Sh64, anvec * sizeof (uint64_t)) == 0) ;
            }
        }
        else
        { 
            // compare Ah and Sh with several threads
            int ntasks = 64 * nthreads ;
            int tid ;
            #pragma omp parallel for num_threads(nthreads) schedule(dynamic)
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
                    if (Aj_is_32)
                    { 
                        my_hsame = (memcmp (Ah32 + kstart, Sh32 + kstart,
                            (kend - kstart) * sizeof (uint32_t)) == 0) ;
                    }
                    else
                    { 
                        my_hsame = (memcmp (Ah64 + kstart, Sh64 + kstart,
                            (kend - kstart) * sizeof (uint64_t)) == 0) ;
                    }
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

    //--------------------------------------------------------------------------
    // transplant S into A, and conform it to its desired sparsity structure
    //--------------------------------------------------------------------------

    GB_OK (GB_transplant_conform (A, A->type, &S, Werk)) ;
    ASSERT (GB_nvec_nonempty_get (A) >= 0) ;

    //--------------------------------------------------------------------------
    // restore the A->Y hyper_hash, if A is still hypersparse
    //--------------------------------------------------------------------------

    if (Y != NULL && GB_IS_HYPERSPARSE (A) && A->Y == NULL &&
        Aj_is_32 == A->j_is_32)
    { 
        // The hyperlist of A has not changed.  A is still hypersparse, and has
        // no A->Y after the transplant/conform above.  The integer sizes in
        // the Y matrix still match the j integers of A, so the
        // transplant/conform did not modify them.  The original A->Y is thus
        // valid, so transplant it back into A.  If A is no longer hypersparse,
        // Y is not transplanted into A, and is freed by GB_FREE_WORKSPACE.
        A->Y = Y ;
        A->Y_shallow = false ;
        Y = NULL ;
        ASSERT (A->Y->i_is_32 == A->j_is_32) ;
        ASSERT (A->Y->j_is_32 == A->j_is_32) ;
        ASSERT (A->Y->p_is_32 == A->j_is_32) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    ASSERT_MATRIX_OK (A, "A final for GB_wait", GB0) ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

