//------------------------------------------------------------------------------
// LG_CC_FastSV5: connected components
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Yongzhe Zhang; modified by Tim Davis, Texas A&M University

//------------------------------------------------------------------------------

// Code is based on the algorithm described in the following paper
// Zhang, Azad, Hu. FastSV: FastSV: A Distributed-Memory Connected Component
// Algorithm with Fast Convergence (SIAM PP20)

// A subsequent update to the algorithm is here (which might not be reflected
// in this code):
//
// Yongzhe Zhang, Ariful Azad, Aydin Buluc: Parallel algorithms for finding
// connected components using linear algebra. J. Parallel Distributed Comput.
// 144: 14-27 (2020).

// Modified by Tim Davis, Texas A&M University

// The input matrix A must be symmetric.  Self-edges (diagonal entries) are
// OK, and are ignored.  The values and type of A are ignored; just its
// structure is accessed.

// The matrix A must have dimension 2^32 or less.
// todo: Need a 64-bit version of this method.

// todo: this function is not thread-safe, since it exports G->A and then
// reimports it back.  G->A is unchanged when the function returns, but during
// execution G->A is invalid.

// A note about "todo" and "fixme" in this file:  these do not need to be fixed
// or changed for this method, since the revised version appears as
// src/algorithm/LG_CC_FastSV6.c.  They have thus been changed here to lower
// case.

// todo: free all workspace in LG_FREE_ALL
#define LG_FREE_ALL ;

#include "LG_internal.h"

#if LAGRAPH_SUITESPARSE

//------------------------------------------------------------------------------
// hash functions: todo describe me
//------------------------------------------------------------------------------

// hash table size must be a power of 2
#define HASH_SIZE 1024

// number of samples to insert into the hash table
// todo: this seems to be a lot of entries for a HASH_SIZE of 1024.
// There could be lots of collisions.
#define HASH_SAMPLES 864

#define HASH(x) (((x << 4) + x) & (HASH_SIZE-1))
#define NEXT(x) ((x + 23) & (HASH_SIZE-1))

//------------------------------------------------------------------------------
// ht_init: todo describe me
//------------------------------------------------------------------------------

// Clear the hash table counts (ht_val [0:HASH_SIZE-1] = 0), and set all hash
// table entries as empty (ht_key [0:HASH_SIZE-1] =-1).

// todo: the memset of ht_key is confusing

// todo: the name "ht_val" is confusing.  It is not a value, but a count of
// the number of times the value x = ht_key [h] has been inserted into the
// hth position in the hash table.  It should be renamed ht_cnt.

static inline void ht_init
(
    int32_t *ht_key,
    int32_t *ht_val
)
{
    memset (ht_key, -1, sizeof (int32_t) * HASH_SIZE) ;
    memset (ht_val,  0, sizeof (int32_t) * HASH_SIZE) ;
}

//------------------------------------------------------------------------------
// ht_sample: todo describe me
//------------------------------------------------------------------------------

//

static inline void ht_sample
(
    uint32_t *V32,      // array of size n (todo: this is a bad variable name)
    int32_t n,
    int32_t samples,    // number of samples to take from V32
    int32_t *ht_key,
    int32_t *ht_val,
    uint64_t *seed
)
{
    for (int32_t k = 0 ; k < samples ; k++)
    {
        // select an entry from V32 at random
        int32_t x = V32 [LG_Random60 (seed) % n] ;

        // find x in the hash table
        // todo: make this loop a static inline function (see also below)
        int32_t h = HASH (x) ;
        while (ht_key [h] != -1 && ht_key [h] != x)
        {
            h = NEXT (h) ;
        }

        ht_key [h] = x ;
        ht_val [h]++ ;
    }
}

//------------------------------------------------------------------------------
// ht_most_frequent: todo describe me
//------------------------------------------------------------------------------

// todo what if key is returned as -1?  Code breaks.  todo: handle this case

static inline int32_t ht_most_frequent
(
    int32_t *ht_key,
    int32_t *ht_val
)
{
    int32_t key = -1 ;
    int32_t val = 0 ;                       // max (ht_val [0:HASH_SIZE-1])
    for (int32_t h = 0 ; h < HASH_SIZE ; h++)
    {
        if (ht_val [h] > val)
        {
            key = ht_key [h] ;
            val = ht_val [h] ;
        }
    }
    return (key) ;      // return most frequent key
}

//------------------------------------------------------------------------------
// Reduce_assign32:  w (index) += s, using MIN as the "+=" accum operator
//------------------------------------------------------------------------------

// mask = NULL, accumulator = GrB_MIN_UINT32, descriptor = NULL.
// Duplicates are summed with the accumulator, which differs from how
// GrB_assign works.  GrB_assign states that the presence of duplicates results
// in undefined behavior.  GrB_assign in SuiteSparse:GraphBLAS follows the
// MATLAB rule, which discards all but the first of the duplicates.

// todo: add this to GraphBLAS as a variant of GrB_assign, either as
// GxB_assign_accum (or another name), or as a GxB_* descriptor setting.

static inline int Reduce_assign32
(
    GrB_Vector *w_handle,   // vector of size n, all entries present
    GrB_Vector *s_handle,   // vector of size n, all entries present
    uint32_t *index,        // array of size n, can have duplicates
    GrB_Index n,
    int nthreads,
    int32_t *ht_key,        // hash table
    int32_t *ht_val,        // hash table (count of # of entries)
    uint64_t *seed,         // random
    char *msg
)
{

    GrB_Type w_type, s_type ;
    GrB_Index w_n, s_n, w_nvals, s_nvals, *w_i, *s_i, w_size, s_size ;
    uint32_t *w_x, *s_x ;
    bool s_iso = false ;

    //--------------------------------------------------------------------------
    // export w and s
    //--------------------------------------------------------------------------

    // export the GrB_Vectors w and s as full arrays, to get direct access to
    // their contents.  Note that this would fail if w or s are not full, with
    // all entries present.
    GRB_TRY (GxB_Vector_export_Full (w_handle, &w_type, &w_n, (void **) &w_x,
        &w_size, NULL, NULL)) ;
    GRB_TRY (GxB_Vector_export_Full (s_handle, &s_type, &s_n, (void **) &s_x,
        &s_size, &s_iso, NULL)) ;

    #if defined ( COVERAGE )
    if (n >= 200)   // for test coverage only; do not use in production!!
    #else
    if (nthreads >= 4)
    #endif
    {

        // allocate a buf array for each thread, of size HASH_SIZE
        uint32_t *mem ;
        LAGRAPH_TRY (LAGraph_Malloc ((void **) &mem, nthreads*HASH_SIZE,
            sizeof (uint32_t), msg)) ;
        // todo: check out-of-memory condition here

        // todo why is hashing needed here?  hashing is slow for what needs
        // to be computed here.  GraphBLAS has fast MIN atomic monoids that
        // do not require hashing.
        ht_init (ht_key, ht_val) ;
        ht_sample (index, n, HASH_SAMPLES, ht_key, ht_val, seed) ;

        int tid;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            // get the thread-specific buf array of size HASH_SIZE
            // todo: buf is a bad variable name; it's not a "buffer",
            // but a local workspace to compute the local version of w_x.
            uint32_t *buf = mem + tid * HASH_SIZE ;

            // copy the values from the global hash table into buf
            for (int32_t h = 0 ; h < HASH_SIZE ; h++)
            {
                if (ht_key [h] != -1)
                {
                    buf [h] = w_x [ht_key [h]] ;
                }
            }

            // this thread works on index [kstart:kend]
            int32_t kstart = (n * tid + nthreads - 1) / nthreads ;
            int32_t kend = (n * tid + n + nthreads - 1) / nthreads ;
            for (int32_t k = kstart ; k < kend ; k++)
            {
                uint32_t i = index [k] ;

                // todo: make this loop a static inline function
                int32_t h = HASH (i) ;
                while (ht_key [h] != -1 && ht_key [h] != i)
                {
                    h = NEXT (h) ;
                }

                if (ht_key [h] == -1)
                {
                    // todo is this a race condition?
                    w_x [i] = LAGRAPH_MIN (w_x [i], s_x [s_iso?0:k]) ;
                }
                else
                {
                    buf [h] = LAGRAPH_MIN (buf [h], s_x [s_iso?0:k]) ;
                }
            }
        }

        // combine intermediate results from each thread
        for (int32_t h = 0 ; h < HASH_SIZE ; h++)
        {
            int32_t i = ht_key [h] ;
            if (i != -1)
            {
                for (tid = 0 ; tid < nthreads ; tid++)
                {
                    w_x [i] = LAGRAPH_MIN (w_x [i], mem [tid * HASH_SIZE + h]) ;
                }
            }
        }

        LAGraph_Free ((void **) &mem, NULL) ;
    }
    else
    {
        // sequential version
        for (GrB_Index k = 0 ; k < n ; k++)
        {
            uint32_t i = index [k] ;
            w_x [i] = LAGRAPH_MIN (w_x [i], s_x [s_iso?0:k]) ;
        }
    }

    //--------------------------------------------------------------------------
    // reimport w and s back into GrB_Vectors, and return result
    //--------------------------------------------------------------------------

    // s is unchanged.  It was exported only to compute w (index) += s

    GRB_TRY (GxB_Vector_import_Full (w_handle, w_type, w_n, (void **) &w_x,
        w_size, false, NULL)) ;
    GRB_TRY (GxB_Vector_import_Full (s_handle, s_type, s_n, (void **) &s_x,
        s_size, s_iso, NULL)) ;

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// LG_CC_FastSV5
//------------------------------------------------------------------------------

// The output of LG_CC_FastSV5 is a vector component, where
// component(i)=s if node i is in the connected compononent whose
// representative node is node s.  If s is a representative, then
// component(s)=s.  The number of connected components in the graph G is the
// number of representatives.

#undef  LG_FREE_ALL
#define LG_FREE_ALL                             \
{                                               \
    LAGraph_Free ((void **) &I, NULL) ;         \
    LAGraph_Free ((void **) &V32, NULL) ;       \
    LAGraph_Free ((void **) &ht_key, NULL) ;    \
    LAGraph_Free ((void **) &ht_val, NULL) ;    \
    /* todo why is T not freed?? */             \
    GrB_free (&f) ;                             \
    GrB_free (&gp) ;                            \
    GrB_free (&mngp) ;                          \
    GrB_free (&gp_new) ;                        \
    GrB_free (&mod) ;                           \
}

#endif

int LG_CC_FastSV5           // SuiteSparse:GraphBLAS method, with GxB extensions
(
    // output
    GrB_Vector *component,  // component(i)=s if node is in the component s
    // inputs
    LAGraph_Graph G,        // input graph, G->A can change
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;

#if !LAGRAPH_SUITESPARSE
    LG_ASSERT_MSG (false, GrB_NOT_IMPLEMENTED, "SuiteSparse required") ;
#else

    uint32_t *V32 = NULL ;
    int32_t *ht_key = NULL, *ht_val = NULL ;
    GrB_Index n, nnz, *I = NULL ;
    GrB_Vector f = NULL, gp_new = NULL, mngp = NULL, mod = NULL, gp = NULL ;
    GrB_Matrix T = NULL ;

    LG_TRY (LAGraph_CheckGraph (G, msg)) ;
    LG_ASSERT (component != NULL, GrB_NULL_POINTER) ;

    LG_ASSERT_MSG ((G->kind == LAGraph_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGraph_ADJACENCY_DIRECTED &&
        G->is_symmetric_structure == LAGraph_TRUE)),
        -1001, "G->A must be known to be symmetric") ;

    GrB_Matrix S = G->A ;
    GRB_TRY (GrB_Matrix_nrows (&n, S)) ;
    GRB_TRY (GrB_Matrix_nvals (&nnz, S)) ;

    LG_ASSERT_MSG (n <= UINT32_MAX, -1, "problem too large (fixme)") ;

    #define FASTSV_SAMPLES 4

    bool sampling = (n * FASTSV_SAMPLES * 2 < nnz) ;

    // random number seed
    uint64_t seed = n ;

    //--------------------------------------------------------------------------
    // initializations
    //--------------------------------------------------------------------------

    // determine # of threads to use for Reduce_assign
    int nthreads, nthreads_outer, nthreads_inner ;
    LG_TRY (LAGraph_GetNumThreads (&nthreads_outer, &nthreads_inner, msg)) ;
    nthreads = nthreads_outer * nthreads_inner ;

    nthreads = LAGRAPH_MIN (nthreads, n / 16) ;
    nthreads = LAGRAPH_MAX (nthreads, 1) ;

    // # of threads to use for typecast
    int nthreads2 = n / (64*1024) ;
    nthreads2 = LAGRAPH_MIN (nthreads2, nthreads) ;
    nthreads2 = LAGRAPH_MAX (nthreads2, 1) ;

    // vectors
    GRB_TRY (GrB_Vector_new (&f,      GrB_UINT32, n)) ;
    GRB_TRY (GrB_Vector_new (&gp_new, GrB_UINT32, n)) ;
    GRB_TRY (GrB_Vector_new (&mod,    GrB_BOOL,   n)) ;

    // temporary arrays
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &I  , n, sizeof (GrB_Index), msg)) ;
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &V32, n, sizeof (uint32_t), msg)) ;

    // prepare vectors
    int64_t i;
    #pragma omp parallel for num_threads(nthreads2) schedule(static)
    for (i = 0 ; i < n ; i++)
    {
        I [i] = i ;
        V32 [i] = (uint32_t) i ;
    }

    GRB_TRY (GrB_Vector_build (f, I, V32, n, GrB_PLUS_UINT32)) ;
    GRB_TRY (GrB_Vector_dup (&gp,   f)) ;
    GRB_TRY (GrB_Vector_dup (&mngp, f)) ;

    // allocate the hash table
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &ht_key, HASH_SIZE, sizeof (int32_t),
        msg)) ;
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &ht_val, HASH_SIZE, sizeof (int32_t),
        msg)) ;

    //--------------------------------------------------------------------------
    // sample phase
    //--------------------------------------------------------------------------

    if (sampling)
    {

        //----------------------------------------------------------------------
        // export S = G->A in CSR format
        //----------------------------------------------------------------------

        // S is not modified.  It is only exported so that its contents can be
        // read by the parallel loops below.

        GrB_Type type ;
        GrB_Index nrows, ncols, nvals ;
        size_t typesize ;
        int64_t nonempty ;
        GrB_Index *Sp, *Sj ;
        void *Sx ;
        bool S_jumbled = false ;
        GrB_Index Sp_size, Sj_size, Sx_size ;
        bool S_iso = false ;

        GRB_TRY (GrB_Matrix_nvals (&nvals, S)) ;
        GRB_TRY (GxB_Matrix_export_CSR (&S, &type, &nrows, &ncols, &Sp, &Sj,
            &Sx, &Sp_size, &Sj_size, &Sx_size,
            &S_iso, &S_jumbled, NULL)) ;
        GRB_TRY (GxB_Type_size (&typesize, type)) ;
        G->A = NULL ;

        //----------------------------------------------------------------------
        // allocate space to construct T
        //----------------------------------------------------------------------

        GrB_Index Tp_len = nrows+1, Tp_size = Tp_len*sizeof(GrB_Index);
        GrB_Index Tj_len = nvals,   Tj_size = Tj_len*sizeof(GrB_Index);
        GrB_Index Tx_len = nvals ;

        GrB_Index *Tp = NULL, *Tj = NULL ;
        GrB_Index Tx_size = typesize ;
        void *Tx = NULL ;
        int32_t *range = NULL ;
        GrB_Index *count = NULL ;

        LAGRAPH_TRY (LAGraph_Malloc ((void **) &Tp, Tp_len,
            sizeof (GrB_Index), msg)) ;
        LAGRAPH_TRY (LAGraph_Malloc ((void **) &Tj, Tj_len,
            sizeof (GrB_Index), msg)) ;
        LAGRAPH_TRY (LAGraph_Calloc (&Tx, 1, typesize, msg)) ;   // T is iso

        //----------------------------------------------------------------------
        // allocate workspace
        //----------------------------------------------------------------------

        LAGRAPH_TRY (LAGraph_Malloc ((void **) &range, nthreads + 1,
            sizeof (int32_t), msg)) ;
        LAGRAPH_TRY (LAGraph_Malloc ((void **) &count, nthreads + 1,
            sizeof (GrB_Index), msg)) ;

        memset (count, 0, sizeof (GrB_Index) * (nthreads + 1)) ;

        //----------------------------------------------------------------------
        // define parallel tasks to construct T
        //----------------------------------------------------------------------

        // thread tid works on rows range[tid]:range[tid+1]-1 of S and T
        for (int tid = 0 ; tid <= nthreads ; tid++)
        {
            range [tid] = (n * tid + nthreads - 1) / nthreads ;
        }

        //----------------------------------------------------------------------
        // determine the number entries to be constructed in T for each thread
        //----------------------------------------------------------------------

        int tid;
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            for (int32_t i = range [tid] ; i < range [tid+1] ; i++)
            {
                int32_t deg = Sp [i + 1] - Sp [i] ;
                count [tid + 1] += LAGRAPH_MIN (FASTSV_SAMPLES, deg) ;
            }
        }

        //----------------------------------------------------------------------
        // count = cumsum (count)
        //----------------------------------------------------------------------

        for (tid = 0 ; tid < nthreads ; tid++)
        {
            count [tid + 1] += count [tid] ;
        }

        //----------------------------------------------------------------------
        // construct T
        //----------------------------------------------------------------------

        // T (i,:) consists of the first FASTSV_SAMPLES of S (i,:).

        // todo: this could be done by GxB_Select, using a new operator.  Need
        // to define a set of GxB_SelectOp operators that would allow for this.

        // Note that Tx is not modified.  Only Tp and Tj are constructed.

        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            GrB_Index p = count [tid] ;
            Tp [range [tid]] = p ;
            for (int32_t i = range [tid] ; i < range [tid+1] ; i++)
            {
                // construct T (i,:) from the first entries in S (i,:)
                for (int32_t j = 0 ;
                    j < FASTSV_SAMPLES && Sp [i] + j < Sp [i + 1] ; j++)
                {
                    Tj [p++] = Sj [Sp [i] + j] ;
                }
                Tp [i + 1] = p ;
            }
        }

        //----------------------------------------------------------------------
        // import the result into the GrB_Matrix T
        //----------------------------------------------------------------------

        // Note that Tx is unmodified.

        // in SuiteSparse:GraphBLAS v5, sizes are in bytes, not entries
        GrB_Index Tp_siz = Tp_size ;
        GrB_Index Tj_siz = Tj_size ;
        GrB_Index Tx_siz = Tx_size ;

        GrB_Index t_nvals = Tp [nrows] ;
        GRB_TRY (GxB_Matrix_import_CSR (&T, type, nrows, ncols,
                &Tp, &Tj, &Tx, Tp_siz, Tj_siz, Tx_siz,
                true,   // T is iso
                S_jumbled, NULL)) ;

        //----------------------------------------------------------------------
        // find the connected components of T
        //----------------------------------------------------------------------

        // todo: this is nearly identical to the final phase below.
        // Make this a function

        bool change = true, is_first = true ;
        while (change)
        {
            // hooking & shortcutting
            GRB_TRY (GrB_mxv (mngp, NULL, GrB_MIN_UINT32,
                GrB_MIN_SECOND_SEMIRING_UINT32, T, gp, NULL)) ;
            if (!is_first)
            {
                LG_TRY (Reduce_assign32 (&f, &mngp, V32, n, nthreads,
                    ht_key, ht_val, &seed, msg)) ;
            }
            GRB_TRY (GrB_eWiseAdd (f, NULL, GrB_MIN_UINT32, GrB_MIN_UINT32,
                mngp, gp, NULL)) ;

            // calculate grandparent
            // fixme: NULL parameter is SS:GrB extension
            GRB_TRY (GrB_Vector_extractTuples (NULL, V32, &n, f)) ; // fixme
            int32_t i;
            #pragma omp parallel for num_threads(nthreads2) schedule(static)
            for (i = 0 ; i < n ; i++)
            {
                I [i] = (GrB_Index) V32 [i] ;
            }
            GRB_TRY (GrB_extract (gp_new, NULL, NULL, f, I, n, NULL)) ;

            // todo: GrB_Vector_extract should have a variant where the index
            // list is not given by an array I, but as a GrB_Vector of type
            // GrB_UINT64 (or which can be typecast to GrB_UINT64).  This is a
            // common issue that arises in other algorithms as well.
            // Likewise GrB_Matrix_extract, and all forms of GrB_assign.

            // check termination
            GRB_TRY (GrB_eWiseMult (mod, NULL, NULL, GrB_NE_UINT32, gp_new,
                gp, NULL)) ;
            GRB_TRY (GrB_reduce (&change, NULL, GrB_LOR_MONOID_BOOL, mod,
                NULL)) ;

            // swap gp and gp_new
            GrB_Vector t = gp ; gp = gp_new ; gp_new = t ;
            is_first = false ;
        }

        //----------------------------------------------------------------------
        // todo: describe me
        //----------------------------------------------------------------------

        ht_init (ht_key, ht_val) ;
        ht_sample (V32, n, HASH_SAMPLES, ht_key, ht_val, &seed) ;
        int32_t key = ht_most_frequent (ht_key, ht_val) ;
        // todo: what if key is returned as -1?  Then T below is invalid.

        int64_t t_nonempty = -1 ;
        bool T_jumbled = false, T_iso = true ;

        // export T
        GRB_TRY (GxB_Matrix_export_CSR (&T, &type, &nrows, &ncols, &Tp, &Tj,
            &Tx, &Tp_siz, &Tj_siz, &Tx_siz,
            &T_iso, &T_jumbled, NULL)) ;

        // todo what is this phase doing?  It is constructing a matrix T that
        // depends only on S, key, and V32.  T contains a subset of the entries
        // in S, except that T (i,:) is empty if

        // The prior content of T is ignored; it is exported from the earlier
        // phase, only to reuse the allocated space for T.  However, T_jumbled
        // is preserved from the prior matrix T, which doesn't make sense.

        // This parallel loop is badly load balanced.  Each thread operates on
        // the same number of rows of S, regardless of how many entries appear
        // in each set of rows.  It uses one thread per task, statically
        // scheduled.

        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            GrB_Index ptr = Sp [range [tid]] ;
            // thread tid scans S (range [tid]:range [tid+1]-1,:),
            // and constructs T(i,:) for all rows in this range.
            for (int32_t i = range [tid] ; i < range [tid+1] ; i++)
            {
                int32_t pv = V32 [i] ;  // what is pv?
                Tp [i] = ptr ;          // start the construction of T(i,:)
                // T(i,:) is empty if pv == key
                if (pv != key)
                {
                    // scan S(i,:)
                    for (GrB_Index p = Sp [i] ; p < Sp [i+1] ; p++)
                    {
                        // get S(i,j)
                        int32_t j = Sj [p] ;
                        if (V32 [j] != key)
                        {
                            // add the entry T(i,j) to T, but skip it if
                            // V32 [j] is equal to key
                            Tj [ptr++] = j ;
                        }
                    }
                    // add the entry T(i,key) if there is room for it in T(i,:)
                    if (ptr - Tp [i] < Sp [i+1] - Sp [i])
                    {
                        Tj [ptr++] = key ;
                    }
                }
            }
            // count the number of entries inserted into T by this thread?
            count [tid] = ptr - Tp [range [tid]] ;
        }

        // Compact empty space out of Tj not filled in from the above phase.
        // This is a lot of work and should be done in parallel.
        GrB_Index offset = 0 ;
        for (tid = 0 ; tid < nthreads ; tid++)
        {

//          this memcpy is not safe (src/dest can overlap)
//          memcpy (Tj + offset, Tj + Tp [range [tid]],
//              sizeof (GrB_Index) * count [tid]) ;

//          // using a for loop instead:
//          GrB_Index *Tj_dest = Tj + offset ;
//          GrB_Index *Tj_src  = Tj + Tp [range [tid]] ;
//          for (int64_t k = 0 ; k < count [tid] ; k++)
//          {
//              Tj_dest [k] = Tj_src [k] ;
//          }

//          this is safe (memmove_s not necessary):
            memmove (Tj + offset, Tj + Tp [range [tid]],
                sizeof (GrB_Index) * count [tid]) ;

            offset += count [tid] ;
            count [tid] = offset - count [tid] ;
        }

        // Compact empty space out of Tp
        #pragma omp parallel for num_threads(nthreads) schedule(static)
        for (tid = 0 ; tid < nthreads ; tid++)
        {
            GrB_Index ptr = Tp [range [tid]] ;
            for (int32_t i = range [tid] ; i < range [tid+1] ; i++)
            {
                Tp [i] -= ptr - count [tid] ;
            }
        }

        // finalize T
        Tp [n] = offset ;

        // free workspace
        LAGraph_Free ((void **) &count, NULL) ;
        LAGraph_Free ((void **) &range, NULL) ;

        // import S (unchanged since last export)
        GRB_TRY (GxB_Matrix_import_CSR (&S, type, nrows, ncols,
                &Sp, &Sj, &Sx, Sp_size, Sj_size, Sx_size,
                S_iso, S_jumbled, NULL)) ;

        // import T for the final phase
        GRB_TRY (GxB_Matrix_import_CSR (&T, type, nrows, ncols,
                &Tp, &Tj, &Tx, Tp_siz, Tj_siz, Tx_siz,
                T_iso, /* T is jumbled: */ true, NULL)) ;

        // restore G->A
        G->A = S ;

    }
    else
    {

        // no sampling; the final phase operates on the whole graph
        T = S ;

    }

    //--------------------------------------------------------------------------
    // final phase
    //--------------------------------------------------------------------------

    GRB_TRY (GrB_Matrix_nvals (&nnz, T)) ;

    bool change = true ;
    while (change && nnz > 0)
    {
        // hooking & shortcutting
        GRB_TRY (GrB_mxv (mngp, NULL, GrB_MIN_UINT32,
                          GrB_MIN_SECOND_SEMIRING_UINT32, T, gp, NULL)) ;
        GRB_TRY (Reduce_assign32 (&f, &mngp, V32, n, nthreads, ht_key,
                                  ht_val, &seed, msg)) ;
        GRB_TRY (GrB_eWiseAdd (f, NULL, GrB_MIN_UINT32, GrB_MIN_UINT32,
                               mngp, gp, NULL)) ;

        // calculate grandparent
        // fixme: NULL parameter is SS:GrB extension
        GRB_TRY (GrB_Vector_extractTuples (NULL, V32, &n, f)) ; // fixme
        int32_t k;
        #pragma omp parallel for num_threads(nthreads2) schedule(static)
        for (k = 0 ; k < n ; k++)
        {
            I [k] = (GrB_Index) V32 [k] ;
        }
        GRB_TRY (GrB_extract (gp_new, NULL, NULL, f, I, n, NULL)) ;

        // check termination
        GRB_TRY (GrB_eWiseMult (mod, NULL, NULL, GrB_NE_UINT32, gp_new, gp,
            NULL)) ;
        GRB_TRY (GrB_reduce (&change, NULL, GrB_LOR_MONOID_BOOL, mod, NULL)) ;

        // swap gp and gp_new
        GrB_Vector t = gp ; gp = gp_new ; gp_new = t ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    (*component) = f ;
    f = NULL ;
    if (sampling)
    {
        GrB_free (&T) ;
    }
    LG_FREE_ALL ;
    return (GrB_SUCCESS) ;
#endif
}
