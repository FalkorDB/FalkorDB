//------------------------------------------------------------------------------
// GB_extract_vector_list: extract vector indices for all entries in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: not needed; factory possible w/ 3 variants (sparse/hyper/full)

// Constructs a list of vector indices for each entry in a matrix.  Creates
// the output J for GB_extractTuples, and I for GB_transpose when the qsort
// method is used.  The integers of J do not have to match the integers of
// A->h, but they must be at least as large.

// FUTURE: pass in an offset to add to J

#include "extractTuples/GB_extractTuples.h"

#define GB_FREE_ALL                         \
{                                           \
    GB_WERK_POP (A_ek_slicing, int64_t) ;   \
}

GrB_Info GB_extract_vector_list // extract vector list from a matrix
(
    // output:
    void *J,                    // size nnz(A) or more
    // input:
    bool is_32,                 // if true, J is 32-bit; else 64-bit
    const GrB_Matrix A,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    ASSERT (J != NULL) ;
    ASSERT (A != NULL) ;
    ASSERT (GB_JUMBLED_OK (A)) ;        // pattern not accessed
    ASSERT (GB_ZOMBIES_OK (A)) ;        // pattern not accessed
    ASSERT (!GB_IS_BITMAP (A)) ;

    //--------------------------------------------------------------------------
    // get A and J
    //--------------------------------------------------------------------------

    GB_Ap_DECLARE (Ap, const) ; GB_Ap_PTR (Ap, A) ;
    GB_Ah_DECLARE (Ah, const) ; GB_Ah_PTR (Ah, A) ;
    const int64_t avlen = A->vlen ;

    GB_IDECL (J, , u) ; GB_IPTR (J, is_32) ;

    //--------------------------------------------------------------------------
    // determine the max number of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    //--------------------------------------------------------------------------
    // slice the entries for each task
    //--------------------------------------------------------------------------

    GB_WERK_DECLARE (A_ek_slicing, int64_t) ;
    int A_ntasks, A_nthreads ;
    GB_SLICE_MATRIX (A, 2) ;

    //--------------------------------------------------------------------------
    // extract the vector index for each entry
    //--------------------------------------------------------------------------

    int tid ;
    #pragma omp parallel for num_threads(A_nthreads) schedule(dynamic,1)
    for (tid = 0 ; tid < A_ntasks ; tid++)
    {

        // if kfirst > klast then task tid does no work at all
        int64_t kfirst = kfirst_Aslice [tid] ;
        int64_t klast  = klast_Aslice  [tid] ;

        for (int64_t k = kfirst ; k <= klast ; k++)
        {

            //------------------------------------------------------------------
            // find the part of A(:,k) to be operated on by this task
            //------------------------------------------------------------------

            int64_t j = GBh_A (Ah, k) ;
            GB_GET_PA (pA_start, pA_end, tid, k, kfirst, klast, pstart_Aslice,
                GBp_A (Ap, k, avlen), GBp_A (Ap, k+1, avlen)) ;

            //------------------------------------------------------------------
            // extract vector indices of A(:,j)
            //------------------------------------------------------------------

            for (int64_t p = pA_start ; p < pA_end ; p++)
            { 
                // J [p] = j ;
                GB_ISET (J, p, j) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

