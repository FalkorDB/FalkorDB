//------------------------------------------------------------------------------
// LAGr_SampleDegree: sample the degree median and mean
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// LAGr_SampleDegree computes estimates of the mean and median of the
// row or column degree of a graph.

#define LG_FREE_ALL LAGraph_Free ((void **) &samples, NULL) ;

#include "LG_internal.h"

int LAGr_SampleDegree
(
    // output:
    double *sample_mean,    // sampled mean degree
    double *sample_median,  // sampled median degree
    // input:
    const LAGraph_Graph G,  // graph of n nodes
    bool byout,             // if true, sample G->out_degree, else G->in_degree
    int64_t nsamples,       // number of samples
    uint64_t seed,          // random number seed
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    int64_t *samples = NULL ;
    LG_ASSERT (sample_mean != NULL, GrB_NULL_POINTER) ;
    LG_ASSERT (sample_median != NULL, GrB_NULL_POINTER) ;
    nsamples = LAGRAPH_MAX (nsamples, 1) ;
    LG_TRY (LAGraph_CheckGraph (G, msg)) ;

    GrB_Vector Degree ;

    if (G->kind == LAGraph_ADJACENCY_UNDIRECTED ||
       (G->kind == LAGraph_ADJACENCY_DIRECTED &&
        G->is_symmetric_structure == LAGraph_TRUE))
    {
        // the structure of A is known to be symmetric
        Degree = G->out_degree ;
    }
    else
    {
        // A is not known to be symmetric
        Degree = (byout) ? G->out_degree : G->in_degree ;
    }

    LG_ASSERT_MSG (Degree != NULL, LAGRAPH_NOT_CACHED, "degree unknown") ;

    //--------------------------------------------------------------------------
    // allocate workspace
    //--------------------------------------------------------------------------

    LG_TRY (LAGraph_Malloc ((void **) &samples, nsamples, sizeof (int64_t),
        msg)) ;

    //--------------------------------------------------------------------------
    // pick nsamples nodes at random and determine their degree
    //--------------------------------------------------------------------------

    // See also the hashed sampling method in LG_CC_FastSV6, which computes a
    // fast estimate of the mode of an integer vector.  This method does not
    // require a hash table.  However, the mode estimator in LG_CC_FastSV6
    // would be a good candidate to add as an LAGraph_SampleMode utility
    // function.

    GrB_Index n ;
    GRB_TRY (GrB_Vector_size (&n, Degree)) ;

    int64_t dsum = 0 ;
    for (int k = 0 ; k < nsamples ; k++)
    {
        uint64_t result = LG_Random60 (&seed) ;
        int64_t i = result % n ;
        // d = Degree (i)
        int64_t d ;
        GRB_TRY (GrB_Vector_extractElement (&d, Degree, i)) ;
        samples [k] = d ;
        dsum += d ;
    }

    // find the mean degree
    (*sample_mean) = ((double) dsum) / nsamples ;

    // find the median degree
    LG_qsort_1a (samples, nsamples) ;
    (*sample_median) = (double) samples [nsamples/2] ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    LG_FREE_ALL ;
    return (GrB_SUCCESS) ;
}
