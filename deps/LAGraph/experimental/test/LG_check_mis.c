//------------------------------------------------------------------------------
// LG_check_mis: test if iset is a maximal independent set
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

#define LG_FREE_WORK                    \
{                                       \
    GrB_free (&C) ;                     \
    LAGraph_Free ((void **) &I, NULL) ; \
    LAGraph_Free ((void **) &X, NULL) ; \
}

#define LG_FREE_ALL                     \
{                                       \
    LG_FREE_WORK ;                      \
}

#include "LG_internal.h"
#include "LG_test.h"

int LG_check_mis        // check if iset is a valid MIS of A
(
    GrB_Matrix A,
    GrB_Vector iset,
    GrB_Vector ignore_node,     // if NULL, no nodes are ignored.  otherwise,
                        // ignore_node(i)=true if node i is to be ignored, and
                        // not added to the independent set.
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check and report the results
    //--------------------------------------------------------------------------

    LG_CLEAR_MSG ;
    GrB_Matrix C = NULL ;
    GrB_Index *I = NULL ;
    bool *X = NULL ;

    GrB_Index n ;
    GRB_TRY (GrB_Matrix_nrows (&n, A)) ;

    int64_t isize ;
    GRB_TRY (GrB_Vector_reduce_INT64 (&isize, NULL, GrB_PLUS_MONOID_INT64,
        iset, NULL)) ;

    GrB_Index nvals ;
    GRB_TRY (GrB_Vector_nvals (&nvals, iset)) ;
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &I, nvals, sizeof (GrB_Index), msg));
    LAGRAPH_TRY (LAGraph_Malloc ((void **) &X, nvals, sizeof (bool), msg)) ;

    GRB_TRY (GrB_Vector_extractTuples_BOOL (I, X, &nvals, iset)) ;

    // I [0..isize-1] is the independent set
    isize = 0 ;
    for (int64_t k = 0 ; k < nvals ; k++)
    {
        if (X [k])
        {
            I [isize++] = I [k] ;
        }
    }

    LAGraph_Free ((void **) &X, NULL) ;

    // printf ("independent set found: %.16g of %.16g nodes\n",
    // (double) isize, (double) n) ;

    //--------------------------------------------------------------------------
    // verify the result
    //--------------------------------------------------------------------------

    // C = A(I,I) must be empty, except for diagonal entries
    GRB_TRY (GrB_Matrix_new (&C, GrB_BOOL, isize, isize)) ;
    GRB_TRY (GrB_Matrix_extract (C, NULL, NULL, A, I, isize, I, isize, NULL)) ;
    GRB_TRY (GrB_select (C, NULL, NULL, GrB_OFFDIAG, C, 0, NULL)) ;
    GRB_TRY (GrB_Matrix_nvals (&nvals, C)) ;
    LG_ASSERT_MSG (nvals == 0, -1, "error!  A(I,I) has an edge!\n") ;
    GrB_Matrix_free (&C) ;

    // now check if all other nodes are adjacent to the iset

    // e = iset
    GrB_Vector e = NULL ;
    GRB_TRY (GrB_Vector_dup (&e, iset)) ;

    // e = e || ignore_node
    int64_t ignored = 0 ;
    if (ignore_node != NULL)
    {
        GRB_TRY (GrB_eWiseAdd (e, NULL, NULL, GrB_LOR, e, ignore_node, NULL)) ;
        GRB_TRY (GrB_reduce (&ignored, NULL, GrB_PLUS_MONOID_INT64,
            ignore_node, NULL)) ;
    }

    // e = (e || A*iset), using the structural semiring
    GRB_TRY (GrB_vxm (e, NULL, GrB_LOR, LAGraph_any_one_bool, iset, A, NULL)) ;

    // drop explicit zeros from e
    // e<e.replace> = e
    GRB_TRY (GrB_assign (e, e, NULL, e, GrB_ALL, n, GrB_DESC_R)) ;

    GRB_TRY (GrB_Vector_nvals (&nvals, e)) ;
    GrB_Vector_free (&e) ;
    LG_ASSERT_MSG (nvals == n, -1, "error! A (I,I is not maximal!\n") ;

    LAGraph_Free ((void **) &I, NULL) ;

    printf ("maximal independent set OK %.16g of %.16g nodes",
        (double) isize, (double) n) ;
    if (ignored > 0) printf (" (%g nodes ignored)\n", (double) ignored) ;
    printf ("\n") ;
    return (GrB_SUCCESS) ;
}
