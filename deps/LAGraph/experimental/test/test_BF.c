//------------------------------------------------------------------------------
// LAGraph/experimental/test/test_BF
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Jinhao Chen and Tim Davis, Texas A&M University

//------------------------------------------------------------------------------

#include <stdio.h>
#include <acutest.h>
#include <LAGraphX.h>
#include <LAGraph_test.h>
#include <LG_Xtest.h>

//------------------------------------------------------------------------------
// globals
//------------------------------------------------------------------------------

#define LEN 512
char filename [LEN+1] ;
char msg [LAGRAPH_MSG_LEN] ;

//------------------------------------------------------------------------------
// test cases
//------------------------------------------------------------------------------

typedef struct
{
    bool has_negative_cycle ;
    bool has_integer_weights ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    0, 1, "karate.mtx",
    1, 0, "west0067.mtx",
    1, 1, "matrix_int8.mtx",
    0, 0, ""
} ;

//------------------------------------------------------------------------------
// setup: start a test
//------------------------------------------------------------------------------

void setup (void)
{
    OK (LAGraph_Init (msg)) ;
    OK (LAGraph_Random_Init (msg)) ;
}

//------------------------------------------------------------------------------
// teardown: finalize a test
//------------------------------------------------------------------------------

void teardown (void)
{
    OK (LAGraph_Random_Finalize (msg)) ;
    OK (LAGraph_Finalize (msg)) ;
}

void test_BF (void)
{

    GrB_Info info ;
    setup ( ) ;

    for (int k = 0 ; ; k++)
    {
        GrB_Matrix A = NULL, AT = NULL, A_orig = NULL ;
        GrB_Index *I = NULL, *J = NULL ; // for col/row indices of entries in A
        double *W = NULL, *d = NULL ;
        int64_t *pi = NULL, *pi10 = NULL ;
        int32_t *W_int32 = NULL, *d10 = NULL ;
        GrB_Vector d1 = NULL, d2 = NULL, d3 = NULL, d4 = NULL, d5 = NULL,
            d5a = NULL, d6 = NULL, d7 = NULL, d8 = NULL, d9 = NULL, h1 = NULL,
            h2 = NULL, h5 = NULL, h5a = NULL, h6 = NULL, pi1 = NULL, pi2 = NULL,
            pi5 = NULL, pi5a = NULL, pi6 = NULL ;

        //----------------------------------------------------------------------
        // read in a matrix from a file
        //----------------------------------------------------------------------

        // load the matrix as A_orig
        const char *aname = files [k].name ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A_orig, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of valued matrix failed") ;
        printf ("\nMatrix: %s\n", aname) ;
        OK (LAGraph_Matrix_Print (A_orig, LAGraph_SHORT, stdout, NULL)) ;

        bool has_negative_cycle  = files [k].has_negative_cycle ;
        bool has_integer_weights = files [k].has_integer_weights ;
        int ktrials = (has_negative_cycle) ? 2 : 1 ;

        //----------------------------------------------------------------------
        // get the size of the problem
        //----------------------------------------------------------------------

        GrB_Index nvals ;
        GrB_Matrix_nvals (&nvals, A_orig) ;
        GrB_Index nrows, ncols ;
        OK (GrB_Matrix_nrows (&nrows, A_orig)) ;
        OK (GrB_Matrix_ncols (&ncols, A_orig)) ;
        GrB_Index n = nrows ;
        OK (LAGraph_Malloc ((void **) &I, nvals, sizeof (GrB_Index), msg)) ;
        OK (LAGraph_Malloc ((void **) &J, nvals, sizeof (GrB_Index), msg)) ;
        OK (LAGraph_Malloc ((void **) &W, nvals, sizeof (double), msg)) ;
        OK (LAGraph_Malloc ((void **) &W_int32, nvals, sizeof (int32_t), msg)) ;

        OK (GrB_Matrix_extractTuples_FP64 (I, J, W, &nvals, A_orig)) ;
        if (has_integer_weights)
        {
            OK (GrB_Matrix_extractTuples_INT32 (I, J, W_int32, &nvals,
                A_orig)) ;
        }

        //----------------------------------------------------------------------
        // copy the matrix and set its diagonal to 0
        //----------------------------------------------------------------------

        OK (GrB_Matrix_dup (&A, A_orig)) ;
        for (GrB_Index i = 0; i < n; i++)
        {
            OK (GrB_Matrix_setElement_FP64 (A, 0, i, i)) ;
        }

        //----------------------------------------------------------------------
        // AT = A'
        //----------------------------------------------------------------------

        double tt = LAGraph_WallClockTime ( ) ;
        OK (GrB_Matrix_free (&AT)) ;
        OK (GrB_Matrix_new (&AT, GrB_FP64, ncols, nrows)) ;
        OK (GrB_transpose (AT, NULL, NULL, A, NULL)) ;
        double transpose_time = LAGraph_WallClockTime ( ) - tt ;
        fprintf (stderr, "transpose     time: %g\n", transpose_time) ;

        //----------------------------------------------------------------------
        // get the source node
        //----------------------------------------------------------------------

        GrB_Index s = 0 ;
        fprintf (stderr, "\n==========input graph: nodes: %g edges: %g "
            "source node: %g\n", (double) n, (double) nvals, (double) s) ;

        //----------------------------------------------------------------------
        // run 1 or 2 trials (2 negative weight cycles)
        //----------------------------------------------------------------------

        for (int kk = 1 ; kk <= ktrials ; kk++)
        {
            int valid = (has_negative_cycle) ? GrB_NO_VALUE : GrB_SUCCESS ;

            //------------------------------------------------------------------
            // run LAGraph_BF_full1 before setting the diagonal to 0
            //------------------------------------------------------------------

            int ntrials = 1 ;   // increase this to 10, 100, whatever, for more
                                // accurate timing
            // start the timer
            double t5 = LAGraph_WallClockTime ( ) ;
            int result ;

            for (int trial = 0 ; trial < ntrials ; trial++)
            {
                GrB_free (&d5) ;
                GrB_free (&pi5) ;
                GrB_free (&h5) ;
                result = (LAGraph_BF_full1 (&d5, &pi5, &h5, A_orig, s)) ;
                printf ("result: %d\n", result) ;
                TEST_CHECK (result == valid) ;
            }

            // stop the timer
            t5 = LAGraph_WallClockTime ( ) - t5 ;
            t5 = t5 / ntrials;
            fprintf (stderr, "BF_full1      time: %12.6e (sec), rate:"
                " %g (1e6 edges/sec)\n", t5, 1e-6*((double) nvals) / t5) ;

            //------------------------------------------------------------------
            // run LAGraph_BF_full1a before setting the diagonal to 0
            //------------------------------------------------------------------

            // start the timer
            double t5a = LAGraph_WallClockTime ( ) ;

            for (int trial = 0 ; trial < ntrials ; trial++)
            {
                GrB_free (&d5a) ;
                GrB_free (&pi5a) ;
                GrB_free (&h5a) ;
                result = (LAGraph_BF_full1a (&d5a, &pi5a, &h5a, A_orig, s)) ;
                TEST_CHECK (result == valid) ;
            }

            // stop the timer
            t5a = LAGraph_WallClockTime ( ) - t5a ;
            t5a = t5a / ntrials;
            fprintf (stderr, "BF_full1a     time: %12.6e (sec), rate:"
                " %g (1e6 edges/sec)\n", t5a, 1e-6*((double) nvals) / t5a) ;

            //------------------------------------------------------------------
            // run LAGraph_BF_full2 before setting the diagonal to 0
            //------------------------------------------------------------------

            // start the timer
            double t6 = LAGraph_WallClockTime ( ) ;

            for (int trial = 0 ; trial < ntrials ; trial++)
            {
                GrB_free (&d6) ;
                GrB_free (&pi6) ;
                GrB_free (&h6) ;
                result = LAGraph_BF_full2 (&d6, &pi6, &h6, A_orig, s) ;
                TEST_CHECK (result == valid) ;
            }

            // stop the timer
            t6 = LAGraph_WallClockTime ( ) - t6 ;
            t6 = t6 / ntrials;
            fprintf (stderr, "BF_full2      time: %12.6e (sec), rate:"
                " %g (1e6 edges/sec)\n", t6, 1e-6*((double) nvals) / t6) ;

            //------------------------------------------------------------------
            // run the LAGraph_BF_full on node s
            //------------------------------------------------------------------

            // start the timer
            double t1 = LAGraph_WallClockTime ( ) ;

            for (int trial = 0 ; trial < ntrials ; trial++)
            {
                GrB_free (&d1) ;
                GrB_free (&pi1) ;
                GrB_free (&h1) ;
                result = LAGraph_BF_full (&d1, &pi1, &h1, A, s) ;
                printf ("result %d\n", result) ;
                TEST_CHECK (result == valid) ;
            }

            // stop the timer
            t1 = LAGraph_WallClockTime ( ) - t1 ;
            t1 = t1 / ntrials;
            fprintf (stderr, "BF_full       time: %12.6e (sec), rate:"
                " %g (1e6 edges/sec)\n", t1, 1e-6*((double) nvals) / t1) ;
            fprintf (stderr, "t(BF_full1) / t(BF_full):      %g\n", t5/t1) ;

            //------------------------------------------------------------------
            // run the BF on node s with LAGraph_BF_basic
            //------------------------------------------------------------------

            // start the timer
            double t2 = LAGraph_WallClockTime ( ) ;

            for (int trial = 0 ; trial < ntrials ; trial++)
            {
                GrB_free (&d3) ;
                result = LAGraph_BF_basic (&d3, A, s) ;
                TEST_CHECK (result == valid) ;
            }

            // stop the timer
            t2 = LAGraph_WallClockTime ( ) - t2 ;
            t2 = t2 / ntrials;
            fprintf (stderr, "BF_basic      time: %12.6e (sec), rate:"
                " %g (1e6 edges/sec)\n", t2, 1e-6*((double) nvals) / t2) ;
            fprintf (stderr, "speedup of BF_basic:       %g\n", t1/t2) ;

            //------------------------------------------------------------------
            // run the BF on node s with LAGraph_pure_c
            //------------------------------------------------------------------

            // start the timer
            double t3 = LAGraph_WallClockTime ( ) ;

            for (int trial = 0 ; trial < ntrials ; trial++)
            {
                LAGraph_Free ((void **) &d, NULL) ;
                LAGraph_Free ((void **) &pi, NULL) ;
                result = LAGraph_BF_pure_c_double (&d, &pi, s, n, nvals,
                    (const int64_t *) I, (const int64_t *) J, W) ;
                TEST_CHECK (result == valid) ;
            }

            // stop the timer
            t3 = LAGraph_WallClockTime ( ) - t3 ;
            t3 = t3 / ntrials;
            fprintf (stderr, "BF_pure_c_double  : %12.6e (sec), rate:"
                " %g (1e6 edges/sec)\n", t3, 1e-6*((double) nvals) / t3) ;
            fprintf (stderr, "speedup of BF_pure_c:      %g\n", t1/t3) ;

            if (has_integer_weights)
            {
                printf ("pure_c integer:\n") ;
                LAGraph_Free ((void **) &d10, NULL) ;
                LAGraph_Free ((void **) &pi10, NULL) ;
                result = LAGraph_BF_pure_c (&d10, &pi10, s, n, nvals,
                    (const int64_t *) I, (const int64_t *) J, W_int32) ;
                LAGraph_Free ((void **) &pi10, NULL) ;
                TEST_CHECK (result == valid) ;
            }

            //------------------------------------------------------------------
            // run the LAGraph_BF_full_mxv on node s
            //------------------------------------------------------------------

            // start the timer
            double t4 = LAGraph_WallClockTime ( ) ;

            for (int trial = 0 ; trial < ntrials ; trial++)
            {
                GrB_free (&d2) ;
                GrB_free (&pi2) ;
                GrB_free (&h2) ;
                result = LAGraph_BF_full_mxv (&d2, &pi2, &h2, AT, s) ;
                TEST_CHECK (result == valid) ;
            }

            // stop the timer
            t4 = LAGraph_WallClockTime ( ) - t4 ;
            t4 = t4 / ntrials;
            fprintf (stderr, "BF_full_mxv   time: %12.6e (sec), rate:"
                " %g (1e6 edges/sec)\n", t4, 1e-6*((double) nvals) / t4) ;
            fprintf (stderr, "speedup of BF_full_mxv:    %g\n", t1/t4) ;

            //------------------------------------------------------------------
            // run the BF on node s with LAGraph_BF_basic_mxv
            //------------------------------------------------------------------

            // start the timer
            double t7 = LAGraph_WallClockTime ( ) ;

            for (int trial = 0 ; trial < ntrials ; trial++)
            {
                GrB_free (&d4) ;
                result = LAGraph_BF_basic_mxv (&d4, AT, s) ;
                TEST_CHECK (result == valid) ;
            }

            // stop the timer
            t7 = LAGraph_WallClockTime ( ) - t7 ;
            t7 = t7 / ntrials;
            fprintf (stderr, "BF_basic_mxv  time: %12.6e (sec), rate:"
                " %g (1e6 edges/sec)\n", t7, 1e-6*((double) nvals) / t7) ;
            fprintf (stderr, "speedup of BF_basic_mxv:   %g\n", t1/t7) ;

            //------------------------------------------------------------------
            // run the BF on node s with LAGraph_BF_basic_pushpull
            //------------------------------------------------------------------

            GrB_free (&d7) ;
            result = (LAGraph_BF_basic_pushpull (&d7, A, AT, s)) ;
            TEST_CHECK (result == valid) ;

            GrB_free (&d8) ;
            result = (LAGraph_BF_basic_pushpull (&d8, NULL, AT, s)) ;
            TEST_CHECK (result == valid) ;

            GrB_free (&d9) ;
            result = (LAGraph_BF_basic_pushpull (&d9, A, NULL, s)) ;
            TEST_CHECK (result == valid) ;

            //------------------------------------------------------------------
            // check results
            //------------------------------------------------------------------

            bool isequal = false ;

            if (!has_negative_cycle)
            {
                TEST_CHECK (d != NULL && d1 != NULL) ;

                for (int64_t i = 0 ; i < n ; i++)
                {
                    double di = INFINITY ;
                    int64_t pii = 0;
                    OK (GrB_Vector_extractElement (&di, d1, i)) ;
                    TEST_CHECK (di == d[i]) ;

                    // since d5 is a dense vector filled with infinity, we have
                    // to compare it against d seperaterly
                    OK (GrB_Vector_extractElement (&di, d5, i)) ;
                    TEST_CHECK (di == d[i]) ;

                    // since d5a is a dense vector filled with infinity, we
                    // have to compare it against d seperaterly
                    OK (GrB_Vector_extractElement (&di, d5a, i)) ;
                    TEST_CHECK (di == d[i]) ;
                    /*
                    OK (GrB_Vector_extractElement (&pii, pi1, i)) ;
                    TEST_CHECK (pii == pi[i]+1) ;
                    */
                }

                if (has_integer_weights)
                {
                    // compare d and d10
                    for (int64_t i = 0 ; i < n ; i++)
                    {
                        double d10i = (double) d10 [i] ;
                        double di = (isinf (d [i])) ? INT32_MAX : d [i] ;
                        TEST_CHECK (d10i == di) ;
                    }
                }

                OK (LAGraph_Vector_IsEqual (&isequal, d1, d3, NULL)) ;
                TEST_CHECK (isequal) ;

                OK (LAGraph_Vector_IsEqual (&isequal, d1, d4, NULL)) ;
                TEST_CHECK (isequal) ;

                OK (LAGraph_Vector_IsEqual (&isequal, d1, d2, NULL)) ;
                TEST_CHECK (isequal) ;

                OK (LAGraph_Vector_IsEqual (&isequal, d1, d6, NULL)) ;
                TEST_CHECK (isequal) ;

                /*
                OK (LAGraph_Vector_IsEqual (&isequal, pi1, pi2, NULL)) ;
                TEST_CHECK (isequal) ;
                */

                OK (LAGraph_Vector_IsEqual (&isequal, d1, d7, NULL)) ;
                TEST_CHECK (isequal) ;

                OK (LAGraph_Vector_IsEqual (&isequal, d1, d8, NULL)) ;
                TEST_CHECK (isequal) ;

                OK (LAGraph_Vector_IsEqual (&isequal, d1, d9, NULL)) ;
                TEST_CHECK (isequal) ;
            }

            //------------------------------------------------------------------
            // ensure the matrix has all positive weights for next trial
            //------------------------------------------------------------------

            if (has_negative_cycle)
            {
                printf ("\n-------------------------- A = abs (A)\n") ;
                OK (GrB_apply (A,  NULL, NULL, GrB_ABS_FP64, A,  NULL)) ;
                OK (GrB_apply (AT, NULL, NULL, GrB_ABS_FP64, AT, NULL)) ;
                OK (GrB_apply (A_orig, NULL, NULL, GrB_ABS_FP64, A_orig, NULL));
                OK (GrB_Matrix_extractTuples_FP64  (I, J, W, &nvals, A_orig)) ;
                if (has_integer_weights)
                {
                    OK (GrB_Matrix_extractTuples_INT32 (I, J, W_int32, &nvals,
                        A_orig)) ;
                }
                has_negative_cycle = false ;
            }
        }

        //----------------------------------------------------------------------
        // free all workspace and finish
        //----------------------------------------------------------------------

        GrB_free (&A) ;
        GrB_free (&A_orig) ;
        GrB_free (&AT) ;
        LAGraph_Free ((void **) &I, NULL) ;
        LAGraph_Free ((void **) &J, NULL) ;
        LAGraph_Free ((void **) &W, NULL) ;
        LAGraph_Free ((void **) &W_int32, NULL) ;
        LAGraph_Free ((void **) &d, NULL) ;
        LAGraph_Free ((void **) &pi, NULL) ;
        GrB_free (&d1) ;
        GrB_free (&pi1) ;
        GrB_free (&h1) ;
        GrB_free (&d2) ;
        GrB_free (&pi2) ;
        GrB_free (&h2) ;
        GrB_free (&d3) ;
        GrB_free (&d4) ;
        GrB_free (&d5) ;
        GrB_free (&pi5) ;
        GrB_free (&h5) ;
        GrB_free (&d5a) ;
        GrB_free (&pi5a) ;
        GrB_free (&h5a) ;
        GrB_free (&d6) ;
        GrB_free (&d7) ;
        GrB_free (&d8) ;
        GrB_free (&d9) ;
        GrB_free (&pi6) ;
        GrB_free (&h6) ;
        LAGraph_Free ((void **) &d10, NULL) ;
    }

    teardown ( ) ;
}

//------------------------------------------------------------------------------
// TEST_LIST: list of tasks for this entire test
//------------------------------------------------------------------------------

TEST_LIST =
{
    { "test_BF", test_BF },
    { NULL, NULL }
} ;
