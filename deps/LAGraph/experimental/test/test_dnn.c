//------------------------------------------------------------------------------
// LAGraph/experimental/test/test_dnn: test a small sparse deep neural network
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

#include <stdio.h>
#include <acutest.h>
#include "LAGraphX.h"
#include "LAGraph_test.h"
#include "LG_Xtest.h"
#include "LG_internal.h"

char msg [LAGRAPH_MSG_LEN] ;

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

//------------------------------------------------------------------------------
// test_dnn: test a small DNN from https://graphchallenge.mit.edu/data-sets
//------------------------------------------------------------------------------

// This test uses the smallest sparse deep neural network at
// https://graphchallenge.mit.edu/data-sets .  The original problem has 120
// layers, but the categories converge to the correct result in the first 27
// layers, so only the first 32 layers are included in this test.

// The original problem also hase 60,000 features (images) but in this
// truncated problem, only the first 2000 features are used.

void test_dnn (void)
{
    GrB_Info info ;
    setup ( ) ;

    #define NLAYERS 30
    #define NLAYERS_ORIG 120
    int nlayers = NLAYERS ;
    float bias = -0.3 ;
    int nneurons = 1024 ;
    int nfeatures = 60000 ;
    int nfeatures_subset = 1200 ;

    printf ("\nSparse deep neural network from"
        " https://graphchallenge.mit.edu/data-sets\n"
        "# neurons: %d, bias: %g\n"
        "original # of layers: %d, layers used here: %d\n"
        "original # of features: %d, features used here: %d\n",
        nneurons, bias, NLAYERS_ORIG, nlayers, nfeatures, nfeatures_subset) ;

    GrB_Matrix Y0 = NULL, Y = NULL, W [NLAYERS], Bias [NLAYERS], T = NULL ;
    GrB_Vector TrueCategories = NULL, Categories = NULL, C = NULL ;
    for (int layer = 0 ; layer < nlayers ; layer++)
    {
        W [layer] = NULL ;
        Bias [layer] = NULL ;
    }

    #define LEN 512
    char filename [LEN] ;

    //--------------------------------------------------------------------------
    // read in the problem
    //--------------------------------------------------------------------------

    snprintf (filename, LEN, LG_DATA_DIR
        "/dnn_data/sparse-images-%d_subset.mtx", nneurons) ;
    FILE *f = fopen (filename, "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&Y0, f, msg)) ;
    fclose (f) ;
    char type_name [LAGRAPH_MAX_NAME_LEN] ;
    OK (LAGraph_Matrix_TypeName (type_name, Y0, msg)) ;
    TEST_CHECK (MATCHNAME (type_name, "float")) ;
    OK (GrB_wait (Y0, GrB_MATERIALIZE)) ;

    for (int layer = 0 ; layer < nlayers ; layer++)
    {
        // read the neuron layer: W [layer]
        snprintf (filename, LEN, LG_DATA_DIR "/dnn_data/n%d-l%d.mtx",
            nneurons, layer+1) ;
        f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&(W [layer]), f, msg)) ;
        fclose (f) ;
        OK (LAGraph_Matrix_TypeName (type_name, W [layer], msg)) ;
        TEST_CHECK (MATCHNAME (type_name, "float")) ;

        // construct the bias matrix: Bias [layer].  Note that all Bias
        // matrices are the same for all layers, and all diagonal
        // entries are also the same.
        OK (GrB_Matrix_new (&(Bias [layer]), GrB_FP32, nneurons, nneurons)) ;
        for (int i = 0 ; i < nneurons ; i++)
        {
            OK (GrB_Matrix_setElement (Bias [layer], bias, i, i)) ;
        }
        OK (GrB_wait (Bias [layer], GrB_MATERIALIZE)) ;
    }

    // read T as a boolean nfeatures_subset-by-1 matrix
    snprintf (filename, LEN, LG_DATA_DIR
        "/dnn_data/neuron%d-l%d-categories_subset.mtx",
        nneurons, NLAYERS_ORIG) ;
    f = fopen (filename, "r") ;
    TEST_CHECK (f != NULL) ;
    OK (LAGraph_MMRead (&T, f, msg)) ;
    OK (LAGraph_Matrix_TypeName (type_name, T, msg)) ;
    TEST_CHECK (MATCHNAME (type_name, "bool")) ;
    // TrueCategories = T, as a boolean nfeatures-by-1 vector
    printf ("\nTrue categories:\n") ;
    OK (GrB_Vector_new (&TrueCategories, GrB_BOOL, nfeatures_subset)) ;
    OK (GrB_Col_extract (TrueCategories, NULL, NULL, T, GrB_ALL,
        nfeatures_subset, 0, NULL)) ;
    OK (LAGraph_Vector_Print (TrueCategories, LAGraph_COMPLETE, stdout, msg)) ;
    GrB_free (&T) ;

    //--------------------------------------------------------------------------
    // solve the problem
    //--------------------------------------------------------------------------

    OK (LAGraph_dnn (&Y, W, Bias, nlayers, Y0)) ;

    //--------------------------------------------------------------------------
    // check the result
    //--------------------------------------------------------------------------

    // C = sum (Y)
    OK (GrB_Vector_new (&C, GrB_FP32, nfeatures_subset)) ;
    OK (GrB_reduce (C, NULL, NULL, GrB_PLUS_FP32, Y, NULL));
    // Categories = pattern of C
    OK (GrB_Vector_new (&Categories, GrB_BOOL, nfeatures_subset)) ;
    OK (GrB_apply (Categories, NULL, NULL, GrB_ONEB_BOOL, C, (bool) true,
        NULL)) ;

    // check if Categories and TrueCategories are the same
    bool isequal ;
    printf ("\nComputed categories:\n") ;
    OK (LAGraph_Vector_Print (Categories, LAGraph_COMPLETE, stdout, msg)) ;
    OK (LAGraph_Vector_IsEqual (&isequal, TrueCategories, Categories, NULL)) ;
    TEST_CHECK (isequal) ;

    //--------------------------------------------------------------------------
    // free everything and finish the test
    //--------------------------------------------------------------------------

    GrB_free (&TrueCategories) ;
    GrB_free (&Categories) ;
    GrB_free (&C) ;
    GrB_free (&Y) ;
    GrB_free (&Y0) ;
    for (int layer = 0 ; layer < nlayers ; layer++)
    {
        GrB_free (& (W [layer])) ;
        GrB_free (& (Bias [layer])) ;
    }

    //--------------------------------------------------------------------------
    // error tests
    //--------------------------------------------------------------------------

    int result = LAGraph_dnn (NULL, NULL, NULL, nlayers, NULL) ;
    TEST_CHECK (result == GrB_NULL_POINTER) ;

    teardown ( ) ;
}

//------------------------------------------------------------------------------
// TEST_LIST: all tests to run
//------------------------------------------------------------------------------

TEST_LIST = {
    {"DNN", test_dnn},
    {NULL, NULL}
} ;
