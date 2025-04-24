//------------------------------------------------------------------------------
// dnn_demo: run all neural networks from http://graphchallenge.org
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Tim Davis, Texas A&M University.

//------------------------------------------------------------------------------

// dnn_demo: test for LAGraph_dnn.

// Usage: ./dnn_demo nproblems

// nproblems is the # of test problems to solve.  If not present, it defaults
// to 12 (run all 12 DNN's).  The problems are solved in order from small to
// big.  The Makefile just runs the first and smallest problem.

#include "LG_internal.h"
#include <LAGraph.h>
#include <LAGraphX.h>

#define LG_XSTR(x) LG_STR(x)
#define LG_STR(x) #x
#define LG_SOURCE_DIR LG_XSTR (LGDIR)

//****************************************************************************
/**
 * LAGraph_tsvread: read a matrix from a tsv file
 *
 * Each line in the file specifies a single entry: i, j, x.
 * The indices i and j are assumed to be one-based.  The dimensions of the
 * matrix must be provided by the caller.  This format is used for matrices at
 * http://graphchallenge.org.  The Matrix Market format is recommended instead;
 * it is more flexible and easier to use, since that format includes the matrix
 * type and size in the file itself.  See LAGraph_mmread and LAGraph_mmwrite.
 *
 * @param[out]  A       Matrix read from the file. It is allocated by this
 *                      method
 * @param[in]   f       A handle to an open file containing the tsv data
 * @param[in]   type    The type of the matrix to create (casting may occur?)
 * @param[in]   nrows   Number of rows to set in the matrix
 * @param[in]   ncols   Number of cols to set in the matrix
 *
 * @retval  0   If operation finishes successfully (GrB_SUCCESS)
 * @return  Various GrB error codes from different issues: null pointer, out
 *          of memory, etc.
 */
int LAGraph_tsvread
(
    GrB_Matrix *A,
    FILE       *f,
    GrB_Type    type,
    GrB_Index   nrows,
    GrB_Index   ncols,
    char *msg
) ;

//------------------------------------------------------------------------------
// LAGraph_tsvread: read a tsv file
//------------------------------------------------------------------------------

// LAGraph_tsvread: read a tsv file.  Contributed by Tim Davis, Texas A&M
// University.

// Reads a tsv file.  Each line in the file specifies a single entry: i, j, x.
// The indices i and j are assumed to be one-based.  The dimensions of the
// matrix must be provided by the caller.  This format is used for matrices at
// http://graphchallenge.org.  The Matrix Market format is recommended instead;
// it is more flexible and easier to use, since that format includes the matrix
// type and size in the file itself.  See LAGraph_mmread and LAGraph_mmwrite.

// Only needed by the dnn_demo so it is only included here.

#undef  LG_FREE_ALL
#define LG_FREE_ALL GrB_free (Chandle) ;

int LAGraph_tsvread
(
    GrB_Matrix *Chandle,        // C, created on output
    FILE *f,                    // file to read from (already open)
    GrB_Type type,              // the type of C to create
    GrB_Index nrows,            // C is nrows-by-ncols
    GrB_Index ncols,
    char *msg
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (Chandle == NULL || f == NULL)
    {
        return (GrB_NULL_POINTER) ;
    }

    //--------------------------------------------------------------------------
    // create the output matrix
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix C = NULL ;
    (*Chandle) = NULL ;
    GRB_TRY (GrB_Matrix_new (&C, type, nrows, ncols)) ;

    //--------------------------------------------------------------------------
    // read the entries
    //--------------------------------------------------------------------------

    GrB_Index i, j ;

    if (type == GrB_INT64)
    {

        //----------------------------------------------------------------------
        // read the entries as int64
        //----------------------------------------------------------------------

        int64_t x ;
        while (fscanf (f, "%"PRIu64"%"PRIu64"%"PRId64"\n", &i, &j, &x) != EOF)
        {
            GRB_TRY (GrB_Matrix_setElement (C, x, i-1, j-1)) ;
        }

    }
    else if (type == GrB_UINT64)
    {

        //----------------------------------------------------------------------
        // read the entries as uint64
        //----------------------------------------------------------------------

        uint64_t x ;
        while (fscanf (f, "%"PRIu64"%"PRIu64"%"PRIu64"\n", &i, &j, &x) != EOF)
        {
            GRB_TRY (GrB_Matrix_setElement (C, x, i-1, j-1)) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // read the entries as double, and typecast to the matrix type
        //----------------------------------------------------------------------

        double x ;
        while (fscanf (f, "%"PRIu64"%"PRIu64"%lg\n", &i, &j, &x) != EOF)
        {
            GRB_TRY (GrB_Matrix_setElement (C, x, i-1, j-1)) ;
        }
    }

    //--------------------------------------------------------------------------
    // finalize the matrix and return the result
    //--------------------------------------------------------------------------

    GrB_Index ignore ;
    GRB_TRY (GrB_Matrix_nvals (&ignore, C)) ;
    (*Chandle) = C ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// dnn_demo main program
//------------------------------------------------------------------------------

#undef  LG_FREE_ALL
#define LG_FREE_ALL ;

int main (int argc, char **argv)
{

    //--------------------------------------------------------------------------
    // start LAGraph and GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    char msg [LAGRAPH_MSG_LEN] ;
    LG_TRY (LAGraph_Init (NULL)) ;

    //--------------------------------------------------------------------------
    // problem size definitions
    //--------------------------------------------------------------------------

    // The 12 problems and their sizes are hard-coded below.

    // It would be better to define these from the input files, but the problem
    // data files are not formatted in a way that makes this easy to do.  A
    // Matrix Market file format would be better (which can specify the type
    // and size of each matrix), with the additional of a problem specification
    // file that defines each of the 12 problems to solve.

    // Each problem is defined by a set of files in the DNN_DATA directory,
    // which can be obtained from http://graphchallenge.org .  The simplest way
    // to redefine the location of the data files is to make ./dnn_data a
    // symbolic link, and leave DNN_DATA unchanged.  The .gitignore file will
    // prevent dnn_data from syncing to github, so you could also simply change
    // ./dnn_data to a true directory and place all files there.  Or, change
    // the DNN_DATA macro to point to your data files.

    #define DNN_DATA LG_SOURCE_DIR "/../dnn_data"

    // Each of the 12 problems is defined by the # of neurons at each layer, N
    // = (1024, 4096, 16384, 65536), and the # of layers, L = (120, 480, or
    // 1920).  Each problem has the same number of features (F = 60000).  The
    // input files for a given problem (N,L) are as follows:

    // Input feature vectors: an F-by-N sparse matrix
    //      ./dnn_data/MNIST/sparse-images-(N).tsv
    // Neural network layers, for i = 1 to L, each an N-by-N sparse matrix:
    //      ./dnn_data/DNN/neuron(N)/n(N)-l(i).tsv
    // True categories, a list of integers, one per line:
    //      ./dnn_data/DNN/neuron(N)-l(L)-categories.tsv

    // The Bias vectors are defined with the single scalar, neuralNetBias[ ],
    // with one scalar for each value of N.  This scalar is used to construct
    // the diagonal Bias matrices for each layer.  All the layers share the
    // same matrix, but they are treated as different matrices here.  In a more
    // general problem, the Bias matrices would differ for each layer and
    // perhaps for each neuron.  As a result, this test is not permitted to
    // exploit the fact that all neurons are biased the same way.

    // Note that for a given number of neurons, N, each of the 3 problems for
    // different layers shares the same weight matrices for the first layers.
    // That is, the first 120 layers of the (1024,480) problem are the same as
    // the 120 layers of the (1024,120) problem.  This is not exploited in
    // LAGraph_dnn, but it is exploited here, simply to reduce the time to load
    // the problems.

    #define FILENAME_LEN 1024
    char filename [FILENAME_LEN] ;

    #define NMAXLAYERS 3
    int maxLayers [NMAXLAYERS] = { 120, 480, 1920 } ;

//  #define NMAXNEURONS 1
//  int Nneurons [NMAXNEURONS] = { 65536 } ;
//  double neuralNetBias [NMAXNEURONS] = { -0.45 } ;

    #define NMAXNEURONS 4
    int Nneurons [NMAXNEURONS] = { 1024, 4096, 16384, 65536 } ;
    double neuralNetBias [NMAXNEURONS] = { -0.3, -0.35, -0.4, -0.45 } ;

    int nfeatures = 60000 ;

    GrB_Matrix Y0 = NULL, Y = NULL, W [65536], Bias [65536] ;
    GrB_Vector TrueCategories = NULL, Categories = NULL, C = NULL ;

    for (int layer = 0 ; layer < 65536 ; layer++)
    {
        W [layer] = NULL ;
        Bias [layer] = NULL ;
    }

    #undef  LG_FREE_ALL
    #define LG_FREE_ALL                                 \
    {                                                   \
        GrB_free (&TrueCategories) ;                    \
        GrB_free (&Categories) ;                        \
        GrB_free (&C) ;                                 \
        GrB_free (&Y) ;                                 \
        GrB_free (&Y0) ;                                \
        for (int layer = 0 ; layer < 65536 ; layer++)   \
        {                                               \
            GrB_free (& (W [layer])) ;                  \
            GrB_free (& (Bias [layer])) ;               \
        }                                               \
    }

    // select the type.  GrB_FP32 is faster and passes all the tests.
//  GrB_Type type = GrB_FP64 ;
    GrB_Type type = GrB_FP32 ;

    printf ("type: ") ;
    if (type == GrB_FP64) printf ("double\n") ; else printf ("float\n") ;

    // get the max # of threads that can be used
    int nthreads_max, nthreads_outer, nthreads_inner ;
    LG_TRY (LAGraph_GetNumThreads (&nthreads_outer, &nthreads_inner, msg)) ;
    nthreads_max = nthreads_outer * nthreads_inner ;
    printf ("max # of nthreads: %d\n", nthreads_max) ;

    #define NNTHREADS 12
    int nthreads_list [NNTHREADS] =
        { 1, 2, 4, 8, 16, 20, 32, 40, 64, 128, 160, 256 } ;

//  #define NNTHREADS 1
//  int nthreads_list [NNTHREADS] = { 40 } ;

    // determine the # of problems to solve
    int nproblems = NMAXNEURONS * NMAXLAYERS ;
    if (argc > 1)
    {
        sscanf (argv [1], "%d", &nproblems) ;
    }
    printf ("# of problems to solve: %d\n", nproblems) ;
    int problem = 0 ;

    //--------------------------------------------------------------------------
    // run all problems
    //--------------------------------------------------------------------------

    for (int kn = 0 ; kn < NMAXNEURONS ; kn++)
    {

        //----------------------------------------------------------------------
        // check if this problem is to be solved
        //----------------------------------------------------------------------

        if (problem > nproblems) continue ;


        //----------------------------------------------------------------------
        // get the number of nneurons and neural bias
        //----------------------------------------------------------------------

        double t = LAGraph_WallClockTime ( ) ;

        int nneurons = Nneurons [kn] ;
        double b = neuralNetBias [kn] ;
        printf ("\n# neurons: %d bias: %g\n", nneurons, b) ;

        //----------------------------------------------------------------------
        // read in the initial feature vectors
        //----------------------------------------------------------------------

        sprintf (filename, "%s/MNIST/sparse-images-%d.tsv", DNN_DATA, nneurons);
        FILE *f = fopen (filename, "r") ;
        if (!f) { printf ("cannot open %s\n", filename) ; abort ( ) ; }
        LG_TRY (LAGraph_tsvread (&Y0, f, type, nfeatures, nneurons, msg)) ;
        fclose (f) ;
        t = LAGraph_WallClockTime ( ) - t ;

        printf ("# features: %g read time: %g sec\n", (double) nfeatures, t) ;
        GrB_Index nvals ;
        GRB_TRY (GrB_Matrix_nvals (&nvals, Y0)) ;
        printf ("# entries in Y0: %g million\n", (double) nvals / 1e6) ;
        fflush (stdout) ;

        //----------------------------------------------------------------------
        // run each problem size (for all #'s of layers)
        //----------------------------------------------------------------------

        for (int kl = 0 ; kl < NMAXLAYERS ; kl++)
        {

            //------------------------------------------------------------------
            // check if this problem is to be solved
            //------------------------------------------------------------------

            problem++ ;
            if (problem > nproblems) continue ;

            //------------------------------------------------------------------
            // get the number of layers in this neural net
            //------------------------------------------------------------------

            int nlayers = maxLayers [kl] ;
            printf ("\n--------------------------------------"
                "neurons per layer: %d layers: %d\n", nneurons, nlayers) ;

            //------------------------------------------------------------------
            // read in the layers in parallel
            //------------------------------------------------------------------

            double t = LAGraph_WallClockTime ( ) ;
            int first_layer = (kl == 0) ? 0 : maxLayers [kl-1] ;
            bool ok = true ;

            // assume the I/O system can handle 2-way parallelism
            int layer;
            #pragma omp parallel for schedule(dynamic,1) reduction(&&:ok) \
                num_threads (2)
            for (layer = first_layer ; layer < nlayers ; layer++)
            {
                // read the neuron layer: W [layer]
                char my_filename [1024] ;
                sprintf (my_filename, "%s/DNN/neuron%d/n%d-l%d.tsv", DNN_DATA,
                    nneurons, nneurons, layer+1) ;
                FILE *my_file = fopen (my_filename, "r") ;

                bool my_ok = true ;
                if (!my_file)
                {
                    printf ("cannot open %s\n", my_filename) ;
                    my_ok = false ;
                    continue ;
                }

                GrB_Info my_info = LAGraph_tsvread (&(W [layer]), my_file,
                    type, nneurons, nneurons, msg) ;
                fclose (my_file) ;
                my_ok = my_ok && (my_info == GrB_SUCCESS) ;

                // construct the bias matrix: Bias [layer].  Note that all Bias
                // matrices are the same for all layers, and all diagonal
                // entries are also the same, but this test must not exploit
                // that fact.
                my_info = GrB_Matrix_new (&(Bias [layer]), type,
                    nneurons, nneurons) ;
                my_ok = my_ok && (my_info == GrB_SUCCESS) ;
                for (int i = 0 ; i < nneurons ; i++)
                {
                    my_info = GrB_Matrix_setElement (Bias [layer], b, i, i) ;
                    my_ok = my_ok && (my_info == GrB_SUCCESS) ;
                }
                GrB_Index ignore ;
                my_info = GrB_Matrix_nvals (&ignore, Bias [layer]) ;
                my_ok = my_ok && (my_info == GrB_SUCCESS) ;
                ok = ok && my_ok ;
            }

            if (!ok)
            {
                printf ("neural read failure\n") ;
                abort ( ) ;
            }

            t = LAGraph_WallClockTime ( ) - t ;
            printf ("read net time %g sec\n", t) ;

            double nedges = 0 ;
            for (layer = 0 ; layer < nlayers ; layer++)
            {
                GrB_Index nvals ;
                GRB_TRY (GrB_Matrix_nvals (&nvals, W [layer])) ;
                nedges += nvals ;
            }
            printf ("# edges in all layers: %g million\n\n",
                (double) nedges / 1e6) ;
            fflush (stdout) ;

            // read TrueCategories as a boolean nfeatures-by-1 vector
            GRB_TRY (GrB_Vector_new (&TrueCategories, GrB_BOOL,
                nfeatures)) ;
            sprintf (filename, "%s/DNN/neuron%d-l%d-categories.tsv", DNN_DATA,
                nneurons, nlayers) ;
            f = fopen (filename, "r") ;
            bool check_result = (f != NULL) ;
            if (check_result)
            {
                while (1)
                {
                    int category ;
                    if (fscanf (f, "%d\n", &category) == EOF) break ;
                    GRB_TRY (GrB_Vector_setElement (TrueCategories,
                        (bool) true, category-1)) ;
                }
                fclose (f) ;
            }
            else
            {
                printf ("cannot open %s\n", filename) ;
                abort ( ) ;
            }

            //------------------------------------------------------------------
            // solve the problem with 1, 2, 4, ..., nthreads_max threads
            //------------------------------------------------------------------

            double t1 = 0, tcheck = 0 ;
            GrB_Index final_ynvals ;

            for (int kth = 0 ; kth < NNTHREADS ; kth++)
            {

                //--------------------------------------------------------------
                // set the # of threads to use
                //--------------------------------------------------------------

                int nthreads = nthreads_list [kth] ;
                if (nthreads > nthreads_max) break ;
                LAGraph_SetNumThreads (1, nthreads, NULL) ;
                printf ("nthreads %3d: ", nthreads) ;
                fflush (stdout) ;

                //--------------------------------------------------------------
                // solve the problem
                //--------------------------------------------------------------

                double t = LAGraph_WallClockTime ( ) ;
                LG_TRY (LAGraph_dnn (&Y, W, Bias, nlayers, Y0)) ;
                t = LAGraph_WallClockTime ( ) - t ;

                printf ("soln time %12.2f sec", t) ;

                if (nthreads == 1)
                {
                    t1 = t ;
                    printf ("                 ") ;
                }
                else
                {
                    printf (" speedup %8.2f", t1/t) ;
                }

                double rate = ((double) nfeatures) * ((double) nedges) / t ;
                printf (" rate %10.4f (1e9 edges/sec) ", rate / 1e9) ;

                //--------------------------------------------------------------
                // check the result
                //--------------------------------------------------------------

                // this is so fast, it's hardly worth timing ...
                tcheck = LAGraph_WallClockTime ( ) ;
                GRB_TRY (GrB_Matrix_nvals (&final_ynvals, Y)) ;

                // C = sum (Y)
                GRB_TRY (GrB_Vector_new (&C, type, nfeatures)) ;
                GRB_TRY (GrB_reduce (C, NULL, NULL, GrB_PLUS_FP64, Y, NULL));
                // Categories = pattern of C
                GRB_TRY (GrB_Vector_new (&Categories, GrB_BOOL, nfeatures)) ;
                GRB_TRY (GrB_apply (Categories, NULL, NULL, GrB_ONEB_BOOL,
                    C, (bool) true, NULL)) ;

                // write out Categories, as a 1-based file
                /*
                sprintf (filename, "my_neuron%d-l%d-categories_threads%d.tsv",
                    nneurons, nlayers, nthreads) ;
                FILE *ff = fopen (filename, "w") ;
                for (int i = 0 ; i < nfeatures ; i++)
                {
                    bool c = false ;
                    GRB_TRY (GrB_Vector_extractElement (&c, Categories, i)) ;
                    if (c) fprintf (ff, "%d\n", i + 1) ;
                }
                fclose (ff) ;
                */

                if (check_result)
                {
                    // check if Categories and TrueCategories are the same
                    bool isequal ;
                    LG_TRY (LAGraph_Vector_IsEqual (&isequal,
                        TrueCategories, Categories, NULL)) ;
                    if (!isequal)
                    {
                        printf ("test failure!\n") ;
                    }
                }
                printf ("\n") ;

                GrB_free (&Categories) ; Categories = NULL;
                GrB_free (&C) ; C = NULL;
                GrB_free (&Y) ; Y = NULL;
                tcheck = LAGraph_WallClockTime ( ) - tcheck ;
            }

            printf ("\n# entries in final Y: %g million\n",
                (double) final_ynvals / 1e6) ;
            printf ("check time: %g sec\n", tcheck) ;
            LAGraph_SetNumThreads (nthreads_outer, nthreads_inner, NULL) ;
        }

        //----------------------------------------------------------------------
        // free the problem
        //----------------------------------------------------------------------

        LG_FREE_ALL ;
    }

    //--------------------------------------------------------------------------
    // finalize LAGraph and GraphBLAS
    //--------------------------------------------------------------------------

    LG_TRY (LAGraph_Finalize (NULL)) ;
    printf ("all tests passed\n") ;
    return (GrB_SUCCESS) ;
}
