//------------------------------------------------------------------------------
// LAGraph/src/test/test_export.c: test LG_check_export
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

#include <LAGraph_test.h>

char msg[LAGRAPH_MSG_LEN];
LAGraph_Graph G = NULL;
GrB_Type atype = NULL ;
char atype_name [LAGRAPH_MAX_NAME_LEN] ;

#define LEN 512
char filename [LEN+1] ;

typedef struct
{
    LAGraph_Kind kind ;
    const char *name ;
}
matrix_info ;

const matrix_info files [ ] =
{
    { LAGraph_ADJACENCY_UNDIRECTED, "A.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "cover.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "jagmesh7.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "ldbc-cdlp-directed-example.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "ldbc-cdlp-undirected-example.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "ldbc-directed-example.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "ldbc-undirected-example.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "ldbc-wcc-example.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "LFAT5.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "msf1.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "msf2.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "msf3.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "sample2.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "sample.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "olm1000.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "bcsstk13.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "cryg2500.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "tree-example.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "west0067.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "karate.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "matrix_bool.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "matrix_int8.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "matrix_int16.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "matrix_int32.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "matrix_int64.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "matrix_uint8.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "matrix_uint16.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "matrix_int32.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "matrix_uint64.mtx" },
    { LAGraph_ADJACENCY_DIRECTED,   "skew_fp32.mtx" },
    { LAGraph_ADJACENCY_UNDIRECTED, "pushpull.mtx" },
    { LAGRAPH_UNKNOWN, "" },
} ;

//------------------------------------------------------------------------------
// test_export
//------------------------------------------------------------------------------

void test_export (void)
{
    LAGraph_Init (msg);
    GrB_Matrix A = NULL, C = NULL ;

    for (int k = 0 ; ; k++)
    {

        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        LAGraph_Kind kind = files [k].kind ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\nMatrix: %s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;
        OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
        OK (LAGraph_TypeFromName (&atype, atype_name, msg)) ;

        // create the graph
        OK (LAGraph_New (&G, &A, kind, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        // export the graph
        GrB_Index *Ap = NULL ;
        GrB_Index *Aj = NULL ;
        void *Ax = NULL ;
        GrB_Index Ap_len, Aj_len, Ax_len, nrows, ncols ;
        size_t typesize ;
        OK (GrB_Matrix_nrows (&nrows, G->A)) ;
        OK (GrB_Matrix_ncols (&ncols, G->A)) ;

        OK (LG_check_export (G, &Ap, &Aj, &Ax, &Ap_len, &Aj_len,
            &Ax_len, &typesize, msg)) ;

        #if LAGRAPH_SUITESPARSE
        #if GxB_IMPLEMENTATION >= GxB_VERSION (6,0,2)
        printf ("reimport and check result\n") ;
        OK (GxB_Matrix_import_CSR (&C, atype, nrows, ncols, &Ap, &Aj, &Ax,
            Ap_len * sizeof (GrB_Index),
            Aj_len * sizeof (GrB_Index),
            Ax_len * typesize,
            false, true, NULL)) ;
        OK (GrB_wait (C, GrB_MATERIALIZE)) ;
        bool ok = false ;
        OK (LAGraph_Matrix_IsEqual (&ok, G->A, C, msg)) ;
        TEST_CHECK (ok) ;
        OK (GrB_free (&C)) ;
        #endif
        #endif

        LAGraph_Free ((void **) &Ap, NULL) ;
        LAGraph_Free ((void **) &Aj, NULL) ;
        LAGraph_Free ((void **) &Ax, NULL) ;
        OK (LAGraph_Delete (&G, msg)) ;
    }

    LAGraph_Finalize(msg);
}

//------------------------------------------------------------------------------
// test_export_brutal
//------------------------------------------------------------------------------

#if LAGRAPH_SUITESPARSE
void test_export_brutal (void)
{
    OK (LG_brutal_setup (msg)) ;

    GrB_Matrix A = NULL, C = NULL ;

    for (int k = 0 ; ; k++)
    {

        // load the adjacency matrix as A
        const char *aname = files [k].name ;
        LAGraph_Kind kind = files [k].kind ;
        if (strlen (aname) == 0) break;
        TEST_CASE (aname) ;
        printf ("\nMatrix: %s\n", aname) ;
        snprintf (filename, LEN, LG_DATA_DIR "%s", aname) ;
        FILE *f = fopen (filename, "r") ;
        TEST_CHECK (f != NULL) ;
        OK (LAGraph_MMRead (&A, f, msg)) ;
        OK (fclose (f)) ;
        TEST_MSG ("Loading of adjacency matrix failed") ;

        OK (LAGraph_Matrix_TypeName (atype_name, A, msg)) ;
        OK (LAGraph_TypeFromName (&atype, atype_name, msg)) ;

        // create the graph
        OK (LAGraph_New (&G, &A, kind, msg)) ;
        TEST_CHECK (A == NULL) ;    // A has been moved into G->A

        // export the graph
        GrB_Index *Ap = NULL ;
        GrB_Index *Aj = NULL ;
        void *Ax = NULL ;
        GrB_Index Ap_len, Aj_len, Ax_len, nrows, ncols ;
        size_t typesize ;
        OK (GrB_Matrix_nrows (&nrows, G->A)) ;
        OK (GrB_Matrix_ncols (&ncols, G->A)) ;

        LG_BRUTAL_BURBLE (LG_check_export (G, &Ap, &Aj, &Ax, &Ap_len, &Aj_len,
            &Ax_len, &typesize, msg)) ;

        #if GxB_IMPLEMENTATION >= GxB_VERSION (6,0,2)
        printf ("reimport and check result\n") ;
        OK (GxB_Matrix_import_CSR (&C, atype, nrows, ncols, &Ap, &Aj, &Ax,
            Ap_len * sizeof (GrB_Index),
            Aj_len * sizeof (GrB_Index),
            Ax_len * typesize,
            false, true, NULL)) ;
        OK (GrB_wait (C, GrB_MATERIALIZE)) ;
        bool ok = false ;
        OK (LAGraph_Matrix_IsEqual (&ok, G->A, C, msg)) ;
        TEST_CHECK (ok) ;
        LG_BRUTAL_BURBLE (LAGraph_Matrix_IsEqual (&ok, G->A, C, msg)) ;
        TEST_CHECK (ok) ;
        OK (GrB_free (&C)) ;
        #endif

        LAGraph_Free ((void **) &Ap, NULL) ;
        LAGraph_Free ((void **) &Aj, NULL) ;
        LAGraph_Free ((void **) &Ax, NULL) ;
        OK (LAGraph_Delete (&G, msg)) ;
    }

    OK (LG_brutal_teardown (msg)) ;
}
#endif

//****************************************************************************
//****************************************************************************
TEST_LIST = {
    {"test_export", test_export },
    #if LAGRAPH_SUITESPARSE
    {"test_export_brutal", test_export_brutal },
    #endif
    {NULL, NULL}
};
