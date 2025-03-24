//------------------------------------------------------------------------------
// GB_mex_test25: more simple tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "../Source/get_set/GB_get_set.h"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    GrB_Info info ;
    bool malloc_debug = GB_mx_get_global (true) ;

    //--------------------------------------------------------------------------
    // GrB_Vector size
    //--------------------------------------------------------------------------

    GrB_Vector v ;
    uint64_t nmax = GrB_INDEX_MAX + 1 ;
    uint64_t nvals ;

    for (uint64_t n = nmax - 5 ; n <= nmax ; n++)
    {
        printf ("n %lu, nmax %lu, (n == nmax): %d\n", n, nmax, n == nmax) ;
        OK (GrB_Vector_new (&v, GrB_BOOL, n)) ;
        OK (GrB_assign (v, NULL, NULL, true, GrB_ALL, n, NULL)) ;
        OK (GxB_print (v, 1)) ;
        OK (GrB_Vector_nvals (&nvals, v)) ;
        CHECK (nvals == n) ;
        OK (GrB_Vector_free (&v)) ;
    }

    GrB_Matrix A ;
    uint64_t n = (GrB_INDEX_MAX + 1) / 2 ;
    OK (GrB_Matrix_new (&A, GrB_BOOL, n, 2)) ;
    OK (GrB_assign (A, NULL, NULL, true, GrB_ALL, n, GrB_ALL, 2, NULL)) ;
    OK (GxB_print (A, 1)) ;
    OK (GrB_Matrix_nvals (&nvals, A)) ;
    printf ("nvals (A) %lu\n", nvals) ;
    CHECK (nvals == INT64_MAX) ;
    OK (GrB_Matrix_free (&A)) ;

    //--------------------------------------------------------------------------
    // GxB_Context_error
    //--------------------------------------------------------------------------

    const char *s = NULL ;
    OK (GrB_error (&s, GxB_CONTEXT_WORLD)) ;
    CHECK (s != NULL) ;
    CHECK (strlen (s) == 0) ;
    printf ("GxB_Context_error [%s] ok\n", s) ;
    OK (GxB_Context_wait (GxB_CONTEXT_WORLD, GrB_COMPLETE)) ;

    //--------------------------------------------------------------------------
    // report object sizes
    //--------------------------------------------------------------------------

    printf ("%4d: sizeof (struct GB_Type_opaque)\n",
             (int) sizeof (struct GB_Type_opaque)) ;
    printf ("%4d: sizeof (struct GB_UnaryOp_opaque)\n",
             (int) sizeof (struct GB_UnaryOp_opaque)) ;
    printf ("%4d: sizeof (struct GB_BinaryOp_opaque)\n",
             (int) sizeof (struct GB_BinaryOp_opaque)) ;
    printf ("%4d: sizeof (struct GB_SelectOp_opaque)\n",
             (int) sizeof (struct GB_SelectOp_opaque)) ;
    printf ("%4d: sizeof (struct GB_Monoid_opaque)\n",
             (int) sizeof (struct GB_Monoid_opaque)) ;
    printf ("%4d: sizeof (struct GB_Semiring_opaque)\n",
             (int) sizeof (struct GB_Semiring_opaque)) ;
    printf ("%4d: sizeof (struct GB_Scalar_opaque)\n",
             (int) sizeof (struct GB_Scalar_opaque)) ;
    printf ("%4d: sizeof (struct GB_Vector_opaque)\n",
             (int) sizeof (struct GB_Vector_opaque)) ;
    printf ("%4d: sizeof (struct GB_Matrix_opaque)\n",
             (int) sizeof (struct GB_Matrix_opaque)) ;
    printf ("%4d: sizeof (struct GB_Descriptor_opaque)\n",
             (int) sizeof (struct GB_Descriptor_opaque)) ;
    printf ("%4d: sizeof (struct GB_Context_opaque)\n",
             (int) sizeof (struct GB_Context_opaque)) ;
    printf ("%4d: sizeof (struct GB_Global_opaque)\n",
             (int) sizeof (struct GB_Global_opaque)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test25:  all tests passed\n\n") ;
}

