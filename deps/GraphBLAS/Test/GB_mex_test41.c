//------------------------------------------------------------------------------
// GB_mex_test41: test iso get/set
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free (&A) ;              \
    GrB_Vector_free (&V) ;              \
}

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
    GrB_Matrix A = NULL ;
    GrB_Vector V = NULL ;
    uint64_t n = 4 ;
    bool iso ;
    int iso2 ;

    //--------------------------------------------------------------------------
    // test iso_hint
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&A, GrB_FP64, n, n)) ;
    OK (GrB_Vector_new (&V, GrB_FP64, n)) ;
    OK (GxB_Matrix_fprint (A, "A empty", 5, NULL)) ;
    OK (GxB_Vector_fprint (V, "V empty", 5, NULL)) ;

    // create iso-valued matrix A and vector V
    OK (GrB_Matrix_assign_FP64 (A, NULL, NULL, 3.14159,
        GrB_ALL, n, GrB_ALL, n, NULL)) ;
    OK (GxB_Matrix_fprint (A, "A iso", 5, NULL)) ;
    iso2 = false ;
    OK (GrB_Matrix_get_INT32 (A, &iso2, GxB_ISO)) ;
    CHECK (iso2) ;
    iso = false ;
    OK (GxB_Matrix_iso (&iso, A)) ;
    CHECK (iso) ;

    OK (GrB_Vector_assign_FP64 (V, NULL, NULL, 15.9, GrB_ALL, n, NULL)) ;
    OK (GxB_Vector_fprint (V, "V iso", 5, NULL)) ;
    iso2 = false ;
    OK (GrB_Vector_get_INT32 (V, &iso2, GxB_ISO)) ;
    CHECK (iso2) ;
    iso = false ;
    OK (GxB_Vector_iso (&iso, V)) ;
    CHECK (iso) ;

    // make them non-iso
    OK (GrB_Matrix_set_INT32 (A, 0, GxB_ISO)) ;
    OK (GxB_Matrix_fprint (A, "A not iso", 5, NULL)) ;
    iso2 = true ;
    OK (GrB_Matrix_get_INT32 (A, &iso2, GxB_ISO)) ;
    CHECK (!iso2) ;
    iso = true ;
    OK (GxB_Matrix_iso (&iso, A)) ;
    CHECK (!iso) ;

    OK (GrB_Vector_set_INT32 (V, 0, GxB_ISO)) ;
    OK (GxB_Vector_fprint (V, "V not iso", 5, NULL)) ;
    iso2 = true ;
    OK (GrB_Vector_get_INT32 (V, &iso2, GxB_ISO)) ;
    CHECK (!iso2) ;
    iso = true ;
    OK (GxB_Vector_iso (&iso, V)) ;
    CHECK (!iso) ;

    // make them iso
    OK (GrB_Matrix_set_INT32 (A, 1, GxB_ISO)) ;
    OK (GxB_Matrix_fprint (A, "A iso again", 5, NULL)) ;
    iso2 = false ;
    OK (GrB_Matrix_get_INT32 (A, &iso2, GxB_ISO)) ;
    CHECK (iso2) ;
    iso = false ;
    OK (GxB_Matrix_iso (&iso, A)) ;
    CHECK (iso) ;

    OK (GrB_Vector_set_INT32 (V, 1, GxB_ISO)) ;
    OK (GxB_Vector_fprint (V, "V iso again", 5, NULL)) ;
    OK (GrB_Vector_get_INT32 (V, &iso2, GxB_ISO)) ;
    CHECK (iso2) ;
    iso = false ;
    OK (GxB_Vector_iso (&iso, V)) ;
    CHECK (iso) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    FREE_ALL ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test41:  all tests passed\n\n") ;
}

