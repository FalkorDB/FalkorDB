//------------------------------------------------------------------------------
// GB_mex_test50: test GrB_select with a VALUE* op of another type than A
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GrB_select typecasts A(i,j) to op->xtype and the thunk to op->ytype.  In
// GraphBLAS v10.5.1 and earlier, if op->xtype was not the type of A, the
// factory kernels for the VALUE* ops compared A(i,j) with the thunk as if the
// thunk had the type of A (so an int8 A with GrB_VALUEEQ_FP64 and a thunk of
// 1.0 kept no entries).  Also, the VALUEEQ ops constructed C as iso with the
// thunk as its value, even when entries with different values were kept (a
// double A with GrB_VALUEEQ_INT64 and a thunk of 1 keeps both 1.0 and 1.5).

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL                    \
{                                   \
    GrB_Vector_free (&A) ;          \
    GrB_Vector_free (&B) ;          \
}

//------------------------------------------------------------------------------
// gb_test50_check: check C = select (A, op, y)
//------------------------------------------------------------------------------

// A has 3 entries, and C should contain A(i) if keep_i is true.

void gb_test50_check (GrB_Vector A, GrB_IndexUnaryOp op, double y,
    bool keep_0, bool keep_1, bool keep_2) ;
void gb_test50_check (GrB_Vector A, GrB_IndexUnaryOp op, double y,
    bool keep_0, bool keep_1, bool keep_2)
{
    GrB_Info info ;
    bool keep [3] = { keep_0, keep_1, keep_2 } ;
    GrB_Vector C = NULL ;
    GrB_Scalar Thunk = NULL ;
    OK (GrB_Vector_new (&C, A->type, 3)) ;
    OK (GrB_Scalar_new (&Thunk, op->ytype)) ;
    OK (GrB_Scalar_setElement_FP64 (Thunk, y)) ;
    OK (GrB_Vector_select_Scalar (C, NULL, NULL, op, A, Thunk, NULL)) ;

    // C(i) must be equal to A(i) if keep [i] is true; otherwise it must not
    // be present
    uint64_t nvals ;
    OK (GrB_Vector_nvals (&nvals, C)) ;
    uint64_t nkeep = 0 ;
    for (int i = 0 ; i < 3 ; i++)
    {
        double a = 0, c = 0 ;
        OK (GrB_Vector_extractElement_FP64 (&a, A, i)) ;
        info = GrB_Vector_extractElement_FP64 (&c, C, i) ;
        if (keep [i])
        {
            CHECK (info == GrB_SUCCESS) ;
            CHECK (c == a) ;
            nkeep++ ;
        }
        else
        {
            CHECK (info == GrB_NO_VALUE) ;
        }
    }
    CHECK (nvals == nkeep) ;

    GrB_Vector_free (&C) ;
    GrB_Scalar_free (&Thunk) ;
}

//------------------------------------------------------------------------------
// GB_mex_test50 mexFunction
//------------------------------------------------------------------------------

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
    GrB_Vector A = NULL, B = NULL ;

    //--------------------------------------------------------------------------
    // test select with A sparse, bitmap, and full
    //--------------------------------------------------------------------------

    int sparsity [3] = { GxB_SPARSE, GxB_BITMAP, GxB_FULL } ;

    for (int k = 0 ; k < 3 ; k++)
    {

        // A = int8 [1, 100, -5]
        OK (GrB_Vector_new (&A, GrB_INT8, 3)) ;
        OK (GrB_Vector_setElement_INT8 (A,   1, 0)) ;
        OK (GrB_Vector_setElement_INT8 (A, 100, 1)) ;
        OK (GrB_Vector_setElement_INT8 (A,  -5, 2)) ;
        OK (GrB_set (A, sparsity [k], GxB_SPARSITY_CONTROL)) ;
        OK (GrB_Vector_wait (A, GrB_MATERIALIZE)) ;

        // B = double [1, 1.5, 2]
        OK (GrB_Vector_new (&B, GrB_FP64, 3)) ;
        OK (GrB_Vector_setElement_FP64 (B, 1.0, 0)) ;
        OK (GrB_Vector_setElement_FP64 (B, 1.5, 1)) ;
        OK (GrB_Vector_setElement_FP64 (B, 2.0, 2)) ;
        OK (GrB_set (B, sparsity [k], GxB_SPARSITY_CONTROL)) ;
        OK (GrB_Vector_wait (B, GrB_MATERIALIZE)) ;

        // op of the same type as A
        gb_test50_check (A, GrB_VALUEEQ_INT8,    1, true,  false, false) ;
        gb_test50_check (B, GrB_VALUEEQ_FP64,    2, false, false, true ) ;

        // op of another type than A
        gb_test50_check (A, GrB_VALUEEQ_FP64,    1, true,  false, false) ;
        gb_test50_check (A, GrB_VALUEGE_FP64,  2.5, false, true,  false) ;
        gb_test50_check (A, GrB_VALUEEQ_INT64, 356, false, false, false) ;
        gb_test50_check (A, GrB_VALUELT_INT64, 300, true,  true,  true ) ;

        // (bool) A(i) is true for all entries; GB_select converts VALUEGT_BOOL
        // with a thunk of false to VALUEEQ_BOOL with a thunk of true
        gb_test50_check (A, GrB_VALUEEQ_BOOL,    1, true,  true,  true ) ;
        gb_test50_check (A, GrB_VALUEGT_BOOL,    0, true,  true,  true ) ;

        // (int64_t) B(i) is 1 for B(0) = 1.0 and B(1) = 1.5
        gb_test50_check (B, GrB_VALUEEQ_INT64,   1, true,  true,  false) ;

        FREE_ALL ;
    }

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    printf ("GB_mex_test50:  all tests passed\n") ;
}

