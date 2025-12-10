//------------------------------------------------------------------------------
// GB_mex_test24: JIT error handling
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "../Source/jitifyer/GB_stringify.h"

#define FREE_ALL ;
#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

void mygunk1 (bool *z, const bool *x) ;
void mygunk1 (bool *z, const bool *x) { (*z) = !(*x) ; }
#define MYGUNK1_DEFN \
"void gb_mygunk (bool *z, const bool *x) { (*z) = !(*x) ; }"

void mygunk2 (bool *z, const bool *x) ;
void mygunk2 (bool *z, const bool *x) { (*z) = (*x) ; }
#define MYGUNK2_DEFN \
"void gb_mygunk (bool *z, const bool *x) { (*z) = (*x) ; }"

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
    // create an op and then change it
    //--------------------------------------------------------------------------

    GrB_Type type1, type2 ;
    GrB_UnaryOp op1, op2 ;
    OK (GxB_set (GxB_BURBLE, true)) ;
    int c ;
    OK (GxB_get (GxB_JIT_C_CONTROL, &c)) ;
    if (c == GxB_JIT_ON)
    {
        printf ("JIT on\n") ;
        OK (GxB_UnaryOp_new (&op1, NULL, GrB_BOOL, GrB_BOOL, "gb_mygunk",
            MYGUNK1_DEFN)) ;
        OK (GxB_UnaryOp_new (&op2, NULL, GrB_BOOL, GrB_BOOL, "gb_mygunk",
            MYGUNK2_DEFN)) ;
        GxB_print (op1, 3) ;
        GxB_print (op2, 3) ;
        GrB_free (&op1) ;
        GrB_free (&op2) ;
        OK (GxB_Type_new (&type1, 0, "gb_mygunktype",
            "typedef int32_t gb_mygunktype ;")) ;
        OK (GxB_Type_new (&type2, 0, "gb_mygunktype",
            "typedef int64_t gb_mygunktype ;")) ;
        GxB_print (type1, 3) ;
        GxB_print (type2, 3) ;
        GrB_free (&type1) ;
        GrB_free (&type2) ;
        OK (GxB_Type_new (&type2, 0, "gb_mygunktype",
            "typedef int64_t gb_mygunktype ;")) ;
        GrB_free (&type2) ;

        OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_OFF)) ;
        OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_LOAD)) ;
        GrB_Info expected = GrB_INVALID_VALUE ;
        ERR (GxB_Type_new (&type1, 0, "gb_mygunktype",
            "typedef int32_t gb_mygunktype ;")) ;
        OK (GxB_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    }
    else
    {
        printf ("JIT off\n") ;
        GrB_Info expected = GrB_NULL_POINTER ;
        ERR (GxB_UnaryOp_new (&op1, NULL, GrB_BOOL, GrB_BOOL, "gb_mygunk",
            MYGUNK1_DEFN)) ;
        ERR (GxB_UnaryOp_new (&op2, NULL, GrB_BOOL, GrB_BOOL, "gb_mygunk",
            MYGUNK2_DEFN)) ;
        expected = GrB_INVALID_VALUE ;
        ERR (GxB_Type_new (&type1, 0, "gb_mygunktype",
            "typedef int32_t gb_mygunktype ;")) ;
        ERR (GxB_Type_new (&type2, 0, "gb_mygunktype",
            "typedef int64_t gb_mygunktype ;")) ;
    }

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test24:  all tests passed\n\n") ;
}

