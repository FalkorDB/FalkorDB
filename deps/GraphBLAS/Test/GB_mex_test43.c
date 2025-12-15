//------------------------------------------------------------------------------
// GB_mex_test43: test GxB_PRINT_FUNCTION
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL                        \
{                                       \
    GrB_free (&A) ;                     \
    GrB_free (&t) ;                     \
    GrB_free (&type) ;                  \
}

//------------------------------------------------------------------------------
// gb_43_type and its print function
//------------------------------------------------------------------------------

#define GB_43_TYPE \
"typedef struct { double x ; int32_t y, z ; } gb_43_type ;"
 typedef struct { double x ; int32_t y, z ; } gb_43_type ;

int64_t gb_43_print         // print the gb_43_type
(
    // output:
    char *string,           // value is printed to the string 
    // input:
    size_t string_size,     // size of the string array
    const void *value,      // value to print
    int verbose             // if >0, print verbosely; else tersely
) ;

int64_t gb_43_print         // print the gb_43_type
(
    // output:
    char *string,           // value is printed to the string 
    // input:
    size_t string_size,     // size of the string array
    const void *value,      // value to print
    int verbose             // if >0, print verbosely; else tersely
)
{
    gb_43_type *g = (gb_43_type *) value ;
    if (g->z == 42)
    { 
        // tell GraphBLAS the string needs to be longer
        if (string_size < 8000)
        {
            return (8000) ;
        }
        return ((int64_t) snprintf (string, string_size, "the answer is 42")) ;
    }
    else if (g->z < 0)
    {
        // trigger a failure
        return (-1) ;
    }
    else
    {
        // typical case
        return ((int64_t) snprintf (string, string_size,
                verbose ?  "(x: %.16g, y: %d, z: %d)" : "(x: %g, y: %d, z: %d)",
                g->x, g->y, g->z)) ;
    }
}

//------------------------------------------------------------------------------
// GB_mex_test43 mexFunction
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
    GrB_Type type = NULL ;
    GrB_Scalar t = NULL ;
    GrB_Matrix A = NULL ;
    bool malloc_debug = GB_mx_get_global (true) ;
    gb_43_type stuff = { .x = 1.2, .y = 3, .z = 42 } ;
    gb_43_type bad   = { .x = 3.1, .y = 4, .z = -1 } ;
    gb_43_type good  = { .x = 9.9, .y = 7, .z =  1 } ;
    void *p = NULL ;

    //--------------------------------------------------------------------------
    // create a new type and set its print function
    //--------------------------------------------------------------------------

    OK (GxB_Type_new (&type, sizeof (gb_43_type), "gb_43_type", GB_43_TYPE)) ;
    OK (GrB_Type_set_VOID (type, &gb_43_print, GxB_PRINT_FUNCTION,
        sizeof (&gb_43_print))) ;

    OK (GrB_Type_get_VOID (type, &p, GxB_PRINT_FUNCTION)) ;
    CHECK (p == gb_43_print) ;

    //--------------------------------------------------------------------------
    // create a scalar and matrix to print
    //--------------------------------------------------------------------------

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    OK (GrB_Scalar_new (&t, type)) ;
    OK (GrB_Scalar_setElement_UDT (t, (void *) &stuff)) ;
    METHOD (GxB_Scalar_fprint (t, "t", GxB_COMPLETE_VERBOSE, stdout)) ;

    // create A as an iso-valued matrix
    OK (GrB_Matrix_new (&A, type, 4, 3)) ;
    OK (GrB_Matrix_assign_Scalar (A, NULL, NULL, t, GrB_ALL, 4, GrB_ALL, 3,
        NULL)) ;
    METHOD (GxB_Matrix_fprint (A, "A iso", GxB_COMPLETE_VERBOSE, stdout)) ;

    OK (GrB_Matrix_setElement_UDT (A, (void *) &good, 2, 2)) ;
    METHOD (GxB_Matrix_fprint (A, "A good", GxB_COMPLETE_VERBOSE, stdout)) ;

    OK (GrB_Matrix_setElement_UDT (A, (void *) &bad, 2, 2)) ;
    info = (GxB_Matrix_fprint (A, "A bad", GxB_COMPLETE_VERBOSE, stdout)) ;
    printf ("expected info for A(2,2) is -3: %d\n", info) ;
    CHECK (info == GrB_INVALID_VALUE) ;

    OK (GrB_Matrix_clear (A)) ;
    OK (GrB_Matrix_setElement_UDT (A, (void *) &good, 0, 0)) ;
    OK (GrB_Matrix_setElement_UDT (A, (void *) &stuff, 1, 1)) ;
    OK (GrB_Matrix_setElement_UDT (A, (void *) &stuff, 1, 2)) ;
    METHOD (GxB_Matrix_fprint (A, "A pending", GxB_COMPLETE_VERBOSE, stdout)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    FREE_ALL ;
    GB_mx_put_global (true) ;
    printf ("\nGB_mex_test43:  all tests passed\n\n") ;
}

