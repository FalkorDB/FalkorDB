//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/wildtype_demo: an arbitrary user-defined type
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Each "scalar" entry of this type consists of a 4x4 matrix and a string of
// length 64.

#include "GraphBLAS.h"

#if defined __INTEL_COMPILER
#pragma warning (disable: 58 167 144 177 181 186 188 589 593 869 981 1418 1419 1572 1599 2259 2282 2557 2547 3280 )
#elif defined __GNUC__
#pragma GCC diagnostic ignored "-Wunused-parameter"
#if defined ( __cplusplus )
#pragma GCC diagnostic ignored "-Wwrite-strings"
#else
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
#endif
#endif

//------------------------------------------------------------------------------
// the wildtype
//------------------------------------------------------------------------------

typedef struct
{
    double stuff [4][4] ;
    char whatstuff [64] ;
}
wildtype ;                      // C version of wildtype

// repeat the typedef as a string, to give to GraphBLAS
#define WILDTYPE_DEFN           \
"typedef struct "               \
"{ "                            \
   "double stuff [4][4] ; "     \
   "char whatstuff [64] ; "     \
"} "                            \
"wildtype ;"

GrB_Type WildType ;             // GraphBLAS version of wildtype

//------------------------------------------------------------------------------
// wildtype_print: print a "scalar" value of wildtype
//------------------------------------------------------------------------------

void wildtype_print (const wildtype *x, const char *name)
{
    printf ("\na wildtype scalar: %s [%s]\n", name, x->whatstuff) ;
    for (int i = 0 ; i < 4 ; i++)
    {
        for (int j = 0 ; j < 4 ; j++)
        {
            printf ("%10.1f ", x->stuff [i][j]) ;
        }
        printf ("\n") ;
    }
}

//------------------------------------------------------------------------------
// wildtype_print_matrix: print a matrix of wildtype scalars
//------------------------------------------------------------------------------

// This examines each entry of A, which is costly if A is very large.  A better
// method would extract all the tuples via GrB_Matrix_extractTuples, and then
// to print those, or to use the GxB_*print methods.  This function is just to
// illustrate the GrB_Matrix_extractElement_UDT method.

void wildtype_print_matrix (GrB_Matrix A, char *name)
{
    printf ("\nPrinting the matrix with GxB_Matrix_fprint:\n") ;
    GxB_Matrix_fprint (A, name, GxB_COMPLETE, stdout) ;
    GrB_Type type ;
    GxB_Matrix_type (&type, A) ;
    if (type != WildType)
    {
        printf ("\nThe matrix %s is not wild enough to print.\n", name) ;
        return ;
    }
    GrB_Index nvals, nrows, ncols ;
    GrB_Matrix_nvals (&nvals, A) ;
    GrB_Matrix_nrows (&nrows, A) ;
    GrB_Matrix_ncols (&ncols, A) ;
    printf ("\n============= printing the WildType matrix: %s (%d-by-%d"
        " with %d entries)\n", name, (int) nrows, (int) ncols, (int) nvals) ;
    for (int i = 0 ; i < (int) nrows ; i++)
    {
        for (int j = 0 ; j < (int) ncols ; j++)
        {
            wildtype scalar ;
            GrB_Info info = GrB_Matrix_extractElement_UDT (&scalar, A, i, j) ;
            if (info == GrB_SUCCESS)
            {
                printf ("\n----------- %s(%d,%d):\n", name, i, j) ;
                wildtype_print (&scalar, "") ;
            }
        }
    }
    printf ("\n============= that was the WildType matrix %s\n", name) ;
}

//------------------------------------------------------------------------------
// add two wildtype "scalars"
//------------------------------------------------------------------------------

// strcpy is not available for use in a GPU kernel, so a while loop is used.

void wildadd (wildtype *z, const wildtype *x, const wildtype *y)
{
    for (int i = 0 ; i < 4 ; i++)
    {
        for (int j = 0 ; j < 4 ; j++)
        {
            z->stuff [i][j] = x->stuff [i][j] + y->stuff [i][j] ;
        }
    }
    const char *psrc = "this was added" ;
    char *pdst = z->whatstuff ;
    while ((*pdst++ = *psrc++)) ;
}

// The newlines (\n) in the definition below are optional.  They just make
// GxB_print output readable.  This example defines wildadd as either a
// macro or as a function.

#define WILDADD_DEFN                                                        \
"void wildadd (wildtype *z, const wildtype *x, const wildtype *y)       \n" \
"{                                                                      \n" \
"   for (int i = 0 ; i < 4 ; i++)                                       \n" \
"   {                                                                   \n" \
"       for (int j = 0 ; j < 4 ; j++)                                   \n" \
"       {                                                               \n" \
"           z->stuff [i][j] = x->stuff [i][j] + y->stuff [i][j] ;       \n" \
"       }                                                               \n" \
"   }                                                                   \n" \
"   const char *psrc = \"this was added\" ;                             \n" \
"   char *pdst = z->whatstuff ;                                         \n" \
"   while ((*pdst++ = *psrc++)) ;                                       \n" \
"}"

//------------------------------------------------------------------------------
// multiply two wildtypes "scalars"
//------------------------------------------------------------------------------

void wildmult (wildtype *z, const wildtype *x, const wildtype *y)
{
    for (int i = 0 ; i < 4 ; i++)
    {
        for (int j = 0 ; j < 4 ; j++)
        {
            z->stuff [i][j] = 0 ;
            for (int k = 0 ; k < 4 ; k++)
            {
                z->stuff [i][j] += (x->stuff [i][k] * y->stuff [k][j]) ;
            }
        }
    }
    const char *psrc = "this was multiplied" ;
    char *pdst = z->whatstuff ;
    while ((*pdst++ = *psrc++)) ;
}

#define WILDMULT_DEFN                                                       \
"void wildmult (wildtype *z, const wildtype *x, const wildtype *y)      \n" \
"{                                                                      \n" \
"   for (int i = 0 ; i < 4 ; i++)                                       \n" \
"   {                                                                   \n" \
"       for (int j = 0 ; j < 4 ; j++)                                   \n" \
"       {                                                               \n" \
"           z->stuff [i][j] = 0 ;                                       \n" \
"           for (int k = 0 ; k < 4 ; k++)                               \n" \
"           {                                                           \n" \
"               z->stuff [i][j] += (x->stuff [i][k] * y->stuff [k][j]) ;\n" \
"           }                                                           \n" \
"       }                                                               \n" \
"   }                                                                   \n" \
"   const char *psrc = \"this was multiplied\" ;                        \n" \
"   char *pdst = z->whatstuff ;                                         \n" \
"   while ((*pdst++ = *psrc++)) ;                                       \n" \
"}"

//------------------------------------------------------------------------------
// wildtype main program
//------------------------------------------------------------------------------

#define LINE \
"----------------------------------------------------------------------------\n"
#define LINE2 \
"============================================================================\n"

int main (void)
{

    // start GraphBLAS
    #if 1
    GrB_init (GrB_NONBLOCKING) ;
    #else
    GxB_init (GxB_NONBLOCKING_GPU, NULL, NULL, NULL, NULL, NULL) ;
    GxB_set (GxB_GPU_ID, 0) ;
    GB_Global_hack_set (2, 1) ; // always use the GPU
    #endif

    GxB_Global_Option_set (GxB_BURBLE, true) ;
    int nthreads ;
    GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads) ;
    fprintf (stderr, "wildtype demo: nthreads %d\n", nthreads) ;

    /* via #defines:
    fprintf (stderr, LINE2 "SuiteSparse:GraphBLAS Version %d.%d.%d, %s\n" LINE2
        "%s" LINE "License: %s" LINE "GraphBLAS API Version %d.%d.%d, %s"
        " (http://graphblas.org)\n%s" LINE2, GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR, GxB_IMPLEMENTATION_SUB,
        GxB_IMPLEMENTATION_DATE,  GxB_IMPLEMENTATION_ABOUT,
        GxB_IMPLEMENTATION_LICENSE, GxB_SPEC_MAJOR, GxB_SPEC_MINOR,
        GxB_SPEC_SUB, GxB_SPEC_DATE, GxB_SPEC_ABOUT) ;
    */

    char *library ;   GxB_Global_Option_get (GxB_LIBRARY_NAME,     &library) ;
    int version [3] ; GxB_Global_Option_get (GxB_LIBRARY_VERSION,  version) ;
    char *date ;      GxB_Global_Option_get (GxB_LIBRARY_DATE,     &date) ;
    char *about ;     GxB_Global_Option_get (GxB_LIBRARY_ABOUT,    &about) ;
    char *url ;       GxB_Global_Option_get (GxB_LIBRARY_URL,      &url) ;
    char *license ;   GxB_Global_Option_get (GxB_LIBRARY_LICENSE,  &license) ;
    char *cdate ;     GxB_Global_Option_get (GxB_LIBRARY_COMPILE_DATE, &cdate) ;
    char *ctime ;     GxB_Global_Option_get (GxB_LIBRARY_COMPILE_TIME, &ctime) ;
    int api_ver [3] ; GxB_Global_Option_get (GxB_API_VERSION,      api_ver) ;
    char *api_date ;  GxB_Global_Option_get (GxB_API_DATE,         &api_date) ;
    char *api_about ; GxB_Global_Option_get (GxB_API_ABOUT,        &api_about) ;
    char *api_url ;   GxB_Global_Option_get (GxB_API_URL,          &api_url) ;

    fprintf (stderr, LINE2 "%s Version %d.%d.%d, %s\n" LINE2 "%s"
        "(%s)\n" LINE "License:\n%s" LINE "GraphBLAS API Version %d.%d.%d, %s"
        " (%s)\n%s" LINE2,
        library, version [0], version [1], version [2], date, about, url,
        license, api_ver [0], api_ver [1], api_ver [2], api_date, api_url,
        api_about) ;
    fprintf (stderr, "compiled: %s %s\n", cdate, ctime) ;

    double hyper_switch ;
    GxB_Global_Option_get (GxB_HYPER_SWITCH, &hyper_switch) ;
    fprintf (stderr, "hyper switch: %g\n", hyper_switch) ;

    GxB_Format_Value format ;
    GxB_Global_Option_get (GxB_FORMAT, &format) ;
    fprintf (stderr, "format: %s\n", (format == GxB_BY_ROW) ? "CSR" : "CSC") ;

    GrB_Mode mode ;
    GxB_Global_Option_get (GxB_MODE, &mode) ;
    fprintf (stderr, "mode: %s\n", (mode == GrB_BLOCKING) ?
        "blocking" : "non-blocking") ;

    int nthreads_max ;
    GxB_Global_Option_get (GxB_GLOBAL_NTHREADS, &nthreads_max) ;
    fprintf (stderr, "max # of threads used internally: %d\n", nthreads_max) ;

    // create the WildType
    GxB_Type_new (&WildType, sizeof (wildtype), "wildtype", WILDTYPE_DEFN) ;
    GxB_Type_fprint (WildType, "WildType", GxB_COMPLETE, stdout) ;

    // get its properties
    size_t s ;
    GxB_Type_size (&s, WildType) ;
    printf ("WildType size: %d\n", (int) s) ;
    GxB_Type_fprint (WildType, "WildType", GxB_COMPLETE, stdout) ;

    // create a 10-by-10 WildType matrix, each entry is a 'scalar' WildType
    GrB_Matrix A ;
    GrB_Matrix_new (&A, WildType, 10, 10) ;

    wildtype scalar1, scalar2 ;
    memset (&scalar1, 0, sizeof (wildtype)) ;
    memset (&scalar2, 0, sizeof (wildtype)) ;
    for (int i = 0 ; i < 4 ; i++)
    {
        for (int j = 0 ; j < 4 ; j++)
        {
            scalar1.stuff [i][j] = 100*i + j ;
        }
    }
    strcpy (scalar1.whatstuff, "this is from scalar1") ;
    wildtype_print (&scalar1, "scalar1") ;

    // A(2,7) = scalar1
    strcpy (scalar1.whatstuff, "this is A(2,7)") ;
    GrB_Matrix_setElement_UDT (A, &scalar1, 2, 7) ;

    // A(3,7) = scalar1 modified
    scalar1.stuff [2][3] = 909 ;
    strcpy (scalar1.whatstuff, "this is A(3,7)") ;
    GrB_Matrix_setElement_UDT (A, &scalar1, 3, 7) ;

    // A(2,4) = scalar1 modified again
    scalar1.stuff [3][3] = 42 ;
    strcpy (scalar1.whatstuff, "this is A(2,4)") ;
    GrB_Matrix_setElement_UDT (A, &scalar1, 2, 4) ;

    // C = A'
    GrB_Matrix C ;
    GrB_Matrix_new (&C, WildType, 10, 10) ;
    GrB_transpose (C, NULL, NULL, A, NULL) ;

    // scalar2 = C(7,2)
    GrB_Info info = GrB_Matrix_extractElement_UDT (&scalar2, C, 7, 2) ;
    if (info == GrB_SUCCESS)
    {
        wildtype_print (&scalar2, "got scalar2 = C(7,2)") ;
    }
    strcpy (scalar2.whatstuff, "here is scalar2") ;

    // create the WildAdd operator
    GrB_BinaryOp WildAdd ;
    GxB_BinaryOp_new (&WildAdd, 
        (GxB_binary_function) wildadd, WildType, WildType, WildType,
        "wildadd", WILDADD_DEFN) ;
    GxB_BinaryOp_fprint (WildAdd, "WildAdd", GxB_COMPLETE, stdout) ;

    // create the WildMult operator
    GrB_BinaryOp WildMult ;
    GxB_BinaryOp_new (&WildMult, 
        (GxB_binary_function) wildmult, WildType, WildType, WildType,
        "wildmult", WILDMULT_DEFN) ;
    GxB_BinaryOp_fprint (WildMult, "WildMult", GxB_COMPLETE, stdout) ;

    // create a matrix B with B (7,2) = scalar2
    GrB_Matrix B ;
    GrB_Matrix_new (&B, WildType, 10, 10) ;
    for (int i = 0 ; i < 4 ; i++)
    {
        for (int j = 0 ; j < 4 ; j++)
        {
            scalar2.stuff [i][j] = (double) (j - i) + 0.5 ;
        }
    }
    wildtype_print (&scalar2, "scalar2") ;

    // B(7,2) = scalar2
    strcpy (scalar2.whatstuff, "this is B(7,2)") ;
    GrB_Matrix_setElement_UDT (B, &scalar2, 7, 2) ;

    // B(7,5) = scalar2 modified
    scalar2.stuff [0][0] = -1 ;
    strcpy (scalar2.whatstuff, "here is B(7,5)") ;
    GrB_Matrix_setElement_UDT (B, &scalar2, 7, 5) ;

    // B(4,2) = scalar2 changed 
    scalar2.stuff [0][3] = 77 ;
    strcpy (scalar2.whatstuff, "finally, B(4,2)") ;
    GrB_Matrix_setElement_UDT (B, &scalar2, 4, 2) ;

    // create the WildAdder monoid 
    GrB_Monoid WildAdder ;
    wildtype scalar_identity ;
    memset (&scalar_identity, 0, sizeof (wildtype)) ;
    for (int i = 0 ; i < 4 ; i++)
    {
        for (int j = 0 ; j < 4 ; j++)
        {
            scalar_identity.stuff [i][j] = 0 ;
        }
    }
    strcpy (scalar_identity.whatstuff, "identity") ;
    wildtype_print (&scalar_identity, "scalar_identity for the monoid") ;
    GrB_Monoid_new_UDT (&WildAdder, WildAdd, &scalar_identity) ;

    // create and print the InTheWild semiring
    GrB_Semiring InTheWild ;
    GrB_Semiring_new (&InTheWild, WildAdder, WildMult) ;
    GxB_Semiring_fprint (InTheWild, "InTheWild", GxB_COMPLETE, stdout) ;

    printf ("\nmultiplication C=A*B InTheWild semiring:\n") ;

    wildtype_print_matrix (A, "input A") ;
    wildtype_print_matrix (B, "input B") ;

    // C = A*B
    // Since there is no accum operator, this overwrites C with A*B; the old
    // content of C is gone.
    GrB_mxm (C, NULL, NULL, InTheWild, A, B, NULL) ;
    wildtype_print_matrix (C, "output C") ;

    // C<M> = C*C'
    printf ("\n------ C<M>=C*C'----------------------------------------\n") ;
    GrB_Matrix M ;
    GrB_Matrix_new (&M, GrB_BOOL, 10, 10) ;
    GrB_Matrix_setElement_BOOL (M, true, 2, 2) ;
    GrB_Matrix_setElement_BOOL (M, true, 2, 3) ;
    GrB_Matrix_setElement_BOOL (M, true, 3, 2) ;
    GrB_Matrix_setElement_BOOL (M, true, 3, 3) ;
    printf ("\nThe mask matrix M:\n") ;
    GxB_Matrix_fprint (M, "M", GxB_COMPLETE, stdout) ;

//  GxB_Global_Option_set (GxB_BURBLE, true) ;
    GrB_mxm (C, M, NULL, InTheWild, C, C, GrB_DESC_RST1) ;
    wildtype_print_matrix (C, "output C") ;

    // reduce C to a scalar using the WildAdder monoid
    wildtype sum ;
    memset (&sum, 0, sizeof (wildtype)) ;
    GrB_Matrix_reduce_UDT (&sum, NULL, WildAdder, C, NULL) ;
    wildtype_print (&sum, "sum (first time)") ;

    // again, to test the JIT lookup
    memset (&sum, 0, sizeof (wildtype)) ;
    GrB_Matrix_reduce_UDT (&sum, NULL, WildAdder, C, NULL) ;
    wildtype_print (&sum, "sum (again)") ;
//  GxB_Global_Option_set (GxB_BURBLE, false) ;

//  for (int k = 0 ; k < 100 ; k++)
//  {
//      GrB_Matrix_reduce_UDT (&sum, NULL, WildAdder, C, NULL) ;
//  }

    // set C to column-oriented format
    GxB_Matrix_Option_set (C, GxB_FORMAT, GxB_BY_COL) ;
    printf ("\nC is now stored by column, but it looks just the same to the\n"
            "GraphBLAS user application.  The difference is opaque, in the\n"
            "internal data structure.\n") ;
    wildtype_print_matrix (C, "output C") ;

    // create a non-wild matrix D and try to print it
    GrB_Matrix D ;
    GrB_Matrix_new (&D, GrB_FP32, 10, 10) ;
    wildtype_print_matrix (D, "D") ;

    // apply some positional operators
    GrB_Matrix E ;
    GrB_Matrix_new (&E, GrB_INT64, 10, 10) ;

    GrB_Matrix_apply (E, NULL, NULL, GxB_POSITIONI_INT64, A, NULL) ;
    GxB_Matrix_fprint (E, "E (positional i)", GxB_COMPLETE, NULL) ;

    GrB_Matrix_apply (E, NULL, NULL, GxB_POSITIONJ_INT64, A, NULL) ;
    GxB_Matrix_fprint (E, "E (positional j)", GxB_COMPLETE, NULL) ;

    // do something invalid
    info = GrB_Matrix_eWiseAdd_BinaryOp (C, NULL, NULL, WildAdd, A, D, NULL) ;
    if (info != GrB_SUCCESS)
    {
        const char *s ;
        GrB_Matrix_error (&s, C) ;
        printf ("\nThis is supposed to fail, as a demo of GrB_error:\n%s\n", s);
    }

    // free everyting
    GrB_Matrix_free (&C) ;
    GrB_Matrix_free (&A) ;
    GrB_Matrix_free (&B) ;
    GrB_Matrix_free (&D) ;
    GrB_Matrix_free (&E) ;
    GrB_Matrix_free (&M) ;
    GrB_Semiring_free (&InTheWild) ;
    GrB_Monoid_free (&WildAdder) ;
    GrB_BinaryOp_free (&WildAdd) ;
    GrB_BinaryOp_free (&WildMult) ;
    GrB_Type_free (&WildType) ;

    GrB_finalize ( ) ;
}

