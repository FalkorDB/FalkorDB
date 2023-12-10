//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/gauss_demo: Gaussian integers
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GraphBLAS.h"

//------------------------------------------------------------------------------
// the Gaussian integer: real and imaginary parts
//------------------------------------------------------------------------------

typedef struct
{
    int32_t real ;
    int32_t imag ;
}
gauss ;

// repeat the typedef as a string, to give to GraphBLAS
#define GAUSS_DEFN              \
"typedef struct "               \
"{ "                            \
   "int32_t real ; "            \
   "int32_t imag ; "            \
"} "                            \
"gauss ;"

typedef struct
{
    int32_t real ;
}
badgauss ;

// just to test the JIT: same 'gauss' name but different definition
#define BAD_GAUSS_DEFN          \
"typedef struct "               \
"{ "                            \
   "int32_t real ; "            \
"} "                            \
"gauss ;"

//------------------------------------------------------------------------------
// addgauss: add two Gaussian integers
//------------------------------------------------------------------------------

// z, x, and/or y can be aliased, but the computation is correct in that case.

void addgauss (gauss *z, const gauss *x, const gauss *y)
{
    z->real = x->real + y->real ;
    z->imag = x->imag + y->imag ;
}

#define ADDGAUSS_DEFN                                           \
"void addgauss (gauss *z, const gauss *x, const gauss *y)   \n" \
"{                                                          \n" \
"    z->real = x->real + y->real ;                          \n" \
"    z->imag = x->imag + y->imag ;                          \n" \
"}"

void badaddgauss (gauss *z, const gauss *x, const gauss *y)
{
    z->real = x->real + y->real ;
    z->imag = -911 ;
}

// just to test the JIT: same name but different definition
#define BAD_ADDGAUSS_DEFN                                       \
"void addgauss (gauss *z, const gauss *x, const gauss *y)   \n" \
"{                                                          \n" \
"    z->real = x->real + y->real ;                          \n" \
"    z->imag = -911 ;                                       \n" \
"}"

//------------------------------------------------------------------------------
// multgauss: multiply two Gaussian integers
//------------------------------------------------------------------------------

// z, x, and/or y can be aliased, so temporary variables zreal and zimag
// are required.

void multgauss (gauss *z, const gauss *x, const gauss *y)
{
    int32_t zreal = x->real * y->real - x->imag * y->imag ;
    int32_t zimag = x->real * y->imag + x->imag * y->real ;
    z->real = zreal ;
    z->imag = zimag ;
}

#define MULTGAUSS_DEFN                                          \
"void multgauss (gauss *z, const gauss *x, const gauss *y)  \n" \
"{                                                          \n" \
"    int32_t zreal = x->real * y->real - x->imag * y->imag ;\n" \
"    int32_t zimag = x->real * y->imag + x->imag * y->real ;\n" \
"    z->real = zreal ;                                      \n" \
"    z->imag = zimag ;                                      \n" \
"}"

//------------------------------------------------------------------------------
// realgauss: real part of a Gaussian integer
//------------------------------------------------------------------------------

void realgauss (int32_t *z, const gauss *x)
{
    (*z) = x->real ;
}

#define REALGAUSS_DEFN                                          \
"void realgauss (int32_t *z, const gauss *x)                \n" \
"{                                                          \n" \
"    (*z) = x->real ;                                       \n" \
"}"

//------------------------------------------------------------------------------
// ijgauss: Gaussian positional op
//------------------------------------------------------------------------------

void ijgauss (int64_t *z, const gauss *x, GrB_Index i, GrB_Index j, 
    const gauss *y)
{
    (*z) = x->real + y->real + i - j ;
}

#define IJGAUSS_DEFN                                                        \
"void ijgauss (int64_t *z, const gauss *x, GrB_Index i, GrB_Index j,    \n" \
"    const gauss *y)                                                    \n" \
"{                                                                      \n" \
"    (*z) = x->real + y->real + i - j ;                                 \n" \
"}"

//------------------------------------------------------------------------------
// printgauss: print a Gauss matrix
//------------------------------------------------------------------------------

// This is a very slow way to print a large matrix, so using this approach is
// not recommended for large matrices.  However, it looks nice for this demo
// since the matrix is small.

#undef  OK
#define OK(x)                   \
{                               \
    if (!(x))                   \
    {                           \
        printf ("info: %d error! Line %d\n", info, __LINE__)  ; \
        fflush (stdout) ;       \
        abort ( ) ;             \
    }                           \
}

#undef  TRY
#define TRY(method)             \
{                               \
    GrB_Info info = method ;    \
    OK (info == GrB_SUCCESS) ;  \
}

void printgauss (GrB_Matrix A, char *name)
{
    // print the matrix
    GrB_Index m, n ;
    TRY (GrB_Matrix_nrows (&m, A)) ;
    TRY (GrB_Matrix_ncols (&n, A)) ;
    printf ("\n%s\nsize: %d-by-%d\n", name, (int) m, (int) n) ;
    for (int i = 0 ; i < m ; i++)
    {
        printf ("row %2d: ", i) ;
        for (int j = 0 ; j < n ; j++)
        {
            gauss a ;
            GrB_Info info = GrB_Matrix_extractElement_UDT (&a, A, i, j) ;
            if (info == GrB_NO_VALUE)
            {
                printf ("      .     ") ;
            }
            else if (info == GrB_SUCCESS)
            {
                printf (" (%4d,%4d)", a.real, a.imag) ;
            }
            else TRY (GrB_PANIC) ;
        }
        printf ("\n") ;
    }
    printf ("\n") ;
}

//------------------------------------------------------------------------------
// gauss main program
//------------------------------------------------------------------------------

int main (void)
{
    // start GraphBLAS
    GrB_Info info ;
    TRY (GrB_init (GrB_NONBLOCKING)) ;
    TRY (GxB_Global_Option_set (GxB_BURBLE, true)) ;
    TRY (GxB_Global_Option_set (GxB_JIT_USE_CMAKE, true)) ;
    printf ("Gauss demo.  Note that all transposes are array transposes,\n"
        "not matrix (conjugate) transposes.\n\n") ;

//  try changing the error log file for compiler errors
//  TRY (GxB_Global_Option_set (GxB_JIT_ERROR_LOG, "/tmp/grb_jit_errors.txt")) ;

//  try changing the cache path
//  TRY (GxB_Global_Option_set (GxB_JIT_CACHE_PATH, "/home/faculty/d/davis/mycache")) ;

    TRY (GxB_Context_fprint (GxB_CONTEXT_WORLD, "World", GxB_COMPLETE, stdout)) ;
    char *compiler, *cache, *flags, *link, *libraries, *preface ;
    int control ;
    TRY (GxB_Global_Option_get (GxB_JIT_C_COMPILER_NAME, &compiler)) ;
    TRY (GxB_Global_Option_get (GxB_JIT_C_COMPILER_FLAGS, &flags)) ;
    TRY (GxB_Global_Option_get (GxB_JIT_C_LINKER_FLAGS, &link)) ;
    TRY (GxB_Global_Option_get (GxB_JIT_C_LIBRARIES, &libraries)) ;
    TRY (GxB_Global_Option_get (GxB_JIT_C_PREFACE, &preface)) ;
    TRY (GxB_Global_Option_get (GxB_JIT_CACHE_PATH, &cache)) ;
    TRY (GxB_Global_Option_get (GxB_JIT_C_CONTROL, &control)) ;
    printf ("JIT configuration: ------------------\n") ;
    printf ("JIT C compiler:   [%s]\n", compiler) ;
    printf ("JIT C flags:      [%s]\n", flags) ;
    printf ("JIT C link flags: [%s]\n", link) ;
    printf ("JIT C libraries:  [%s]\n", libraries) ;
    printf ("JIT C preface:    [%s]\n", preface) ;
    printf ("JIT cache:        [%s]\n", cache) ;
    printf ("JIT C control:    [%d]\n", control) ;
    TRY (GxB_Global_Option_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;
    TRY (GxB_Global_Option_get (GxB_JIT_C_CONTROL, &control)) ;
    printf ("JIT C control:    [%d] reset\n", control) ;
    printf ("-------------------------------------\n\n") ;

    // revise the header for each JIT kernel; this is not required but appears
    // here just as a demo of the feature.
    TRY (GxB_Global_Option_set (GxB_JIT_C_PREFACE,
        "// kernel generated by gauss_demo.c\n"
        "#include <math.h>\n")) ;
    TRY (GxB_Global_Option_get (GxB_JIT_C_PREFACE, &preface)) ;
    printf ("JIT C preface (revised):\n%s\n", preface) ;

    // create the Gauss type but do it wrong the first time.  This will always
    // require a new JIT kernel to be compiled: if this is the first run of
    // this demo, the cache folder is empty.  Otherwise, the good gauss type
    // will be left in the cache folder from a prior run of this program, and
    // its type defintion does not match this one.  The burble will say "jit:
    // loaded but must recompile" in this case.
    GrB_Type BadGauss = NULL ;
    info = GxB_Type_new (&BadGauss, 0, "gauss", BAD_GAUSS_DEFN) ;
    if (info != GrB_SUCCESS)
    {
        // JIT disabled
        printf ("JIT: unable to determine type size: set it to %d\n",
            (int) sizeof (badgauss)) ;
        TRY (GrB_Type_new (&BadGauss, sizeof (badgauss))) ;
    }
    TRY (GxB_Type_fprint (BadGauss, "BadGauss", GxB_COMPLETE, stdout)) ;
    size_t sizeof_gauss ;
    TRY (GxB_Type_size (&sizeof_gauss, BadGauss)) ;
    OK (sizeof_gauss == sizeof (badgauss)) ;
    GrB_Type_free (&BadGauss) ;

    // create the Gauss type, and let the JIT determine the size.  This causes
    // an intentional name collision.  The new 'gauss' type does not match the
    // old one (above), and this will be safely detected.  The burble will say
    // "(jit type: changed)" and the JIT kernel will be recompiled.  The
    // Gauss type is created twice, just to exercise the JIT.
    GrB_Type Gauss = NULL ;
    for (int trial = 0 ; trial <= 1 ; trial++)
    {
        // free the type and create it yet again, to test the JIT again
        GrB_Type_free (&Gauss) ;
        info = GxB_Type_new (&Gauss, 0, "gauss", GAUSS_DEFN) ;
        if (info != GrB_SUCCESS)
        {
            // JIT disabled
            printf ("JIT: unable to determine type size: set it to %d\n",
                (int) sizeof (gauss)) ;
            TRY (GrB_Type_new (&Gauss, sizeof (gauss))) ;
        }
        TRY (GxB_Type_fprint (Gauss, "Gauss", GxB_COMPLETE, stdout)) ;
        TRY (GxB_Type_size (&sizeof_gauss, Gauss)) ;
//      printf ("sizeof_gauss  %lu %lu\n", sizeof_gauss, sizeof (gauss)) ;
        OK (sizeof_gauss == sizeof (gauss)) ;
    }

    // create the BadAddGauss operator; use a NULL function pointer to test the
    // JIT.  Like the BadGauss type, this will always require a JIT
    // compilation, because the type will not match the good 'addgauss'
    // definition from a prior run of this demo.
    GrB_BinaryOp BadAddGauss = NULL ; 
    info = GxB_BinaryOp_new (&BadAddGauss, NULL,
        Gauss, Gauss, Gauss, "addgauss", BAD_ADDGAUSS_DEFN) ;
    if (info != GrB_SUCCESS)
    {
        // JIT disabled
        printf ("JIT: unable to compile the BadAddGauss kernel\n") ;
        TRY (GrB_BinaryOp_new (&BadAddGauss, (void *) badaddgauss,
            Gauss, Gauss, Gauss)) ;
    }
    TRY (GxB_BinaryOp_fprint (BadAddGauss, "BadAddGauss", GxB_COMPLETE,
        stdout)) ;
    GrB_BinaryOp_free (&BadAddGauss) ;

    // create the AddGauss operator; use a NULL function pointer to test the
    // JIT.  Causes an intentional name collision because of reusing the name
    // 'addgauss' with a different definition.  This is safely detected and
    // the kernel is recompiled.  The operator is created twice to exercise
    // the JIT.  The first trial will report "jit op: changed" and the 2nd
    // will say "jit op: ok".
    GrB_BinaryOp AddGauss = NULL ; 
    for (int trial = 0 ; trial <= 1 ; trial++)
    {
        GrB_BinaryOp_free (&AddGauss) ;
        info = GxB_BinaryOp_new (&AddGauss, NULL,
            Gauss, Gauss, Gauss, "addgauss", ADDGAUSS_DEFN) ;
        if (info != GrB_SUCCESS)
        {
            // JIT disabled
            printf ("JIT: unable to compile the AddGauss kernel\n") ;
            TRY (GrB_BinaryOp_new (&AddGauss, (void *) addgauss,
                Gauss, Gauss, Gauss)) ;
        }
        TRY (GxB_BinaryOp_fprint (AddGauss, "AddGauss", GxB_COMPLETE, stdout)) ;
    }

//  printf ("JIT: off\n") ;
//  TRY (GxB_Global_Option_set (GxB_JIT_C_CONTROL, GxB_JIT_OFF)) ;
//  printf ("JIT: on\n") ;
//  TRY (GxB_Global_Option_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;

    // create the AddMonoid
    gauss zero ;
    zero.real = 0 ;
    zero.imag = 0 ;
    GrB_Monoid AddMonoid ;
    TRY (GrB_Monoid_new_UDT (&AddMonoid, AddGauss, &zero)) ;
    TRY (GxB_Monoid_fprint (AddMonoid, "AddMonoid", GxB_COMPLETE, stdout)) ;

    // create the MultGauss operator
    GrB_BinaryOp MultGauss ;
    TRY (GxB_BinaryOp_new (&MultGauss, (void *) multgauss,
        Gauss, Gauss, Gauss, "multgauss", MULTGAUSS_DEFN)) ;
    TRY (GxB_BinaryOp_fprint (MultGauss, "MultGauss", GxB_COMPLETE, stdout)) ;

    // create the GaussSemiring
    GrB_Semiring GaussSemiring ;
    TRY (GrB_Semiring_new (&GaussSemiring, AddMonoid, MultGauss)) ;
    TRY (GxB_Semiring_fprint (GaussSemiring, "GaussSemiring", GxB_COMPLETE,
        stdout)) ;

    // create a 4-by-4 Gauss matrix, each entry A(i,j) = (i+1,2-j),
    // except A(0,0) is missing
    GrB_Matrix A, D ;
    TRY (GrB_Matrix_new (&A, Gauss, 4, 4)) ;
    TRY (GrB_Matrix_new (&D, GrB_BOOL, 4, 4)) ;
    gauss a ;
    for (int i = 0 ; i < 4 ; i++)
    {
        TRY (GrB_Matrix_setElement_BOOL (D, 1, i, i)) ;
        for (int j = 0 ; j < 4 ; j++)
        {
            if (i == 0 && j == 0) continue ;
            a.real = i+1 ;
            a.imag = 2-j ;
            TRY (GrB_Matrix_setElement_UDT (A, &a, i, j)) ;
        }
    }
    printgauss (A, "\n=============== Gauss A matrix:\n") ;

    // a = sum (A)
    TRY (GrB_Matrix_reduce_UDT (&a, NULL, AddMonoid, A, NULL)) ;
    printf ("\nsum (A) = (%d,%d)\n", a.real, a.imag) ;

    // A = A*A
    TRY (GrB_mxm (A, NULL, NULL, GaussSemiring, A, A, NULL)) ;
    printgauss (A, "\n=============== Gauss A = A^2 matrix:\n") ;

    // a = sum (A)
    TRY (GrB_Matrix_reduce_UDT (&a, NULL, AddMonoid, A, NULL)) ;
    printf ("\nsum (A^2) = (%d,%d)\n", a.real, a.imag) ;

    // C<D> = A*A' where A and D are sparse
    GrB_Matrix C ;
    TRY (GrB_Matrix_new (&C, Gauss, 4, 4)) ;
    printgauss (C, "\nGauss C empty matrix") ;
    TRY (GxB_Matrix_Option_set (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    TRY (GxB_Matrix_Option_set (D, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    TRY (GrB_mxm (C, D, NULL, GaussSemiring, A, A, GrB_DESC_T1)) ;
    printgauss (C, "\n=============== Gauss C = diag(AA') matrix:\n") ;

    // C = D*A
    GrB_Matrix_free (&D) ;
    TRY (GrB_Matrix_new (&D, Gauss, 4, 4)) ;
    TRY (GxB_Matrix_Option_set (A, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    TRY (GxB_Matrix_Option_set (D, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    TRY (GrB_Matrix_select_INT64 (D, NULL, NULL, GrB_DIAG, A, 0, NULL)) ;
    printgauss (D, "\nGauss D matrix") ;
    TRY (GrB_mxm (C, NULL, NULL, GaussSemiring, D, A, NULL)) ;
    printgauss (C, "\n=============== Gauss C = D*A matrix:\n") ;

    // convert D to bitmap then back to sparse
    TRY (GxB_Matrix_Option_set (D, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    TRY (GxB_Matrix_Option_set (D, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
    printgauss (D, "\nGauss D matrix (bitmap)") ;
    TRY (GxB_Matrix_fprint (D, "D", GxB_COMPLETE, stdout)) ;
    TRY (GxB_Matrix_Option_set (D, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    printgauss (D, "\nGauss D matrix (back to sparse)") ;
    TRY (GxB_Matrix_fprint (D, "D", GxB_COMPLETE, stdout)) ;

    // C = A*D
    TRY (GrB_mxm (C, NULL, NULL, GaussSemiring, A, D, NULL)) ;
    printgauss (C, "\n=============== Gauss C = A*D matrix:\n") ;

    // C = (1,2) then C += A*A' where C is full
    gauss ciso ;
    ciso.real = 1 ;
    ciso.imag = -2 ;
    TRY (GrB_Matrix_assign_UDT (C, NULL, NULL, &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== Gauss C = (1,-2) matrix:\n") ;
    printgauss (A, "\n=============== Gauss A matrix:\n") ;
    TRY (GrB_mxm (C, NULL, AddGauss, GaussSemiring, A, A, GrB_DESC_T1)) ;
    printgauss (C, "\n=============== Gauss C += A*A' matrix:\n") ;

    // C += B*A where B is full and A is sparse
    GrB_Matrix B ;
    TRY (GrB_Matrix_new (&B, Gauss, 4, 4)) ;
    TRY (GrB_Matrix_assign_UDT (B, NULL, NULL, &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (B, "\n=============== Gauss B = (1,-2) matrix:\n") ;
    TRY (GrB_mxm (C, NULL, AddGauss, GaussSemiring, B, A, NULL)) ;
    printgauss (C, "\n=============== Gauss C += B*A:\n") ;

    // C += A*B where B is full and A is sparse
    TRY (GrB_mxm (C, NULL, AddGauss, GaussSemiring, A, B, NULL)) ;
    printgauss (C, "\n=============== Gauss C += A*B:\n") ;

    // C = ciso+A
    TRY (GrB_Matrix_apply_BinaryOp1st_UDT (C, NULL, NULL, AddGauss,
        (void *) &ciso, A, NULL)) ;
    printgauss (C, "\n=============== Gauss C = (1,-2) + A:\n") ;

    // C = A*ciso
    TRY (GrB_Matrix_apply_BinaryOp2nd_UDT (C, NULL, NULL, MultGauss, A,
        (void *) &ciso, NULL)) ;
    printgauss (C, "\n=============== Gauss C = A*(1,-2):\n") ;

    // C = A'*ciso
    TRY (GrB_Matrix_apply_BinaryOp2nd_UDT (C, NULL, NULL, MultGauss, A,
        (void *) &ciso, GrB_DESC_T0)) ;
    printgauss (C, "\n=============== Gauss C = A'*(1,-2):\n") ;

    // C = ciso*A'
    TRY (GrB_Matrix_apply_BinaryOp1st_UDT (C, NULL, NULL, MultGauss,
        (void *) &ciso, A, GrB_DESC_T1)) ;
    printgauss (C, "\n=============== Gauss C = (1,-2)*A':\n") ;

    // create the RealGauss unary op
    GrB_UnaryOp RealGauss ;
    TRY (GxB_UnaryOp_new (&RealGauss, (void *) realgauss, GrB_INT32, Gauss,
        "realgauss", REALGAUSS_DEFN)) ;
    TRY (GxB_UnaryOp_fprint (RealGauss, "RealGauss", GxB_COMPLETE, stdout)) ;
    GrB_Matrix R ;
    TRY (GrB_Matrix_new (&R, GrB_INT32, 4, 4)) ;
    // R = RealGauss (C)
    TRY (GrB_Matrix_apply (R, NULL, NULL, RealGauss, C, NULL)) ;
    TRY (GxB_Matrix_fprint (R, "R", GxB_COMPLETE, stdout)) ;
    // R = RealGauss (C')
    printgauss (C, "\n=============== R = RealGauss (C')\n") ;
    TRY (GrB_Matrix_apply (R, NULL, NULL, RealGauss, C, GrB_DESC_T0)) ;
    TRY (GxB_Matrix_fprint (R, "R", GxB_COMPLETE, stdout)) ;
    GrB_Matrix_free (&R) ;

    // create the IJGauss IndexUnaryOp
    GrB_IndexUnaryOp IJGauss ;
    TRY (GxB_IndexUnaryOp_new (&IJGauss, (void *) ijgauss, GrB_INT64, Gauss,
        Gauss, "ijgauss", IJGAUSS_DEFN)) ;
    TRY (GrB_Matrix_new (&R, GrB_INT64, 4, 4)) ;
    printgauss (C, "\n=============== C \n") ;
    TRY (GrB_Matrix_apply_IndexOp_UDT (R, NULL, NULL, IJGauss, C,
        (void *) &ciso, NULL)) ;
    printf ("\nR = ijgauss (C)\n") ;
    TRY (GxB_Matrix_fprint (R, "R", GxB_COMPLETE, stdout)) ;
    GrB_Index I [100], J [100], rnvals = 100 ;
    double X [100] ;
    TRY (GrB_Matrix_extractTuples_FP64 (I, J, X, &rnvals, R)) ;
    for (int k = 0 ; k < rnvals ; k++)
    { 
        printf ("R (%d,%d) = %g\n", (int) I [k], (int) J [k], X [k]) ;
    }

    printgauss (C, "\n=============== C\n") ;
    TRY (GrB_transpose (C, NULL, NULL, C, NULL)) ;
    printgauss (C, "\n=============== C = C'\n") ;

    for (int trial = 0 ; trial <= 1 ; trial++)
    {
        GrB_Matrix Z, E ;
        int ncols = 8 ;
        int nrows = (trial == 0) ? 256 : 16 ;
        TRY (GrB_Matrix_new (&Z, Gauss, nrows, ncols)) ;
        TRY (GrB_Matrix_new (&E, Gauss, nrows-8, 4)) ;
        TRY (GxB_Matrix_Option_set (Z, GxB_FORMAT, GxB_BY_COL)) ;
        GrB_Matrix Tiles [3][2] ;
        Tiles [0][0] = C ; Tiles [0][1] = D ;
        Tiles [1][0] = E ; Tiles [1][1] = E ;
        Tiles [2][0] = D ; Tiles [2][1] = C ;
        TRY (GxB_Matrix_concat (Z, (GrB_Matrix *) Tiles, 3, 2, NULL)) ;
        printgauss (Z, "\n=============== Z = [C D ; E E ; D C]") ;
        TRY (GxB_Matrix_fprint (Z, "Z", GxB_COMPLETE, stdout)) ;

        GrB_Matrix CTiles [4] ;
        GrB_Index Tile_nrows [2] ;
        GrB_Index Tile_ncols [2] ;
        Tile_nrows [0] = nrows / 2 ;
        Tile_nrows [1] = nrows / 2 ;
        Tile_ncols [0] = 3 ;
        Tile_ncols [1] = 5 ;
        TRY (GxB_Matrix_split (CTiles, 2, 2, Tile_nrows, Tile_ncols, Z, NULL)) ;

        for (int k = 0 ; k < 4 ; k++)
        {
            printgauss (CTiles [k], "\n=============== C Tile from Z:\n") ;
            TRY (GxB_Matrix_fprint (CTiles [k], "CTiles [k]", GxB_COMPLETE,
                stdout)) ;
            GrB_Matrix_free (& (CTiles [k])) ;
        }

        GrB_Matrix_free (&Z) ;
        GrB_Matrix_free (&E) ;
    }

    // try using cmake instead of a direct compile/link command
    TRY (GxB_Global_Option_set (GxB_JIT_USE_CMAKE, true)) ;

    // C += ciso
    TRY (GrB_Matrix_assign_UDT (C, NULL, AddGauss, (void *) &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== C = C + ciso\n") ;

    // split the full matrix C
    TRY (GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, GxB_FULL)) ;
    GrB_Matrix STiles [4] ;
    GrB_Index Tile_nrows [2] = { 1, 3 } ;
    GrB_Index Tile_ncols [2] = { 2, 2 } ;
    TRY (GxB_Matrix_split (STiles, 2, 2, Tile_nrows, Tile_ncols, C, NULL)) ;

    for (int k = 0 ; k < 4 ; k++)
    {
        printgauss (STiles [k], "\n=============== S Tile from C:\n") ;
        TRY (GxB_Matrix_fprint (STiles [k], "STiles [k]", GxB_COMPLETE,
            stdout)) ;
        GrB_Matrix_free (& (STiles [k])) ;
    }

    // pause the JIT
    printf ("JIT: paused\n") ;
    TRY (GxB_Global_Option_set (GxB_JIT_C_CONTROL, GxB_JIT_PAUSE)) ;

    // C += ciso
    printgauss (C, "\n=============== C: \n") ;
    TRY (GrB_Matrix_assign_UDT (C, NULL, AddGauss, (void *) &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== C = C + ciso (JIT paused):\n") ;

    // C *= ciso
    printgauss (C, "\n=============== C: \n") ;
    TRY (GrB_Matrix_assign_UDT (C, NULL, MultGauss, (void *) &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== C = C * ciso (JIT paused):\n") ;

    // re-enable the JIT, but not to compile anything new
    printf ("JIT: run (may not load or compile)\n") ;
    TRY (GxB_Global_Option_set (GxB_JIT_C_CONTROL, GxB_JIT_RUN)) ;

    // C += ciso, using the previous loaded JIT kernel
    TRY (GrB_Matrix_assign_UDT (C, NULL, AddGauss, (void *) &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== C = C + ciso (JIT run):\n") ;

    // C *= ciso, but using generic since it is not compiled
    TRY (GrB_Matrix_assign_UDT (C, NULL, MultGauss, (void *) &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== C = C * ciso (JIT not loaded):\n") ;

    // re-enable the JIT entirely
    printf ("JIT: on\n") ;
    TRY (GxB_Global_Option_set (GxB_JIT_C_CONTROL, GxB_JIT_ON)) ;

    // C *= ciso, compiling a new JIT kernel if needed
    TRY (GrB_Matrix_assign_UDT (C, NULL, MultGauss, (void *) &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== C = C * ciso (full JIT):\n") ;

    gauss result ;
    TRY (GrB_Matrix_extractElement_UDT (&result, C, 3, 3)) ;
    if (result.real == 65 && result.imag == 1170)
    {
        fprintf (stderr, "gauss_demo: all tests pass\n") ;
    }
    else
    {
        fprintf (stderr, "gauss_demo: test failure\n") ;
    }

    // free everything and finalize GraphBLAS
    GrB_Matrix_free (&A) ;
    GrB_Matrix_free (&B) ;
    GrB_Matrix_free (&D) ;
    GrB_Matrix_free (&C) ;
    GrB_Matrix_free (&R) ;
    GrB_Type_free (&Gauss) ;
    GrB_BinaryOp_free (&AddGauss) ;
    GrB_UnaryOp_free (&RealGauss) ;
    GrB_IndexUnaryOp_free (&IJGauss) ;
    GrB_Monoid_free (&AddMonoid) ;
    GrB_BinaryOp_free (&MultGauss) ;
    GrB_Semiring_free (&GaussSemiring) ;
    GrB_finalize ( ) ;
}

