//------------------------------------------------------------------------------
// GraphBLAS/Demo/Program/gauss_demo: Gaussian integers
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "graphblas_demos.h"
#define FREE_ALL ;

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

void printgauss (GrB_Matrix A, char *name)
{
    // print the matrix
    GrB_Info info = GrB_SUCCESS ;
    GrB_Index m, n ;
    GrB_Matrix_nrows (&m, A) ;
    GrB_Matrix_ncols (&n, A) ;
    printf ("\n%s\nsize: %d-by-%d\n", name, (int) m, (int) n) ;
    for (int i = 0 ; i < m ; i++)
    {
        printf ("row %2d: ", i) ;
        for (int j = 0 ; j < n ; j++)
        {
            gauss a ;
            info = GrB_Matrix_extractElement_UDT (&a, A, i, j) ;
            if (info == GrB_NO_VALUE)
            {
                printf ("      .     ") ;
            }
            else if (info == GrB_SUCCESS)
            {
                printf (" (%4d,%4d)", a.real, a.imag) ;
            }
            else
            {
                printf (" error: %d!", info) ;
            }
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
    fprintf (stderr, "\ngauss_demo:\n") ;

    // start GraphBLAS
    GrB_Info info = GrB_SUCCESS ;
    OK (GrB_init (GrB_NONBLOCKING)) ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true, GxB_BURBLE)) ;

    // try using cmake to build all JIT kernels, just as a test.  This setting
    // is ignored by Windows (for MSVC it is treated as always true, and for
    // MINGW it is treated as always false).  Only Linux and Mac can change
    // this setting.
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true, GxB_JIT_USE_CMAKE)) ;

    printf ("Gauss demo.  Note that all transposes are array transposes,\n"
        "not matrix (conjugate) transposes.\n\n") ;

    OK (GxB_Context_fprint (GxB_CONTEXT_WORLD, "World", GxB_COMPLETE, stdout));

    printf ("JIT configuration: ------------------\n") ;
    char str [5000] ;

    OK (GrB_Global_get_String (GrB_GLOBAL, str, GxB_JIT_C_COMPILER_NAME)) ;
    printf ("JIT C compiler:   [%s]\n", str) ;

    OK (GrB_Global_get_String (GrB_GLOBAL, str, GxB_JIT_C_COMPILER_FLAGS)) ;
    printf ("JIT C flags:      [%s]\n", str) ;

    OK (GrB_Global_get_String (GrB_GLOBAL, str, GxB_JIT_C_LINKER_FLAGS)) ;
    printf ("JIT C link flags: [%s]\n", str) ;

    OK (GrB_Global_get_String (GrB_GLOBAL, str, GxB_JIT_C_LIBRARIES)) ;
    printf ("JIT C libraries:  [%s]\n", str) ;

    OK (GrB_Global_get_String (GrB_GLOBAL, str, GxB_JIT_C_PREFACE)) ;
    printf ("JIT C preface:    [%s]\n", str) ;

    OK (GrB_Global_get_String (GrB_GLOBAL, str, GxB_JIT_CACHE_PATH)) ;
    printf ("JIT cache:        [%s]\n", str) ;

    int control ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &control, GxB_JIT_C_CONTROL)) ;
    printf ("JIT C control:    [%d]\n", control) ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_ON, GxB_JIT_C_CONTROL)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &control, GxB_JIT_C_CONTROL)) ;
    int save = control ;
    printf ("JIT C control:    [%d] reset\n", control) ;
    printf ("-------------------------------------\n\n") ;

    // revise the header for each JIT kernel; this is not required but appears
    // here just as a demo of the feature.
    OK (GrB_Global_set_String (GrB_GLOBAL,
        "// kernel generated by gauss_demo.c\n"
        "#include <math.h>\n",  GxB_JIT_C_PREFACE)) ;
    OK (GrB_Global_get_String (GrB_GLOBAL, str, GxB_JIT_C_PREFACE)) ;
    printf ("JIT C preface (revised):\n%s\n", str) ;

    // create the Gauss type but do it wrong the first time.  This will always
    // require a new JIT kernel to be compiled: if this is the first run of
    // this demo, the cache folder is empty.  Otherwise, the good gauss type
    // will be left in the cache folder from a prior run of this program, and
    // its type definition does not match this one.  The burble will say "jit:
    // loaded but must recompile" in this case.  This is skipped if the JIT
    // is disabled, since trying the BadGauss type will disable the good
    // PreJIT kernel.
    size_t sizeof_gauss ;
    if (control == GxB_JIT_ON)
    {
        GrB_Type BadGauss = NULL ;
        info = GxB_Type_new (&BadGauss, 0, "gauss", BAD_GAUSS_DEFN) ;
        if (info != GrB_SUCCESS)
        {
            // JIT disabled
            printf ("JIT: unable to determine type size: set it to %d\n",
                (int) sizeof (badgauss)) ;
            OK (GrB_Type_new (&BadGauss, sizeof (badgauss))) ;
        }
        OK (GxB_Type_fprint (BadGauss, "BadGauss", GxB_COMPLETE, stdout)) ;

        OK (GrB_Type_get_SIZE (BadGauss, &sizeof_gauss, GrB_SIZE)) ;
        CHECK (sizeof_gauss == sizeof (badgauss), GrB_PANIC) ;
        GrB_Type_free (&BadGauss) ;
    }

    // the JIT should have been successful, unless it was originally off
    #define OK_JIT \
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &control, GxB_JIT_C_CONTROL)) ; \
    CHECK (control == save, GrB_PANIC) ;
    OK_JIT

    // renable the JIT in case the JIT was disabled when GraphBLAS was built;
    // this will enable any prejit kernels.
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_ON, GxB_JIT_C_CONTROL)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &save, GxB_JIT_C_CONTROL)) ;
    printf ("jit: status %d\n", save) ;

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
            OK (GrB_Type_new (&Gauss, sizeof (gauss))) ;
        }
        OK (GxB_Type_fprint (Gauss, "Gauss", GxB_COMPLETE, stdout)) ;
        OK (GrB_Type_get_SIZE (Gauss, &sizeof_gauss, GrB_SIZE)) ;
        CHECK (sizeof_gauss == sizeof (gauss), GrB_PANIC) ;
        OK_JIT
    }

    printf ("JIT: off\n") ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_OFF, GxB_JIT_C_CONTROL)) ;
    printf ("JIT: on\n") ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_ON, GxB_JIT_C_CONTROL)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &control, GxB_JIT_C_CONTROL)) ;
    printf ("jit: status %d\n", control) ;

    // create the BadAddGauss operator; use a NULL function pointer to test the
    // JIT.  Like the BadGauss type, this will always require a JIT
    // compilation, because the type will not match the good 'addgauss'
    // definition from a prior run of this demo.  Skip this if the JIT is
    // disabled, to allow PreJIT kernels to be used instead.  Creating
    // the invalid addgauss operator will disable the good PreJIT addgauss.
    GrB_BinaryOp BadAddGauss = NULL ; 
    if (control == GxB_JIT_ON)
    {
        info = GxB_BinaryOp_new (&BadAddGauss, NULL,
            Gauss, Gauss, Gauss, "addgauss", BAD_ADDGAUSS_DEFN) ;
        if (info != GrB_SUCCESS)
        {
            // JIT disabled
            printf ("JIT: unable to compile the BadAddGauss kernel\n") ;
            OK (GrB_BinaryOp_new (&BadAddGauss, (GxB_binary_function) badaddgauss,
                Gauss, Gauss, Gauss)) ;
        }
        OK (GxB_BinaryOp_fprint (BadAddGauss, "BadAddGauss", GxB_COMPLETE,
            stdout)) ;
        GrB_BinaryOp_free (&BadAddGauss) ;
    }

    OK_JIT

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
            OK (GrB_BinaryOp_new (&AddGauss, (GxB_binary_function) addgauss,
                Gauss, Gauss, Gauss)) ;
        }
        OK (GxB_BinaryOp_fprint (AddGauss, "AddGauss", GxB_COMPLETE, stdout)) ;
        OK_JIT
    }

    printf ("JIT: off\n") ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_OFF, GxB_JIT_C_CONTROL)) ;
    printf ("JIT: on\n") ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_ON, GxB_JIT_C_CONTROL)) ;

    // renable the JIT in case the JIT was disabled when GraphBLAS was built;
    // this will enable any prejit kernels.
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_ON, GxB_JIT_C_CONTROL)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &save, GxB_JIT_C_CONTROL)) ;
    printf ("jit: status %d\n", save) ;

    // create the AddMonoid
    gauss zero ;
    zero.real = 0 ;
    zero.imag = 0 ;
    GrB_Monoid AddMonoid ;
    OK (GrB_Monoid_new_UDT (&AddMonoid, AddGauss, &zero)) ;
    OK (GxB_Monoid_fprint (AddMonoid, "AddMonoid", GxB_COMPLETE, stdout)) ;

    // create the MultGauss operator
    GrB_BinaryOp MultGauss ;
    OK (GxB_BinaryOp_new (&MultGauss, (GxB_binary_function) multgauss,
        Gauss, Gauss, Gauss, "multgauss", MULTGAUSS_DEFN)) ;
    OK (GxB_BinaryOp_fprint (MultGauss, "MultGauss", GxB_COMPLETE, stdout)) ;

    // create the GaussSemiring
    GrB_Semiring GaussSemiring ;
    OK (GrB_Semiring_new (&GaussSemiring, AddMonoid, MultGauss)) ;
    OK (GxB_Semiring_fprint (GaussSemiring, "GaussSemiring", GxB_COMPLETE,
        stdout)) ;

    // create a 4-by-4 Gauss matrix, each entry A(i,j) = (i+1,2-j),
    // except A(0,0) is missing
    GrB_Matrix A, D ;
    OK (GrB_Matrix_new (&A, Gauss, 4, 4)) ;
    OK (GrB_Matrix_new (&D, GrB_BOOL, 4, 4)) ;
    gauss a ;
    for (int i = 0 ; i < 4 ; i++)
    {
        OK (GrB_Matrix_setElement_BOOL (D, 1, i, i)) ;
        for (int j = 0 ; j < 4 ; j++)
        {
            if (i == 0 && j == 0) continue ;
            a.real = i+1 ;
            a.imag = 2-j ;
            OK (GrB_Matrix_setElement_UDT (A, &a, i, j)) ;
        }
    }
    printgauss (A, "\n=============== Gauss A matrix:\n") ;

    // a = sum (A)
    OK (GrB_Matrix_reduce_UDT (&a, NULL, AddMonoid, A, NULL)) ;
    printf ("\nsum (A) = (%d,%d)\n", a.real, a.imag) ;
    OK_JIT

    // A = A*A
    OK (GrB_mxm (A, NULL, NULL, GaussSemiring, A, A, NULL)) ;
    printgauss (A, "\n=============== Gauss A = A^2 matrix:\n") ;
    OK_JIT

    // a = sum (A)
    OK (GrB_Matrix_reduce_UDT (&a, NULL, AddMonoid, A, NULL)) ;
    printf ("\nsum (A^2) = (%d,%d)\n", a.real, a.imag) ;
    OK_JIT

    // C<D> = A*A' where A and D are sparse
    GrB_Matrix C ;
    OK (GrB_Matrix_new (&C, Gauss, 4, 4)) ;
    printgauss (C, "\nGauss C empty matrix") ;

    OK (GrB_Matrix_set_INT32 (A, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_set_INT32 (D, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_mxm (C, D, NULL, GaussSemiring, A, A, GrB_DESC_T1)) ;
    printgauss (C, "\n=============== Gauss C = diag(AA') matrix:\n") ;
    OK_JIT

    // C = D*A
    GrB_Matrix_free (&D) ;
    OK (GrB_Matrix_new (&D, Gauss, 4, 4)) ;
    OK (GrB_Matrix_set_INT32 (A, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_set_INT32 (D, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_select_INT64 (D, NULL, NULL, GrB_DIAG, A, 0, NULL)) ;
    printgauss (D, "\nGauss D matrix") ;
    OK (GrB_mxm (C, NULL, NULL, GaussSemiring, D, A, NULL)) ;
    printgauss (C, "\n=============== Gauss C = D*A matrix:\n") ;
    OK_JIT

    // convert D to bitmap then back to sparse
    OK (GrB_Matrix_set_INT32 (D, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    OK (GrB_Matrix_set_INT32 (D, GxB_BITMAP, GxB_SPARSITY_CONTROL)) ;

    printgauss (D, "\nGauss D matrix (bitmap)") ;
    OK (GxB_Matrix_fprint (D, "D", GxB_COMPLETE, stdout)) ;
    OK (GrB_Matrix_set_INT32 (D, GxB_SPARSE, GxB_SPARSITY_CONTROL)) ;
    printgauss (D, "\nGauss D matrix (back to sparse)") ;
    OK (GxB_Matrix_fprint (D, "D", GxB_COMPLETE, stdout)) ;
    OK_JIT

    // C = A*D
    OK (GrB_mxm (C, NULL, NULL, GaussSemiring, A, D, NULL)) ;
    printgauss (C, "\n=============== Gauss C = A*D matrix:\n") ;
    OK_JIT

    // C = (1,2) then C += A*A' where C is full
    gauss ciso ;
    ciso.real = 1 ;
    ciso.imag = -2 ;
    OK (GrB_Matrix_assign_UDT (C, NULL, NULL, &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== Gauss C = (1,-2) matrix:\n") ;
    printgauss (A, "\n=============== Gauss A matrix:\n") ;
    OK (GrB_mxm (C, NULL, AddGauss, GaussSemiring, A, A, GrB_DESC_T1)) ;
    printgauss (C, "\n=============== Gauss C += A*A' matrix:\n") ;
    OK_JIT

    // C += B*A where B is full and A is sparse
    GrB_Matrix B ;
    OK (GrB_Matrix_new (&B, Gauss, 4, 4)) ;
    OK (GrB_Matrix_assign_UDT (B, NULL, NULL, &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (B, "\n=============== Gauss B = (1,-2) matrix:\n") ;
    OK (GrB_mxm (C, NULL, AddGauss, GaussSemiring, B, A, NULL)) ;
    printgauss (C, "\n=============== Gauss C += B*A:\n") ;
    OK_JIT

    // C += A*B where B is full and A is sparse
    OK (GrB_mxm (C, NULL, AddGauss, GaussSemiring, A, B, NULL)) ;
    printgauss (C, "\n=============== Gauss C += A*B:\n") ;
    OK_JIT

    // C = ciso+A
    OK (GrB_Matrix_apply_BinaryOp1st_UDT (C, NULL, NULL, AddGauss,
        (void *) &ciso, A, NULL)) ;
    printgauss (C, "\n=============== Gauss C = (1,-2) + A:\n") ;
    OK_JIT

    // C = A*ciso
    OK (GrB_Matrix_apply_BinaryOp2nd_UDT (C, NULL, NULL, MultGauss, A,
        (void *) &ciso, NULL)) ;
    printgauss (C, "\n=============== Gauss C = A*(1,-2):\n") ;
    OK_JIT

    // C = A'*ciso
    OK (GrB_Matrix_apply_BinaryOp2nd_UDT (C, NULL, NULL, MultGauss, A,
        (void *) &ciso, GrB_DESC_T0)) ;
    printgauss (C, "\n=============== Gauss C = A'*(1,-2):\n") ;
    OK_JIT

    // C = ciso*A'
    OK (GrB_Matrix_apply_BinaryOp1st_UDT (C, NULL, NULL, MultGauss,
        (void *) &ciso, A, GrB_DESC_T1)) ;
    printgauss (C, "\n=============== Gauss C = (1,-2)*A':\n") ;
    OK_JIT

    // create the RealGauss unary op
    GrB_UnaryOp RealGauss ;
    OK (GxB_UnaryOp_new (&RealGauss, (GxB_unary_function) realgauss,
        GrB_INT32, Gauss, "realgauss", REALGAUSS_DEFN)) ;
    OK (GxB_UnaryOp_fprint (RealGauss, "RealGauss", GxB_COMPLETE, stdout)) ;
    GrB_Matrix R ;
    OK (GrB_Matrix_new (&R, GrB_INT32, 4, 4)) ;
    OK_JIT

    // R = RealGauss (C)
    OK (GrB_Matrix_apply (R, NULL, NULL, RealGauss, C, NULL)) ;
    OK (GxB_Matrix_fprint (R, "R", GxB_COMPLETE, stdout)) ;
    OK_JIT

    // R = RealGauss (C')
    printgauss (C, "\n=============== R = RealGauss (C')\n") ;
    OK (GrB_Matrix_apply (R, NULL, NULL, RealGauss, C, GrB_DESC_T0)) ;
    OK (GxB_Matrix_fprint (R, "R", GxB_COMPLETE, stdout)) ;
    GrB_Matrix_free (&R) ;
    OK_JIT

    // create the IJGauss IndexUnaryOp
    GrB_IndexUnaryOp IJGauss ;
    OK (GxB_IndexUnaryOp_new (&IJGauss, (GxB_index_unary_function) ijgauss,
        GrB_INT64, Gauss, Gauss, "ijgauss", IJGAUSS_DEFN)) ;
    OK (GrB_Matrix_new (&R, GrB_INT64, 4, 4)) ;
    printgauss (C, "\n=============== C \n") ;
    OK (GrB_Matrix_apply_IndexOp_UDT (R, NULL, NULL, IJGauss, C,
        (void *) &ciso, NULL)) ;
    printf ("\nR = ijgauss (C)\n") ;
    OK (GxB_Matrix_fprint (R, "R", GxB_COMPLETE, stdout)) ;
    GrB_Index I [100], J [100], rnvals = 100 ;
    double X [100] ;
    OK (GrB_Matrix_extractTuples_FP64 (I, J, X, &rnvals, R)) ;
    for (int k = 0 ; k < rnvals ; k++)
    { 
        printf ("R (%d,%d) = %g\n", (int) I [k], (int) J [k], X [k]) ;
    }
    OK_JIT

    // C = C'
    printgauss (C, "\n=============== C\n") ;
    OK (GrB_transpose (C, NULL, NULL, C, NULL)) ;
    printgauss (C, "\n=============== C = C'\n") ;
    OK_JIT

    for (int trial = 0 ; trial <= 1 ; trial++)
    {
        GrB_Matrix Z, E ;
        int ncols = 8 ;
        int nrows = (trial == 0) ? 256 : 16 ;
        OK (GrB_Matrix_new (&Z, Gauss, nrows, ncols)) ;
        OK (GrB_Matrix_new (&E, Gauss, nrows-8, 4)) ;
        OK (GrB_Matrix_set_INT32 (Z, GrB_COLMAJOR,
            GrB_STORAGE_ORIENTATION_HINT)) ;
        GrB_Matrix Tiles [3][2] ;
        Tiles [0][0] = C ; Tiles [0][1] = D ;
        Tiles [1][0] = E ; Tiles [1][1] = E ;
        Tiles [2][0] = D ; Tiles [2][1] = C ;
        OK (GxB_Matrix_concat (Z, (GrB_Matrix *) Tiles, 3, 2, NULL)) ;
        printgauss (Z, "\n=============== Z = [C D ; E E ; D C]") ;
        OK (GxB_Matrix_fprint (Z, "Z", GxB_COMPLETE, stdout)) ;
        OK_JIT

        GrB_Matrix CTiles [4] ;
        GrB_Index Tile_nrows [2] ;
        GrB_Index Tile_ncols [2] ;
        Tile_nrows [0] = nrows / 2 ;
        Tile_nrows [1] = nrows / 2 ;
        Tile_ncols [0] = 3 ;
        Tile_ncols [1] = 5 ;
        OK (GxB_Matrix_split (CTiles, 2, 2, Tile_nrows, Tile_ncols, Z, NULL)) ;
        OK_JIT

        for (int k = 0 ; k < 4 ; k++)
        {
            printgauss (CTiles [k], "\n=============== C Tile from Z:\n") ;
            OK (GxB_Matrix_fprint (CTiles [k], "CTiles [k]", GxB_COMPLETE,
                stdout)) ;
            GrB_Matrix_free (& (CTiles [k])) ;
            OK_JIT
        }

        GrB_Matrix_free (&Z) ;
        GrB_Matrix_free (&E) ;
    }

    // try using cmake instead of a direct compile/link command
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true, GxB_JIT_USE_CMAKE)) ;
    OK_JIT

    // C += ciso
    OK (GrB_Matrix_assign_UDT (C, NULL, AddGauss, (void *) &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== C = C + ciso\n") ;
    OK_JIT

    // split the full matrix C
    OK (GrB_Matrix_set_INT32 (C, GxB_FULL, GxB_SPARSITY_CONTROL)) ;
    GrB_Matrix STiles [4] ;
    GrB_Index Tile_nrows [2] = { 1, 3 } ;
    GrB_Index Tile_ncols [2] = { 2, 2 } ;
    OK (GxB_Matrix_split (STiles, 2, 2, Tile_nrows, Tile_ncols, C, NULL)) ;
    OK_JIT

    for (int k = 0 ; k < 4 ; k++)
    {
        printgauss (STiles [k], "\n=============== S Tile from C:\n") ;
        OK (GxB_Matrix_fprint (STiles [k], "STiles [k]", GxB_COMPLETE,
            stdout)) ;
        GrB_Matrix_free (& (STiles [k])) ;
        OK_JIT
    }

    // pause the JIT
    printf ("JIT: paused\n") ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_PAUSE, GxB_JIT_C_CONTROL)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &control, GxB_JIT_C_CONTROL)) ;
    save = control ;
    OK_JIT

    // C += ciso
    printgauss (C, "\n=============== C: \n") ;
    OK (GrB_Matrix_assign_UDT (C, NULL, AddGauss, (void *) &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== C = C + ciso (JIT paused):\n") ;
    OK_JIT

    // C *= ciso
    printgauss (C, "\n=============== C: \n") ;
    OK (GrB_Matrix_assign_UDT (C, NULL, MultGauss, (void *) &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== C = C * ciso (JIT paused):\n") ;
    OK_JIT

    // re-enable the JIT, but not to compile anything new
    printf ("JIT: run (the JIT can only run, not load or compile)\n") ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_RUN, GxB_JIT_C_CONTROL)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &control, GxB_JIT_C_CONTROL)) ;
    save = control ;
    OK_JIT

    // C += ciso, using the previous loaded JIT kernel
    OK (GrB_Matrix_assign_UDT (C, NULL, AddGauss, (void *) &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== C = C + ciso (JIT run):\n") ;
    OK_JIT

    // C *= ciso, but using generic since it is not compiled
    OK (GrB_Matrix_assign_UDT (C, NULL, MultGauss, (void *) &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== C = C * ciso (JIT not loaded):\n") ;
    OK_JIT

    // re-enable the JIT entirely
    printf ("JIT: on\n") ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, GxB_JIT_ON, GxB_JIT_C_CONTROL)) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &control, GxB_JIT_C_CONTROL)) ;
    save = control ;
    OK_JIT

    // C *= ciso, compiling a new JIT kernel if needed
    OK (GrB_Matrix_assign_UDT (C, NULL, MultGauss, (void *) &ciso,
        GrB_ALL, 4, GrB_ALL, 4, NULL)) ;
    printgauss (C, "\n=============== C = C * ciso (full JIT):\n") ;
    OK_JIT

    gauss result ;
    OK (GrB_Matrix_extractElement_UDT (&result, C, 3, 3)) ;

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
    OK_JIT
    GrB_finalize ( ) ;

    // return result
    bool ok = (result.real == 65 && result.imag == 1170) ;
    if (ok)
    {
        fprintf (stderr, "gauss_demo: all tests pass\n") ;
    }
    else
    {
        fprintf (stderr, "gauss_demo: test failure\n") ;
    }
    return (ok ? 0 : 1) ;
}

