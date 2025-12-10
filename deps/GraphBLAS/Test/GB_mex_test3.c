//------------------------------------------------------------------------------
// GB_mex_test3: still more basic tests
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Test lots of random stuff.  The function otherwise serves no purpose.

#include "GB_mex.h"
#include "GB_mex_errors.h"
#include "../Source/assign/GB_bitmap_assign_methods.h"

#define USAGE "GB_mex_test3"

int myprintf (const char *restrict format, ...) ;

int myprintf (const char *restrict format, ...)
{
    printf ("[[myprintf:") ;
    va_list ap ;
    va_start (ap, format) ;
    vprintf (format, ap) ;
    va_end (ap) ;
    printf ("]]") ;
    return (1) ;
}

int myflush (void) ;

int myflush (void)
{
    printf ("myflush\n") ;
    fflush (stdout) ;
    return (0) ;
}

typedef int (* printf_func_t) (const char *restrict format, ...) ;
typedef int (* flush_func_t)  (void) ;

 typedef struct { int64_t stuff [4] ; } gb_my4x64 ;
#define MY4X64 \
"typedef struct { int64_t stuff [4] ; } gb_my4x64 ;"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    GrB_Matrix C = NULL, A = NULL, M = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_Vector w = NULL ;
    GrB_Type myint = NULL, My4x64 = NULL ;
    GB_void *Null = NULL ;
    char *err ;

    //--------------------------------------------------------------------------
    // startup GraphBLAS
    //--------------------------------------------------------------------------

    bool malloc_debug = GB_mx_get_global (true) ;
    FILE *f = fopen ("errlog4.txt", "w") ;
    int expected = GrB_SUCCESS ;

    //--------------------------------------------------------------------------
    // GxB_set/get for printf and flush
    //--------------------------------------------------------------------------

    bool save_burble = false ;
    OK (GxB_Global_Option_get (GxB_BURBLE, &save_burble)) ;
    OK (GxB_Global_Option_set (GxB_BURBLE, true)) ;
    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;

    printf ("\nBurble with standard printf/flush:\n") ;
    uint64_t nvals ;
    OK (GrB_Matrix_nvals (&nvals, C)) ;
    CHECK (nvals == 0) ;

    OK (GxB_Global_Option_set (GxB_PRINTF, myprintf)) ;
    OK (GxB_Global_Option_set (GxB_FLUSH, myflush)) ;

    printf_func_t mypr, mypr2 ;
    OK (GxB_Global_Option_get (GxB_PRINTF, &mypr)) ;
    OK (GxB_Global_Option_get_FUNCTION (GxB_PRINTF, (void **) &mypr2)) ;
    CHECK (mypr == myprintf) ;
    CHECK (mypr2 == myprintf) ;

    flush_func_t myfl, myfl2 ;
    OK (GxB_Global_Option_get (GxB_FLUSH, &myfl)) ;
    OK (GxB_Global_Option_get_FUNCTION (GxB_FLUSH, (void **) &myfl2)) ;
    CHECK (myfl == myflush) ;
    CHECK (myfl2 == myflush) ;

    printf ("\nBurble with myprintf/myflush:\n") ;
    OK (GrB_Matrix_nvals (&nvals, C)) ;
    CHECK (nvals == 0) ;
    OK (GxB_Global_Option_set (GxB_BURBLE, save_burble)) ;

    OK (GxB_Global_Option_set (GxB_PRINTF, printf)) ;
    OK (GxB_Global_Option_set (GxB_FLUSH, NULL)) ;

    OK (GxB_Global_Option_set_FUNCTION (GxB_PRINTF, myprintf)) ;
    OK (GxB_Global_Option_set_FUNCTION (GxB_FLUSH, myflush)) ;

    OK (GxB_Global_Option_get_FUNCTION (GxB_PRINTF, (void **) &mypr2)) ;
    OK (GxB_Global_Option_get_FUNCTION (GxB_FLUSH, (void **) &myfl2)) ;
    CHECK (mypr2 == myprintf) ;
    CHECK (myfl2 == myflush) ;

    OK (GxB_Global_Option_set (GxB_PRINTF, printf)) ;
    OK (GxB_Global_Option_set (GxB_FLUSH, NULL)) ;

    //--------------------------------------------------------------------------
    // test GxB_set/get for options no longer used
    //--------------------------------------------------------------------------

    int64_t limit [64], limit2 [64] ;
    int64_t defaults [64] ;
    for (int k = 0 ; k < 64 ; k++)
    {
        limit [k] = k+1 ;
        limit2 [k] = 3*k + 4 ;
    }

    OK (GxB_Global_Option_set (GxB_MEMORY_POOL, NULL)) ;
    OK (GxB_Global_Option_get (GxB_MEMORY_POOL, defaults)) ;
    OK (GxB_Global_Option_get_INT64 (GxB_MEMORY_POOL, limit2)) ;
    printf ("\ndefault memory pool limits:\n") ;
    for (int k = 0 ; k < 64 ; k++)
    {
        CHECK (defaults [k] == 0) ;
    }

    for (int k = 0 ; k < 64 ; k++)
    {
        limit [k] = k ;
    }
    OK (GxB_Global_Option_set (GxB_MEMORY_POOL, limit)) ;
    OK (GxB_Global_Option_get (GxB_MEMORY_POOL, limit)) ;
    OK (GxB_Global_Option_get_INT64 (GxB_MEMORY_POOL, limit2)) ;
    for (int k = 0 ; k < 64 ; k++)
    {
        CHECK (limit [k] == 0) ;
        CHECK (limit2 [k] == 0) ;
    }
    for (int k = 0 ; k < 64 ; k++)
    {
        limit [k] = 0 ;
        limit2 [k] = 0 ;
    }

    OK (GxB_Global_Option_set (GxB_MEMORY_POOL, limit)) ;
    OK (GxB_Global_Option_get (GxB_MEMORY_POOL, limit)) ;
    OK (GxB_Global_Option_get_INT64 (GxB_MEMORY_POOL, limit2)) ;
    for (int k = 0 ; k < 64 ; k++)
    {
        CHECK (limit [k] == 0) ;
        CHECK (limit2 [k] == 0) ;
    }

    OK (GxB_Global_Option_set_INT64_ARRAY (GxB_MEMORY_POOL, NULL)) ;
    OK (GxB_Global_Option_get_INT64 (GxB_MEMORY_POOL, limit2)) ;
    for (int k = 0 ; k < 64 ; k++)
    {
        CHECK (defaults [k] == 0) ;
    }

    //--------------------------------------------------------------------------
    // GrB_reduce with invalid binary op
    //--------------------------------------------------------------------------

    OK (GrB_Vector_new (&w, GrB_FP32, 10)) ;
    info = GrB_Matrix_reduce_BinaryOp (w, NULL, NULL, GrB_LT_FP32, C, NULL) ;
    CHECK (info == GrB_DOMAIN_MISMATCH) ;
    const char *s ;
    OK (GrB_error (&s, w)) ;
    printf ("expected error: [%s]\n", s) ;

    info = GrB_Matrix_reduce_BinaryOp (w, NULL, NULL, GrB_DIV_FP32, C, NULL) ;
    CHECK (info == GrB_NOT_IMPLEMENTED) ;
    OK (GrB_error (&s, w)) ;
    printf ("expected error: [%s]\n", s) ;
    GrB_Vector_free_(&w) ;

    //--------------------------------------------------------------------------
    // GB_nnz_held, GB_is_shallow
    //--------------------------------------------------------------------------

    CHECK (GB_nnz_held (NULL) == 0) ;
    CHECK (!GB_is_shallow (NULL)) ;

    //--------------------------------------------------------------------------
    // invalid iso matrix
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_fprint (C, "C ok", GxB_COMPLETE, NULL)) ;
    void *save = C->x ;
    C->x = NULL ;
    C->iso = true ;
    expected = GrB_INVALID_OBJECT ;
    ERR (GxB_Matrix_fprint (C, "C iso invald", GxB_COMPLETE, NULL)) ;
    C->x = save ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // empty scalar for iso build
    //--------------------------------------------------------------------------

    uint64_t I [4] = { 1, 2, 3, 4 } ;       // OK
    GrB_Scalar scalar = NULL ;
    OK (GrB_Scalar_new (&scalar, GrB_FP32)) ;
    OK (GxB_Scalar_fprint (scalar, "scalar init", GxB_COMPLETE, NULL)) ;
    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Vector_new (&w, GrB_FP32, 10)) ;
    expected = GrB_EMPTY_OBJECT ;
    ERR (GxB_Matrix_build_Scalar_(C, I, I, scalar, 4)) ;
    OK (GrB_error (&s, C)) ;
    printf ("expected error: [%s]\n", s) ;

    ERR (GxB_Vector_build_Scalar_(w, I, scalar, 4)) ;
    OK (GrB_error (&s, w)) ;
    printf ("expected error: [%s]\n", s) ;

    GrB_Vector I_vector = NULL, J_vector = NULL, X_vector = NULL ;
    OK (GrB_Vector_new (&I_vector, GrB_UINT32, 4)) ;
    OK (GrB_Vector_new (&J_vector, GrB_UINT32, 5)) ;
    OK (GrB_Vector_new (&X_vector, GrB_UINT32, 6)) ;
    for (int k = 0 ; k < 4 ; k++)
    {
        OK (GrB_Vector_setElement_UINT32 (I_vector, I [k], k)) ;
    }
    for (int k = 0 ; k < 5 ; k++)
    {
        OK (GrB_Vector_setElement_UINT32 (J_vector, k, k)) ;
    }
    for (int k = 0 ; k < 6 ; k++)
    {
        OK (GrB_Vector_setElement_UINT32 (X_vector, 2*k+1, k)) ;
    }

    ERR (GxB_Matrix_build_Scalar_Vector_(C, I_vector, I_vector, scalar, NULL)) ;
    OK (GrB_error (&s, C)) ;
    printf ("expected error: [%s]\n", s) ;

    ERR (GxB_Matrix_build_Scalar_Vector_(C, I_vector, I_vector, scalar, NULL)) ;
    OK (GrB_error (&s, C)) ;
    printf ("expected error: [%s]\n", s) ;

    ERR (GxB_Vector_build_Scalar_Vector_(w, I_vector, scalar, NULL)) ;
    OK (GrB_error (&s, w)) ;
    printf ("expected error: [%s]\n", s) ;

    OK (GrB_Scalar_setElement_FP32 (scalar, 3.0)) ;

    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Matrix_build_Scalar_Vector_(C, I_vector, J_vector, scalar, NULL)) ;
    OK (GrB_error (&s, C)) ;
    printf ("expected error: [%s]\n", s) ;

    ERR (GxB_Matrix_build_Vector_(C, I_vector, J_vector, X_vector,
        GrB_PLUS_UINT32, NULL)) ;
    OK (GrB_error (&s, C)) ;
    printf ("expected error: [%s]\n", s) ;

    ERR (GxB_Vector_build_Vector_(w, I_vector, X_vector,
        GrB_PLUS_UINT32, NULL)) ;
    OK (GrB_error (&s, w)) ;
    printf ("expected error: [%s]\n", s) ;

    GrB_Vector_free_(&I_vector) ;
    GrB_Vector_free_(&J_vector) ;
    GrB_Vector_free_(&X_vector) ;
    GrB_Vector_free_(&w) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // build error handling
    //--------------------------------------------------------------------------

    OK (GrB_Type_new (&myint, sizeof (int))) ;
    OK (GrB_Matrix_new (&C, myint, 10, 10)) ;
    OK (GrB_Scalar_setElement_FP32 (scalar, 3.0)) ;
    OK (GxB_Scalar_fprint (scalar, "scalar set", GxB_COMPLETE, NULL)) ;
    OK (GxB_Matrix_Option_set ((GrB_Matrix) scalar, GxB_SPARSITY_CONTROL,
        GxB_SPARSE)) ;
    scalar->jumbled = true ;
    OK (GrB_Scalar_wait (scalar, GrB_MATERIALIZE)) ;

    OK (GxB_Scalar_fprint (scalar, "scalar", GxB_COMPLETE, NULL)) ;

    expected = GrB_DOMAIN_MISMATCH ;
    ERR (GrB_Matrix_build_UINT64 (C, I, I, I, 4, GrB_PLUS_UINT64)) ;
    OK (GrB_error (&s, C)) ;
    printf ("expected error: [%s]\n", s) ;

    ERR (GxB_Matrix_build_Scalar_(C, I, I, scalar, 4)) ;
    OK (GrB_error (&s, C)) ;
    printf ("expected error: [%s]\n", s) ;
    GrB_Matrix_free_(&C) ;
    GrB_Scalar_free_(&scalar) ;
    GrB_Type_free_(&myint) ;

    //--------------------------------------------------------------------------
    // import/export
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    for (int k = 0 ; k < 8 ; k++)
    {
        OK (GrB_Matrix_setElement_FP32 (C, 1, k, k+1)) ;
    }

    GrB_Type type ;
    uint64_t nrows, ncols, Ap_size, Ai_size, Ax_size, Ah_size, nvec ;
    uint64_t *Ap = NULL, *Ai = NULL, *Ah = NULL ;   // OK
    float *Ax = NULL ;
    bool iso, jumbled ;
    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    OK (GxB_Matrix_fprint (C, "C to export", GxB_COMPLETE, NULL)) ;

    // export as CSC
    OK (GxB_Matrix_export_CSC (&C, &type, &nrows, &ncols, &Ap, &Ai,
        (void **) &Ax, &Ap_size, &Ai_size, &Ax_size, &iso, &jumbled, NULL)) ;

    // import as CSC
    expected = GrB_INVALID_VALUE ;
    ERR (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai,
        (void **) &Ax, 0, Ai_size, Ax_size, iso, jumbled, NULL)) ;
    ERR (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai,
        (void **) &Ax, Ap_size, 0, Ax_size, iso, jumbled, NULL)) ;
    ERR (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai,
        (void **) &Ax, Ap_size, Ai_size, 0, iso, jumbled, NULL)) ;
    ERR (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai,
        (void **) &Null, Ap_size, Ai_size, 0, true, jumbled, NULL)) ;

    OK (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai,
        (void **) &Ax, Ap_size, Ai_size, Ax_size, iso, jumbled, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C imported sparse", GxB_COMPLETE, NULL)) ;

    // export as HyperCSC
    OK (GxB_Matrix_export_HyperCSC (&C, &type, &nrows, &ncols,
        &Ap, &Ah, &Ai, (void **) &Ax,
        &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso, &nvec, &jumbled, NULL)) ;

    // import as HyperCSC
    ERR (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols,
        &Ap, &Ah, &Ai, (void **) &Ax,
        0, Ah_size, Ai_size, Ax_size, iso, nvec, jumbled, NULL)) ;
    ERR (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols,
        &Ap, &Ah, &Ai, (void **) &Ax,
        Ap_size, 0, Ai_size, Ax_size, iso, nvec, jumbled, NULL)) ;
    ERR (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols,
        &Ap, &Ah, &Ai, (void **) &Ax,
        Ap_size, Ah_size, 0, Ax_size, iso, nvec, jumbled, NULL)) ;
    ERR (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols,
        &Ap, &Ah, &Ai, (void **) &Ax,
        Ap_size, Ah_size, Ai_size, 0, iso, nvec, jumbled, NULL)) ;
    OK (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols,
        &Ap, &Ah, &Ai, (void **) &Ax,
        Ap_size, Ah_size, Ai_size, Ax_size, iso, nvec, jumbled, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C imported hyper", GxB_SHORT, NULL)) ;
    GrB_Matrix_free_(&C) ;

    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Matrix_assign_FP32 (C, NULL, NULL, 1, GrB_ALL, 10, GrB_ALL, 10,
        NULL)) ;

    // export as CSC, non-iso
    OK (GxB_Matrix_export_CSC (&C, &type, &nrows, &ncols, &Ap, &Ai,
        (void **) &Ax, &Ap_size, &Ai_size, &Ax_size, NULL, &jumbled, NULL)) ;

    ERR (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai,
        (void **) &Ax, Ap_size, Ai_size, 0, false, jumbled, NULL)) ;

    OK (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai,
        (void **) &Ax, Ap_size, Ai_size, Ax_size, false, jumbled, NULL)) ;

    OK (GxB_Matrix_fprint (C, "C imported non-iso", GxB_SHORT, NULL)) ;
    OK (GrB_Matrix_free_(&C)) ;

    // export as CSC iso
    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Matrix_assign_FP32 (C, NULL, NULL, 1, GrB_ALL, 10, GrB_ALL, 10,
        NULL)) ;
    OK (GxB_Matrix_export_CSC (&C, &type, &nrows, &ncols, &Ap, &Ai,
        (void **) &Ax, &Ap_size, &Ai_size, &Ax_size, &iso, &jumbled, NULL)) ;

    // import as CSC iso
    OK (GxB_Matrix_import_CSC (&C, type, nrows, ncols, &Ap, &Ai,
        (void **) &Ax, Ap_size, Ai_size, Ax_size, iso, jumbled, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C imported iso", GxB_SHORT, NULL)) ;
    OK (GrB_Matrix_free_(&C)) ;

    //--------------------------------------------------------------------------
    // split a user-defined matrix
    //--------------------------------------------------------------------------

    CHECK (sizeof (GB_blob16) == 2 * sizeof (uint64_t)) ;

    OK (GxB_Type_new (&My4x64, sizeof (gb_my4x64), "gb_my4x64", MY4X64)) ;
    OK (GxB_Type_fprint (My4x64, "My4x64", GxB_COMPLETE, NULL)) ;
    gb_my4x64 my4x64_scalar ;
    OK (GrB_Matrix_new (&C, My4x64, 4, 4)) ;

    GrB_Matrix Tiles [4]  = { NULL, NULL, NULL, NULL} ;
    uint64_t Tile_nrows [2] = { 2, 2 } ;
    uint64_t Tile_ncols [2] = { 2, 2 } ;

    for (int sparsity_control = 1 ;
             sparsity_control <= 8 ;
             sparsity_control *= 2)
    {
        
        printf ("\n################# sparsity_control %d\n", sparsity_control) ;
        OK (GrB_Matrix_clear (C)) ;

        for (int i = 0 ; i < 4 ; i++)
        {
            for (int j = 0 ; j < 4 ; j++)
            {
                if (sparsity_control < 8 && i == j) continue ;
                my4x64_scalar.stuff [0] = i ;
                my4x64_scalar.stuff [1] = j ;
                my4x64_scalar.stuff [2] = 32 ;
                my4x64_scalar.stuff [3] = 99 ;
                OK (GrB_Matrix_setElement_UDT (C, &my4x64_scalar, i, j)) ;
            }
        }
        OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
        OK (GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, sparsity_control)) ;
        OK (GxB_Matrix_fprint (C, "C stuff", GxB_SHORT, NULL)) ;

        for (int i = 0 ; i < 4 ; i++)
        {
            for (int j = 0 ; j < 4 ; j++)
            {
                OK (GrB_Matrix_extractElement_UDT (&my4x64_scalar, C, i, j)) ;
                if (sparsity_control < 8 && i == j) continue ;
                CHECK (my4x64_scalar.stuff [0] == i) ;
                CHECK (my4x64_scalar.stuff [1] == j) ;
                CHECK (my4x64_scalar.stuff [2] == 32) ;
                CHECK (my4x64_scalar.stuff [3] == 99) ;
                printf ("C(%d,%d) = [%d, %d, %d, %d]\n", i, j,
                    my4x64_scalar.stuff [0], my4x64_scalar.stuff [1],
                    my4x64_scalar.stuff [2], my4x64_scalar.stuff [3]) ;
            }
        }

        OK (GxB_Matrix_split (Tiles, 2, 2, Tile_nrows, Tile_ncols, C, NULL)) ;

        for (int k = 0 ; k < 4 ; k++)
        {
            printf ("\n================ Tile %d\n", k) ;
            OK (GxB_Matrix_fprint (Tiles [k], "Tile", GxB_COMPLETE, NULL)) ;
            int istart = (k == 0 || k == 1) ? 0 : 2 ;
            int jstart = (k == 0 || k == 2) ? 0 : 2 ;
            for (int i = 0 ; i < 2 ; i++)
            {
                for (int j = 0 ; j < 2 ; j++)
                {
                    if (sparsity_control < 8 && i+istart == j+jstart) continue ;
                    OK (GrB_Matrix_extractElement_UDT (&my4x64_scalar,
                        Tiles [k], i, j)) ;
                    printf ("Tile(%d,%d) = [%d, %d, %d, %d]\n", i, j,
                        my4x64_scalar.stuff [0], my4x64_scalar.stuff [1],
                        my4x64_scalar.stuff [2], my4x64_scalar.stuff [3]) ;
                    CHECK (my4x64_scalar.stuff [0] == i + istart) ;
                    CHECK (my4x64_scalar.stuff [1] == j + jstart) ;
                    CHECK (my4x64_scalar.stuff [2] == 32) ;
                    CHECK (my4x64_scalar.stuff [3] == 99) ;
                }
            }
            OK (GrB_Matrix_free_(& (Tiles [k]))) ;
        }
    }

    // create an iso matrix
    OK (GrB_Matrix_assign_UDT (C, NULL, NULL, (void *) &my4x64_scalar,
        GrB_ALL, 10, GrB_ALL, 10, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C stuff iso", GxB_COMPLETE, NULL)) ;

    // export as FullC, non-iso
    OK (GxB_Matrix_export_FullC (&C, &type, &nrows, &ncols,
        (void **) &Ax, &Ax_size, NULL, NULL)) ;

    // import as FullC, non-iso
    OK (GxB_Matrix_import_FullC (&C, type, nrows, ncols,
        (void **) &Ax, Ax_size, false, NULL)) ;

    OK (GxB_Matrix_fprint (C, "C stuff iso imported", GxB_COMPLETE, NULL)) ;

    for (int i = 0 ; i < 4 ; i++)
    {
        for (int j = 0 ; j < 4 ; j++)
        {
            OK (GrB_Matrix_extractElement_UDT (&my4x64_scalar, C, i, j)) ;
            printf ("C(%d,%d) = [%d, %d, %d, %d]\n", i, j,
                my4x64_scalar.stuff [0], my4x64_scalar.stuff [1],
                my4x64_scalar.stuff [2], my4x64_scalar.stuff [3]) ;
        }
    }

    // change to iso sparse, and test GB_ix_realloc
    OK (GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GrB_Matrix_assign_UDT (C, NULL, NULL, (void *) &my4x64_scalar,
        GrB_ALL, 10, GrB_ALL, 10, NULL)) ;
    OK (GB_ix_realloc (C, 32)) ;
    OK (GxB_Matrix_fprint (C, "C stuff sparse non-iso", GxB_COMPLETE, NULL)) ;

    // test wait on jumbled matrix (non-iso)
    OK (GrB_Matrix_setElement_UDT (C, &my4x64_scalar, 0, 0)) ;
    my4x64_scalar.stuff [0] = 1007 ;
    OK (GrB_Matrix_setElement_UDT (C, &my4x64_scalar, 1, 1)) ;
    C->jumbled = true ;
    OK (GxB_Matrix_fprint (C, "C stuff jumbled", GxB_COMPLETE, NULL)) ;
    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    OK (GxB_Matrix_fprint (C, "C stuff wait", GxB_COMPLETE, NULL)) ;

    // converting a non-iso matrix to non-iso does nothing
    OK (GB_convert_any_to_non_iso (C, true)) ;
    OK (GxB_Matrix_fprint (C, "C stuff non iso", GxB_COMPLETE, NULL)) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // setElement
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Matrix_assign_FP32 (C, NULL, NULL, 1, GrB_ALL, 10, GrB_ALL, 10,
        NULL)) ;
    OK (GxB_Matrix_fprint (C, "C iso full", GxB_SHORT, NULL)) ;
    OK (GrB_Matrix_setElement_FP32 (C, 2, 0, 0)) ;
    OK (GxB_Matrix_fprint (C, "C non-iso full", GxB_SHORT, NULL)) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // assign_prep: wait if iso property of C is changing
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Matrix_new (&A, GrB_FP32, 4, 4)) ;
    OK (GrB_Matrix_assign_FP32 (A, NULL, NULL, 1, GrB_ALL, 4, GrB_ALL, 4,
        NULL)) ;
    OK (GxB_Matrix_fprint (A, "A iso", GxB_SHORT, NULL)) ;
    OK (GrB_Matrix_assign_(C, NULL, NULL, A, I, 4, I, 4, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C iso with pending", GxB_SHORT, NULL)) ;
    GrB_Matrix_free_(&A) ;

    OK (GrB_Matrix_new (&A, GrB_FP32, 2, 2)) ;
    OK (GrB_Matrix_setElement_FP32 (A, 1.1, 0, 0)) ;
    OK (GrB_Matrix_setElement_FP32 (A, 1.2, 0, 1)) ;
    OK (GrB_Matrix_setElement_FP32 (A, 2.1, 1, 0)) ;
    OK (GrB_Matrix_setElement_FP32 (A, 2.2, 1, 1)) ;
    OK (GxB_Matrix_fprint (A, "A non-iso", GxB_SHORT, NULL)) ;
    OK (GrB_Matrix_assign_(C, NULL, NULL, A, I, 2, I, 2, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C non-iso", GxB_SHORT, NULL)) ;

    GrB_Matrix_free_(&A) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // iso masker
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_FP32, 100, 100)) ;
    OK (GrB_Matrix_new (&M, GrB_BOOL, 100, 100)) ;
    OK (GrB_Matrix_setElement_BOOL (M, true, 0, 0)) ;
    OK (GrB_Matrix_setElement_BOOL (M, true, 1, 0)) ;
    OK (GrB_Matrix_assign_FP32 (C, M, NULL, 1, GrB_ALL, 100, GrB_ALL, 100,
        NULL)) ;
    OK (GxB_Matrix_fprint (C, "C iso", GxB_SHORT, NULL)) ;

    OK (GrB_Matrix_new (&A, GrB_FP32, 100, 100)) ;
    OK (GrB_Matrix_clear (M)) ;
    OK (GrB_Matrix_setElement_BOOL (M, true, 0, 2)) ;
    OK (GrB_Matrix_setElement_BOOL (M, true, 1, 2)) ;
    OK (GrB_Matrix_assign_FP32 (A, M, NULL, 1, GrB_ALL, 100, GrB_ALL, 100,
        NULL)) ;
    OK (GxB_Matrix_fprint (C, "A iso", GxB_SHORT, NULL)) ;

    // C<M>=op(A)
    OK (GrB_Matrix_apply (C, M, NULL, GrB_ABS_FP32, A, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C iso, C<M>=op(A)", GxB_SHORT, NULL)) ;

    GrB_Matrix_free_(&M) ;
    GrB_Matrix_free_(&A) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // bitmap assign
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Matrix_setElement_FP32 (C, 1.1, 0, 0)) ;
    OK (GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
    OK (GrB_Matrix_setElement_FP32 (C, 9.9, 4, 4)) ;
    OK (GrB_Matrix_setElement_FP32 (C, 9.7, 3, 3)) ;
    OK (GxB_Matrix_fprint (C, "C for C<M>=A", GxB_SHORT, NULL)) ;

    OK (GrB_Matrix_new (&A, GrB_FP32, 2, 2)) ;
    OK (GrB_Matrix_setElement_FP32 (A, 1.1, 0, 0)) ;
    OK (GrB_Matrix_setElement_FP32 (A, 1.2, 0, 1)) ;
    OK (GxB_Matrix_Option_set (A, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
    OK (GxB_Matrix_fprint (A, "A for C<M>=A", GxB_SHORT, NULL)) ;

    // C(I,I) = A
    OK (GB_bitmap_assign_6 (C, true,
        I, false, 2, 2, GB_LIST, NULL, I, false, 2, 2, GB_LIST, NULL,
        false, true, A, NULL, GrB_BOOL, GB_ASSIGN, NULL)) ;

    OK (GxB_Matrix_fprint (C, "C after C<M>=A", GxB_SHORT, NULL)) ;

    GrB_Matrix_free_(&A) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // descriptor
    //--------------------------------------------------------------------------

    OK (GrB_Descriptor_new (&desc)) ;
    OK (GrB_Descriptor_set (desc, GrB_MASK, GrB_COMP)) ;
    OK (GxB_Descriptor_fprint (desc, "descriptor !M", GxB_SHORT, NULL)) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // shallow alias
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Matrix_setElement_FP32 (C, 9.9, 4, 4)) ;
    OK (GrB_Matrix_setElement_FP32 (C, 9.7, 3, 3)) ;
    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    OK (GrB_Matrix_dup (&A, C)) ;
    save = A->x ;
    A->x = C->x ;
    A->x_shallow = true ;
    CHECK (GB_any_aliased (A, C)) ;
    A->x = save ;
    A->x_shallow = false ;
    GrB_Matrix_free_(&A) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // vector extractElement with pending work
    //--------------------------------------------------------------------------

    float gamma = 999 ;
    OK (GrB_Vector_new (&w, GrB_FP32, 100)) ;
    OK (GrB_Vector_setElement_FP32 (w, 0, 0)) ;
    OK (GxB_Vector_fprint (w, "w after first setElement", GxB_SHORT, NULL)) ;
    OK (GrB_Vector_setElement_FP64 (w, 1, 1)) ;     // converts to non-iso
    OK (GrB_Vector_setElement_FP32 (w, 2, 2)) ;
    OK (GrB_Vector_setElement_FP32 (w, -1, 1)) ;
    OK (GrB_Vector_setElement_FP32 (w, 3, 2)) ;
    OK (GrB_Vector_setElement_FP64 (w, 2, 0)) ;
    OK (GxB_Vector_fprint (w, "w before extractElement", GxB_SHORT, NULL)) ;
    OK (GrB_Vector_extractElement_FP32 (&gamma, w, 1)) ;
    OK (GxB_Vector_fprint (w, "w after extractElement", GxB_SHORT, NULL)) ;
    CHECK (gamma == -1) ;
    OK (GrB_Vector_extractElement_FP32 (&gamma, w, 0)) ;
    CHECK (gamma == 2) ;
    GrB_Vector_free_(&w) ;

    //--------------------------------------------------------------------------
    // transpose with pending work
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Matrix_setElement_FP32 (C, 8.25, 2, 4)) ;
    OK (GrB_Matrix_setElement_FP32 (C, 9.5, 1, 3)) ;
    OK (GxB_Matrix_fprint (C, "C before transpose", GxB_SHORT, NULL)) ;
    OK (GrB_transpose (C, NULL, NULL, C, NULL)) ;
    OK (GxB_Matrix_fprint (C, "C after transpose", GxB_SHORT, NULL)) ;
    OK (GrB_Matrix_extractElement_FP32 (&gamma, C, 3, 1)) ;
    CHECK (gamma == 9.5) ;
    OK (GrB_Matrix_extractElement_FP32 (&gamma, C, 4, 2)) ;
    CHECK (gamma == 8.25) ;
    OK (GrB_Matrix_nvals (&nvals, C)) ;
    CHECK (nvals == 2) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // GB_ek_slice
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_FP32, 3, 4)) ;
    OK (GrB_Matrix_assign_FP32 (C, NULL, NULL, 1, GrB_ALL, 4, GrB_ALL, 4,
        NULL)) ;
    int64_t A_ek_slicing [1000] ;
    for (int ntasks = 1 ; ntasks < 20 ; ntasks++)
    {
        GB_ek_slice (A_ek_slicing, C, ntasks) ;

        int64_t *kfirst_slice = A_ek_slicing ;
        int64_t *klast_slice  = A_ek_slicing + ntasks ;
        int64_t *pstart_slice = A_ek_slicing + ntasks * 2 ;

        printf ("\n----------------- ntasks: %d\n", ntasks) ;
        for (int tid = 0 ; tid < ntasks ; tid++)
        {
            printf ("task: %3d kfirst: %d klast: %d pfirst: %2d plast: %2d\n",
                tid, kfirst_slice [tid], klast_slice [tid],
                pstart_slice [tid], pstart_slice [tid+1]-1) ;
        }
    }
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // GB_unop_one
    //--------------------------------------------------------------------------

    CHECK (GB_unop_one (GB_UDT_code) == NULL) ;

    //--------------------------------------------------------------------------
    // GB_all_entries_are_iso
    //--------------------------------------------------------------------------

    CHECK (!GB_all_entries_are_iso (NULL)) ;
    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Matrix_assign_FP32 (C, NULL, NULL, 1, GrB_ALL, 4, GrB_ALL, 4,
        NULL)) ;
    OK (GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    OK (GrB_Matrix_wait_(C, GrB_MATERIALIZE)) ;
    CHECK (GB_all_entries_are_iso (C)) ;
    GrB_Matrix_free_(&C) ;

    OK (GrB_Matrix_new (&C, My4x64, 10, 10)) ;
    OK (GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
    my4x64_scalar.stuff [0] = 1 ;
    my4x64_scalar.stuff [1] = 2 ;
    my4x64_scalar.stuff [2] = 3 ;
    my4x64_scalar.stuff [3] = 4 ;
    OK (GrB_Matrix_setElement_UDT (C, &my4x64_scalar, 3, 2)) ;
    OK (GrB_Matrix_setElement_UDT (C, &my4x64_scalar, 0, 0)) ;
    CHECK (!GB_all_entries_are_iso (C)) ;
    OK (GrB_Matrix_wait_(C, GrB_MATERIALIZE)) ;
    CHECK (GB_all_entries_are_iso (C)) ;

    my4x64_scalar.stuff [0] = 4 ;
    OK (GrB_Matrix_setElement_UDT (C, &my4x64_scalar, 4, 4)) ;
    CHECK (!GB_all_entries_are_iso (C)) ;
    OK (GrB_Matrix_wait_(C, GrB_MATERIALIZE)) ;
    CHECK (!GB_all_entries_are_iso (C)) ;
    GrB_Matrix_free_(&C) ;

    //--------------------------------------------------------------------------
    // memory size
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_new (&C, GrB_FP32, 10, 10)) ;
    OK (GrB_Vector_new (&w, GrB_FP32, 100)) ;
    OK (GrB_Scalar_new (&scalar, GrB_FP32)) ;

    size_t size ;
    OK (GxB_Matrix_fprint (C, "empty C for size", GxB_COMPLETE, NULL)) ;
    OK (GxB_Matrix_memoryUsage (&size, C)) ;
    printf ("size of C: %lu bytes\n", size) ;

    OK (GxB_Vector_fprint (w, "empty w for size", GxB_COMPLETE, NULL)) ;
    OK (GxB_Vector_memoryUsage (&size, w)) ;
    printf ("size of w: %lu bytes\n", size) ;

    OK (GxB_Scalar_fprint (scalar, "empty scalar for size",
        GxB_COMPLETE, NULL)) ;
    OK (GxB_Scalar_memoryUsage (&size, scalar)) ;
    printf ("size of scalar: %lu bytes\n", size) ;

    for (int k = 0 ; k < 8 ; k++)
    {
        OK (GrB_Matrix_setElement_FP32 (C, (double) k, k, k+1)) ;
        OK (GrB_Vector_setElement_FP32 (w, (double) k, k)) ;
    }
    OK (GrB_Scalar_setElement_FP32 (scalar, 3.0)) ;

    OK (GxB_Matrix_fprint (C, "non-empty C for size (with pending)",
        GxB_COMPLETE, NULL)) ;
    OK (GxB_Matrix_memoryUsage (&size, C)) ;
    printf ("size of C: %lu bytes\n", size) ;

    OK (GxB_Vector_fprint (w, "non-empty w for size (with pending)",
        GxB_COMPLETE, NULL)) ;
    OK (GxB_Vector_memoryUsage (&size, w)) ;
    printf ("size of w: %lu bytes\n", size) ;

    OK (GxB_Scalar_fprint (scalar, "non-empty scalar for size",
        GxB_COMPLETE, NULL)) ;
    OK (GxB_Scalar_memoryUsage (&size, scalar)) ;
    printf ("size of scalar: %lu bytes\n", size) ;

    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
    OK (GrB_Vector_wait (w, GrB_MATERIALIZE)) ;

    OK (GxB_Matrix_fprint (C, "non-empty C for size (no pending)",
        GxB_COMPLETE, NULL)) ;
    OK (GxB_Matrix_memoryUsage (&size, C)) ;
    printf ("size of C: %lu bytes\n", size) ;

    OK (GxB_Vector_fprint (w, "non-empty w for size (no pending)",
        GxB_COMPLETE, NULL)) ;
    OK (GxB_Vector_memoryUsage (&size, w)) ;
    printf ("size of w: %lu bytes\n", size) ;

    GrB_Matrix_free_(&C) ;
    GrB_Vector_free_(&w) ;
    GrB_Scalar_free_(&scalar) ;

    int shallow = false ;
    GB_Global_stats_mem_shallow_set (true) ;
    CHECK (GB_Global_stats_mem_shallow_get ( )) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &shallow,
        GxB_INCLUDE_READONLY_STATISTICS)) ;
    CHECK (shallow) ;

    shallow = true ;
    GB_Global_stats_mem_shallow_set (false) ;
    CHECK (!GB_Global_stats_mem_shallow_get ( )) ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &shallow,
        GxB_INCLUDE_READONLY_STATISTICS)) ;
    CHECK (!shallow) ;

    OK (GrB_Global_set_INT32 (GrB_GLOBAL, true,
        GxB_INCLUDE_READONLY_STATISTICS)) ;
    CHECK (GB_Global_stats_mem_shallow_get ( )) ;

    shallow = false ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &shallow,
        GxB_INCLUDE_READONLY_STATISTICS)) ;
    CHECK (shallow) ;
    CHECK (GB_Global_stats_mem_shallow_get ( )) ;

    OK (GrB_Global_set_INT32 (GrB_GLOBAL, false,
        GxB_INCLUDE_READONLY_STATISTICS)) ;
    CHECK (!GB_Global_stats_mem_shallow_get ( )) ;

    int64_t nallocs ;
    size_t mem_deep, mem_shallow ;
    GB_memoryUsage (&nallocs, &mem_deep, &mem_shallow, NULL, false) ;
    CHECK (nallocs == 0) ;
    CHECK (mem_deep == 0) ;
    CHECK (mem_shallow == 0) ;

    //--------------------------------------------------------------------------
    // wrapup
    //--------------------------------------------------------------------------

    GrB_Type_free_(&My4x64) ;
    GB_mx_put_global (true) ;   
    fclose (f) ;
    printf ("\nGB_mex_test3: all tests passed\n\n") ;
}

