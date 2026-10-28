//------------------------------------------------------------------------------
// GB_mex_test48: test arena methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"
#include "GB_mex_errors.h"

#undef  FREE_ALL
#define FREE_ALL ;

 void gb_test48_1 (int32_t *z, int32_t *x) ;
 void gb_test48_1 (int32_t *z, int32_t *x) { (*z) = -(*x) ; }
#define GB_TEST48_1 \
"void gb_test48_1 (int32_t *z, int32_t *x) { (*z) = -(*x) ; }"

 void gb_test48_2 (int32_t *z, int32_t *x, int32_t *y) ;
 void gb_test48_2 (int32_t *z, int32_t *x, int32_t *y) { (*z) = (*x) + (*y) ; }
#define GB_TEST48_2 \
"void gb_test48_2 (int32_t *z, int32_t *x, int32_t *y) { (*z) = (*x) + (*y) ; }"

 void gb_test48_3 (int32_t *z, int32_t *x, GrB_Index ix, GrB_Index jx,
    int32_t *y, GrB_Index iy, GrB_Index jy, int32_t *theta) ;
 void gb_test48_3 (int32_t *z, int32_t *x, GrB_Index ix, GrB_Index jx,
    int32_t *y, GrB_Index iy, GrB_Index jy, int32_t *theta)
    {
        (*z) = (*x) + (*y) + ix + jx + iy + jy + (*theta) ;
    }
#define GB_TEST48_3 \
"void gb_test48_3 (int32_t *z, int32_t *x, GrB_Index ix, GrB_Index jx,  \n" \
"   int32_t *y, GrB_Index iy, GrB_Index jy, int32_t *theta)             \n" \
"   {                                                                   \n" \
"       (*z) = (*x) + (*y) + ix + jx + iy + jy + (*theta) ;             \n" \
"   }                                                                   \n"

 void gb_test48_4 (int32_t *z, int32_t *x, GrB_Index ix, GrB_Index jx,
    int32_t *y) ;
 void gb_test48_4 (int32_t *z, int32_t *x, GrB_Index ix, GrB_Index jx,
    int32_t *y)
    {
        (*z) = (*x) + (*y) + ix + jx ;
    }
#define GB_TEST48_4 \
"void gb_test48_4 (int32_t *z, int32_t *x, GrB_Index ix, GrB_Index jx,  \n" \
"   int32_t *y)                                                         \n" \
"   {                                                                   \n" \
"       (*z) = (*x) + (*y) + ix + jx ;                                  \n" \
"   }                                                                   \n"

//------------------------------------------------------------------------------
// GB_mex_test48 mexFunction
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

    //--------------------------------------------------------------------------
    // test arena methods
    //--------------------------------------------------------------------------

    // create a new arena with just malloc/free
    int arena = 3 ;
    OK (GxB_arena_init (3, malloc, NULL, NULL, free)) ;
    int flag = false ;
    OK (GxB_arena_initialized (&flag, arena)) ;
    CHECK (flag == true) ;

    GrB_Descriptor desc = NULL ;
    OK (GxB_Descriptor_new_arena (&desc, arena)) ;

    GrB_Type type = NULL ;
    OK (GxB_Type_new_arena (&type, sizeof (int32_t), "gb_test48_int32",
        "typedef int32_t gb_test48_int32", arena)) ;

    GrB_UnaryOp unop = NULL ;
    OK (GxB_UnaryOp_new_arena (&unop, gb_test48_1, GrB_INT32, GrB_INT32,
        "gb_test48_1", GB_TEST48_1, arena)) ;

    GrB_BinaryOp binop = NULL ;
    OK (GxB_BinaryOp_new_arena (&binop, gb_test48_2, GrB_INT32, GrB_INT32,
        GrB_INT32, "gb_test48_2", GB_TEST48_2, arena)) ;

    GxB_IndexBinaryOp idxbinop = NULL ;
    OK (GxB_IndexBinaryOp_new_arena (&idxbinop, gb_test48_3, GrB_INT32,
        GrB_INT32, GrB_INT32, GrB_INT32, "gb_test48_3", GB_TEST48_3, arena)) ;

    GrB_IndexUnaryOp idxunop = NULL ;
    OK (GxB_IndexUnaryOp_new_arena (&idxunop, gb_test48_4, GrB_INT32,
        GrB_INT32, GrB_INT32, "gb_test48_4", GB_TEST48_4, arena)) ;

    GrB_Monoid monoid = NULL ;
    OK (GxB_Monoid_new_arena_INT32 (&monoid, binop, 0, arena)) ;

    GrB_Semiring semiring = NULL ;
    OK (GxB_Semiring_new_arena (&semiring, monoid, binop, arena)) ;

    GrB_Matrix A = NULL, A2 = NULL, A3 = NULL ;
    OK (GxB_Matrix_new_arena (&A, GrB_INT32, 4, 4, arena, arena)) ;
    OK (GxB_Matrix_dup_arena (&A2, A, arena, arena)) ;
    OK (GxB_Matrix_reshapeDup_arena (&A3, A, true, 8, 2, arena, arena, NULL)) ;

    GrB_Vector V = NULL, V2 = NULL ;
    OK (GxB_Vector_new_arena (&V, GrB_INT32, 4, arena, arena)) ;
    OK (GxB_Vector_dup_arena (&V2, V, arena, arena)) ;

    GrB_Scalar S = NULL, S2 = NULL ;
    OK (GxB_Scalar_new_arena (&S, GrB_INT32, arena, arena)) ;
    OK (GxB_Scalar_dup_arena (&S2, S, arena, arena)) ;

    GxB_Context Context = NULL ;
    OK (GxB_Context_new_arena (&Context, arena)) ;

    GxB_Container Container = NULL ;
    OK (GxB_Container_new_arena (&Container, arena, arena)) ;

    GxB_Iterator Iterator = NULL ;
    OK (GxB_Iterator_new_arena (&Iterator, arena)) ;

    GrB_Matrix *Tiles = (GrB_Matrix *) mxCalloc (4 * 4, sizeof (GrB_Matrix)) ;
    GrB_Index Tile_nrows [2] = {2, 2} ;
    GrB_Index Tile_ncols [2] = {2, 2} ;
    OK (GxB_Matrix_split_arena (Tiles, 2, 2, Tile_nrows, Tile_ncols, A,
        arena, arena, NULL)) ;

    GrB_Matrix C = NULL ;
    OK (GxB_Matrix_diag_arena (&C, V, 0, arena, arena)) ;

    void *blob = NULL ;
    GrB_Index blob_size = 0 ;
    OK (GxB_Matrix_serialize_arena (&blob, &blob_size, A, arena, NULL)) ;

    void *blob2 = NULL ;
    GrB_Index blob2_size = 0 ;
    OK (GxB_Vector_serialize_arena (&blob2, &blob2_size, V, arena, NULL)) ;

    GrB_Matrix C2 = NULL ;
    OK (GxB_Matrix_deserialize_arena (&C2, GrB_INT32, blob, blob_size,
        arena, arena, NULL)) ;

    GrB_Vector W2 = NULL ;
    OK (GxB_Vector_deserialize_arena (&W2, GrB_INT32, blob2, blob2_size,
        arena, arena, NULL)) ;

    //--------------------------------------------------------------------------
    // error handling
    //--------------------------------------------------------------------------

    int expected = GrB_INVALID_VALUE ;

    arena = 4 ;
    OK (GxB_arena_initialized (&flag, arena)) ;
    CHECK (flag == false) ;

    GrB_Semiring_free (&semiring) ;
    ERR (GxB_Semiring_new_arena (&semiring, monoid, binop, arena)) ;

    GrB_Monoid_free (&monoid) ;
    ERR (GxB_Monoid_new_arena_INT32 (&monoid, binop, 0, arena)) ;

    GrB_Descriptor_free (&desc) ;
    ERR (GxB_Descriptor_new_arena (&desc, arena)) ;

    GrB_Type_free (&type) ;
    ERR (GxB_Type_new_arena (&type, sizeof (int32_t), "gb_test48_int32",
        "typedef int32_t gb_test48_int32", arena)) ;

    GrB_UnaryOp_free (&unop) ;
    ERR (GxB_UnaryOp_new_arena (&unop, gb_test48_1, GrB_INT32, GrB_INT32,
        "gb_test48_1", GB_TEST48_1, arena)) ;

    GrB_BinaryOp_free (&binop) ;
    ERR (GxB_BinaryOp_new_arena (&binop, gb_test48_2, GrB_INT32, GrB_INT32,
        GrB_INT32, "gb_test48_2", GB_TEST48_2, arena)) ;

    GxB_IndexBinaryOp_free (&idxbinop) ;
    ERR (GxB_IndexBinaryOp_new_arena (&idxbinop, gb_test48_3, GrB_INT32,
        GrB_INT32, GrB_INT32, GrB_INT32, "gb_test48_3", GB_TEST48_3, arena)) ;

    GrB_IndexUnaryOp_free (&idxunop) ;
    ERR (GxB_IndexUnaryOp_new_arena (&idxunop, gb_test48_4, GrB_INT32,
        GrB_INT32, GrB_INT32, "gb_test48_4", GB_TEST48_4, arena)) ;

    for (int k = 0 ; k < 4 ; k++)
    { 
        GrB_Matrix T = Tiles [k] ;
        GrB_Matrix_free (&T) ;
    }

    ERR (GxB_Matrix_split_arena (Tiles, 2, 2, Tile_nrows, Tile_ncols, A,
        arena, arena, NULL)) ;

    GrB_Matrix_free (&C) ;
    ERR (GxB_Matrix_diag_arena (&C, V, 0, arena, arena)) ;

    GrB_Matrix_free (&C2) ;
    ERR (GxB_Matrix_deserialize_arena (&C2, GrB_INT32, blob, blob_size,
        arena, arena, NULL)) ;

    GrB_Vector_free (&W2) ;
    ERR (GxB_Vector_deserialize_arena (&W2, GrB_INT32, blob2, blob2_size,
        arena, arena, NULL)) ;

    free (blob) ;
    blob = NULL ;
    ERR (GxB_Matrix_serialize_arena (&blob, &blob_size, A, arena, NULL)) ;

    free (blob2) ;
    blob2 = NULL ;
    ERR (GxB_Vector_serialize_arena (&blob2, &blob2_size, V, arena, NULL)) ;

    GrB_Matrix_free (&A2) ;
    ERR (GxB_Matrix_dup_arena (&A2, A, arena, arena)) ;

    GrB_Matrix_free (&A3) ;
    ERR (GxB_Matrix_reshapeDup_arena (&A3, A, true, 8, 2, arena, arena, NULL)) ;

    GrB_Matrix_free (&A) ;
    ERR (GxB_Matrix_new_arena (&A, GrB_INT32, 4, 4, arena, arena)) ;

    GrB_Vector_free (&V2) ;
    ERR (GxB_Vector_dup_arena (&V2, V, arena, arena)) ;

    GrB_Vector_free (&V) ;
    ERR (GxB_Vector_new_arena (&V, GrB_INT32, 4, arena, arena)) ;

    GrB_Scalar_free (&S2) ;
    ERR (GxB_Scalar_dup_arena (&S2, S, arena, arena)) ;

    GrB_Scalar_free (&S) ;
    ERR (GxB_Scalar_new_arena (&S, GrB_INT32, arena, arena)) ;

    GxB_Context_free (&Context) ;
    ERR (GxB_Context_new_arena (&Context, arena)) ;

    GxB_Container_free (&Container) ;
    ERR (GxB_Container_new_arena (&Container, arena, arena)) ;

    GxB_Iterator_free (&Iterator) ;
    ERR (GxB_Iterator_new_arena (&Iterator, arena)) ;

    //--------------------------------------------------------------------------
    // finalize GraphBLAS
    //--------------------------------------------------------------------------

    mxFree (Tiles) ;
    GB_mx_put_global (true) ;
    printf ("GB_mex_test48:  all tests passed\n") ;
}

