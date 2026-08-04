//------------------------------------------------------------------------------
// gb_get_first_scalar: x = find (V, 'first')
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#undef  FREE_WORK
#define FREE_WORK           \
    GrB_Vector_free (&T) ;

#undef  FREE_ALL
#define FREE_ALL            \
    FREE_WORK ;             \
    GrB_Scalar_free (x) ;

GrB_Info gb_get_first_scalar
(
    // output:
    GrB_Scalar *x,          // x = find (V, 'first')
    // input:
    GrB_Vector V,
    GrB_Type type,
    const int arena,
    char err [ERRLEN]
)
{ 

    //--------------------------------------------------------------------------
    // get the first entry from a vector V
    //--------------------------------------------------------------------------

    (*x) = NULL ;
    GrB_Vector T = NULL ;

    OK (GxB_Scalar_new_arena (x, type, arena, arena)) ;
    OK (GxB_Vector_new_arena (&T, type, 0, arena, arena)) ;
    OK (GxB_Vector_extractTuples_Vector (NULL, T, V, NULL)) ;
    OK (GrB_Vector_extractElement_Scalar (*x, T, 0)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;
    return (GrB_SUCCESS) ;
}

#undef  FREE_WORK
#define FREE_WORK
#undef  FREE_ALL
#define FREE_ALL

