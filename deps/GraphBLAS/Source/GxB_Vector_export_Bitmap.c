//------------------------------------------------------------------------------
// GxB_Vector_export_Bitmap: export a bitmap vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Vector_export_Bitmap   // export and free a bitmap vector
(
    GrB_Vector *v,      // handle of vector to export and free
    GrB_Type *type,     // type of vector exported
    GrB_Index *n,       // length of the vector

    int8_t **vb,        // bitmap
    void **vx,          // values
    GrB_Index *vb_size, // size of vb in bytes
    GrB_Index *vx_size, // size of vx in bytes
    bool *iso,          // if true, A is iso

    GrB_Index *nvals,    // # of entries in bitmap
    const GrB_Descriptor desc
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_export_Bitmap (&v, &type, &n, "
        "&vb, &vx, &vb_size, &vx_size, &iso, &nvals, desc)") ;
    // GB_BURBLE_START ("GxB_Vector_export_Bitmap") ;
    GB_RETURN_IF_NULL (v) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*v) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (*v) ;

    //--------------------------------------------------------------------------
    // ensure the vector is bitmap CSC
    //--------------------------------------------------------------------------

    ASSERT ((*v)->is_csc) ;
    GB_OK (GB_convert_any_to_bitmap ((GrB_Matrix) *v, Werk)) ;

    //--------------------------------------------------------------------------
    // export the vector
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_BITMAP (*v)) ;
    ASSERT ((*v)->is_csc) ;
    ASSERT (!GB_ZOMBIES (*v)) ;
    ASSERT (!GB_JUMBLED (*v)) ;
    ASSERT (!GB_PENDING (*v)) ;

    int sparsity ;
    bool is_csc ;
    GrB_Index vdim ;

    info = GB_export (false, (GrB_Matrix *) v, type, n, &vdim, false,
        NULL, NULL,     // Ap
        NULL, NULL,     // Ah
        vb,   vb_size,  // Ab
        NULL, NULL,     // Ai
        vx,   vx_size,  // Ax
        nvals, NULL, NULL,                  // nvals for bitmap
        &sparsity, &is_csc,                 // bitmap by col
        iso, Werk) ;

    if (info == GrB_SUCCESS)
    {
        ASSERT (sparsity == GxB_BITMAP) ;
        ASSERT (is_csc) ;
        ASSERT (vdim == 1) ;
    }
    // GB_BURBLE_END ;
    return (info) ;
}

