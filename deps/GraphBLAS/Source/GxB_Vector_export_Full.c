//------------------------------------------------------------------------------
// GxB_Vector_export_Full: export a full vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Vector_export_Full   // export and free a full vector
(
    GrB_Vector *v,      // handle of vector to export and free
    GrB_Type *type,     // type of vector exported
    GrB_Index *n,       // length of the vector

    void **vx,          // values
    GrB_Index *vx_size, // size of vx in bytes
    bool *iso,          // if true, v is iso

    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_export_Full (&v, &type, &n, "
        "&vx, &vx_size, &iso, desc)") ;
    // GB_BURBLE_START ("GxB_Vector_export_Full") ;
    GB_RETURN_IF_NULL (v) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*v) ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    GB_MATRIX_WAIT (*v) ;
    if (!GB_is_dense ((GrB_Matrix) (*v)))
    { 
        // v must be dense or full
        return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // ensure the vector is full CSC
    //--------------------------------------------------------------------------

    ASSERT ((*v)->is_csc) ;
    GB_convert_any_to_full ((GrB_Matrix) *v) ;

    //--------------------------------------------------------------------------
    // export the vector
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_FULL (*v)) ;
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
        NULL, NULL,     // Ab
        NULL, NULL,     // Ai
        vx,   vx_size,  // Ax
        NULL, NULL, NULL,
        &sparsity, &is_csc,                 // full by col
        iso, Werk) ;

    if (info == GrB_SUCCESS)
    {
        ASSERT (sparsity == GxB_FULL) ;
        ASSERT (is_csc) ;
        ASSERT (vdim == 1) ;
    }
    // GB_BURBLE_END ;
    return (info) ;
}

