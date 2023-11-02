//------------------------------------------------------------------------------
// GxB_Vector_export_CSC: export a vector in CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Vector_export_CSC  // export and free a CSC vector
(
    GrB_Vector *v,      // handle of vector to export and free
    GrB_Type *type,     // type of vector exported
    GrB_Index *n,       // length of the vector

    GrB_Index **vi,     // indices
    void **vx,          // values
    GrB_Index *vi_size, // size of Ai in bytes
    GrB_Index *vx_size, // size of Ax in bytes
    bool *iso,          // if true, A is iso

    GrB_Index *nvals,   // # of entries in vector
    bool *jumbled,      // if true, indices may be unsorted
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_export_CSC (&v, &type, &n, "
        "&vi, &vx, &vi_size, &vx_size, &iso, &nvals, &jumbled, desc)") ;
    // GB_BURBLE_START ("GxB_Vector_export_CSC") ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_RETURN_IF_NULL (v) ;
    GB_RETURN_IF_NULL_OR_FAULTY (*v) ;
    GB_RETURN_IF_NULL (nvals) ;
    ASSERT_VECTOR_OK (*v, "v to export", GB0) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    if (jumbled == NULL)
    { 
        // the exported vector cannot be jumbled
        GB_MATRIX_WAIT (*v) ;
    }
    else
    { 
        // the exported vector is allowed to be jumbled
        GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (*v) ;
    }

    //--------------------------------------------------------------------------
    // ensure the vector is sparse
    //--------------------------------------------------------------------------

    GB_OK (GB_convert_any_to_sparse ((GrB_Matrix) *v, Werk)) ;

    //--------------------------------------------------------------------------
    // export the vector
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_SPARSE (*v)) ;
    ASSERT ((*v)->is_csc) ;
    ASSERT (!GB_ZOMBIES (*v)) ;
    ASSERT (GB_IMPLIES (jumbled == NULL, !GB_JUMBLED (*v))) ;
    ASSERT (!GB_PENDING (*v)) ;

    int sparsity ;
    bool is_csc ;
    GrB_Index vdim ;

    info = GB_export (false, (GrB_Matrix *) v, type, n, &vdim, true,
        NULL, NULL,     // Ap
        NULL, NULL,     // Ah
        NULL, NULL,     // Ab
        vi,   vi_size,  // Ai
        vx,   vx_size,  // Ax
        nvals, jumbled, NULL,               // jumbled or not
        &sparsity, &is_csc,                 // sparse by col
        iso, Werk) ;

    if (info == GrB_SUCCESS)
    { 
        ASSERT (sparsity == GxB_SPARSE) ;
        ASSERT (is_csc) ;
        ASSERT (vdim == 1) ;
    }
    // GB_BURBLE_END ;
    return (info) ;
}

