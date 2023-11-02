//------------------------------------------------------------------------------
// GxB_Vector_unpack_CSC: unpack a vector in CSC format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Vector_unpack_CSC  // unpack a CSC vector
(
    GrB_Vector v,       // vector to unpack (type and length unchanged)
    GrB_Index **vi,     // indices
    void **vx,          // values
    GrB_Index *vi_size, // size of vi in bytes
    GrB_Index *vx_size, // size of vx in bytes
    bool *iso,          // if true, v is iso
    GrB_Index *nvals,   // # of entries in vector
    bool *jumbled,      // if true, indices may be unsorted
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_Vector_unpack_CSC (v, "
        "&vi, &vx, &vi_size, &vx_size, &iso, &nvals, &jumbled, desc)") ;
    GB_BURBLE_START ("GxB_Vector_unpack_CSC") ;
    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;
    GB_RETURN_IF_NULL_OR_FAULTY (v) ;
    GB_RETURN_IF_NULL (nvals) ;

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    if (jumbled == NULL)
    { 
        // the unpacked vector cannot be jumbled
        GB_MATRIX_WAIT (v) ;
    }
    else
    { 
        // the unpacked vector is allowed to be jumbled
        GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (v) ;
    }

    //--------------------------------------------------------------------------
    // ensure the vector is sparse
    //--------------------------------------------------------------------------

    GB_OK (GB_convert_any_to_sparse ((GrB_Matrix) v, Werk)) ;

    //--------------------------------------------------------------------------
    // unpack the vector
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_SPARSE (v)) ;
    ASSERT (v->is_csc) ;
    ASSERT (!GB_ZOMBIES (v)) ;
    ASSERT (GB_IMPLIES (jumbled == NULL, !GB_JUMBLED (v))) ;
    ASSERT (!GB_PENDING (v)) ;

    int sparsity ;
    bool is_csc ;
    GrB_Type type ;
    GrB_Index vlen, vdim ;

    info = GB_export (true, (GrB_Matrix *) (&v), &type, &vlen, &vdim, true,
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
    GB_BURBLE_END ;
    return (info) ;
}

