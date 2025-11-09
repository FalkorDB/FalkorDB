//------------------------------------------------------------------------------
// GxB_Matrix_unpack_CSR: unpack a matrix in CSR format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "import_export/GB_export.h"

#define GB_FREE_ALL ;

GrB_Info GxB_Matrix_unpack_CSR  // unpack a CSR matrix
(
    GrB_Matrix A,       // matrix to unpack (type, nrows, ncols unchanged)
    uint64_t **Ap,      // row "pointers"
    uint64_t **Aj,      // column indices
    void **Ax,          // values
    uint64_t *Ap_size,  // size of Ap in bytes
    uint64_t *Aj_size,  // size of Aj in bytes
    uint64_t *Ax_size,  // size of Ax in bytes
    bool *iso,          // if true, A is iso
    bool *jumbled,      // if true, indices in each row may be unsorted
    const GrB_Descriptor desc
)
{

    //--------------------------------------------------------------------------
    // check inputs and get the descriptor
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_OUTPUT_IS_READONLY (A) ;
    GB_WHERE_1 (A, "GxB_Matrix_unpack_CSR (A, &Ap, &Aj, &Ax, &Ap_size,"
        " &Aj_size, &Ax_size, &iso, &jumbled, desc)") ;
    GB_BURBLE_START ("GxB_Matrix_unpack_CSR") ;

    GB_GET_DESCRIPTOR (info, desc, xx1, xx2, xx3, xx4, xx5, xx6, xx7) ;

    //--------------------------------------------------------------------------
    // ensure the matrix is sparse by-row
    //--------------------------------------------------------------------------

    if (A->is_csc)
    { 
        // A = A', done in-place, to put A in by-row format
        GB_OK (GB_transpose_in_place (A, false, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // finish any pending work
    //--------------------------------------------------------------------------

    if (jumbled == NULL)
    { 
        // the unpacked matrix cannot be jumbled
        GB_MATRIX_WAIT (A) ;
    }
    else
    { 
        // the unpacked matrix is allowed to be jumbled
        GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;
    }

    //--------------------------------------------------------------------------
    // ensure the matrix is sparse
    //--------------------------------------------------------------------------

    GB_OK (GB_convert_any_to_sparse (A, Werk)) ;

    //--------------------------------------------------------------------------
    // unpack the matrix
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_SPARSE (A)) ;
    ASSERT (!(A->is_csc)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (GB_IMPLIES (jumbled == NULL, !GB_JUMBLED (A))) ;
    ASSERT (!GB_PENDING (A)) ;

    int sparsity ;
    bool is_csc ;
    GrB_Type type ;
    uint64_t vlen, vdim ;

    info = GB_export (true, &A, &type, &vlen, &vdim, false,
        Ap,   Ap_size,  // Ap
        NULL, NULL,     // Ah
        NULL, NULL,     // Ab
        Aj,   Aj_size,  // Ai
        Ax,   Ax_size,  // Ax
        NULL, jumbled, NULL,                // jumbled or not
        &sparsity, &is_csc,                 // sparse by row
        iso, Werk) ;

    if (info == GrB_SUCCESS)
    {
        ASSERT (sparsity == GxB_SPARSE) ;
        ASSERT (!is_csc) ;
    }
    GB_BURBLE_END ;
    return (info) ;
}

