//------------------------------------------------------------------------------
// GB_export.h: definitions for import/export
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_EXPORT_H
#define GB_EXPORT_H
#include "GB_transpose.h"


GrB_Info GB_import      // import/pack a matrix in any format
(
    bool packing,       // pack if true, create and import false

    GrB_Matrix *A,      // handle of matrix to create, or pack
    GrB_Type type,      // type of matrix to create
    GrB_Index vlen,     // vector length
    GrB_Index vdim,     // vector dimension
    bool is_sparse_vector,      // true if A is a sparse GrB_Vector

    // the 5 arrays:
    GrB_Index **Ap,     // pointers, for sparse and hypersparse formats.
    GrB_Index Ap_size,  // size of Ap in bytes

    GrB_Index **Ah,     // vector indices for hypersparse matrices
    GrB_Index Ah_size,  // size of Ah in bytes

    int8_t **Ab,        // bitmap, for bitmap format only.
    GrB_Index Ab_size,  // size of Ab in bytes

    GrB_Index **Ai,     // indices for hyper and sparse formats
    GrB_Index Ai_size,  // size of Ai in bytes

    void **Ax,          // values
    GrB_Index Ax_size,  // size of Ax in bytes

    // additional information for specific formats:
    GrB_Index nvals,    // # of entries for bitmap format, or for a vector
                        // in CSC format.
    bool jumbled,       // if true, sparse/hypersparse may be jumbled.
    GrB_Index nvec,     // size of Ah for hypersparse format.

    // information for all formats:
    int sparsity,       // hypersparse, sparse, bitmap, or full
    bool is_csc,        // if true then matrix is by-column, else by-row
    bool iso,           // if true then A is iso and only one entry is provided
                        // in Ax, regardless of nvals(A).
    // fast vs secure import:
    bool fast_import,   // if true: trust the data, if false: check it

    bool add_to_memtable,   // if true: add to debug memtable
    GB_Werk Werk
) ;

GrB_Info GB_export      // export/unpack a matrix in any format
(
    bool unpacking,     // unpack if true, export and free if false

    GrB_Matrix *A,      // handle of matrix to export and free, or unpack
    GrB_Type *type,     // type of matrix to export
    GrB_Index *vlen,    // vector length
    GrB_Index *vdim,    // vector dimension
    bool is_sparse_vector,      // true if A is a sparse GrB_Vector

    // the 5 arrays:
    GrB_Index **Ap,     // pointers
    GrB_Index *Ap_size, // size of Ap in bytes

    GrB_Index **Ah,     // vector indices
    GrB_Index *Ah_size, // size of Ah in bytes

    int8_t **Ab,        // bitmap
    GrB_Index *Ab_size, // size of Ab in bytes

    GrB_Index **Ai,     // indices
    GrB_Index *Ai_size, // size of Ai in bytes

    void **Ax,          // values
    GrB_Index *Ax_size, // size of Ax in bytes

    // additional information for specific formats:
    GrB_Index *nvals,   // # of entries for bitmap format.
    bool *jumbled,      // if true, sparse/hypersparse may be jumbled.
    GrB_Index *nvec,    // size of Ah for hypersparse format.

    // information for all formats:
    int *sparsity,      // hypersparse, sparse, bitmap, or full
    bool *is_csc,       // if true then matrix is by-column, else by-row
    bool *iso,          // if true then A is iso and only one entry is returned
                        // in Ax, regardless of nvals(A).
    GB_Werk Werk
) ;

#endif

