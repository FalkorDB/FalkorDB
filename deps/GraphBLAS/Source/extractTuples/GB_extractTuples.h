//------------------------------------------------------------------------------
// GB_extractTuples.h: definitions for GB_extractTuples and related methods
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#ifndef GB_EXTRACTTUPLES_H
#define GB_EXTRACTTUPLES_H

#include "GB.h"

GrB_Info GB_extract_vector_list // extract vector list from a matrix
(
    // output:
    void *J,                    // size nnz(A) or more
    // input:
    bool is_32,                 // if true, J is 32-bit; else 64-bit
    const GrB_Matrix A,
    GB_Werk Werk
) ;

GrB_Info GB_extractTuples       // extract all tuples from a matrix
(
    void *I_out,                // array for returning row indices of tuples
    bool I_is_32,               // if true, I is 32-bit; else 64 bit
    void *J_out,                // array for returning col indices of tuples
    bool J_is_32,               // if true, J is 32-bit; else 64 bit
    void *X,                    // array for returning values of tuples
    uint64_t *p_nvals,          // I,J,X size on input; # tuples on output
    const GrB_Type xtype,       // type of array X
    const GrB_Matrix A,         // matrix to extract tuples from
    GB_Werk Werk
) ;

GrB_Info GB_extractTuples_prep
(
    GrB_Vector V,               // an output vector for I, J, or X
    uint64_t nvals,             // # of values V must hold
    const GrB_Type vtype        // desired type of V
) ;

#endif

