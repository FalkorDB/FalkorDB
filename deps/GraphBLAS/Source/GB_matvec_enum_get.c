//------------------------------------------------------------------------------
// GB_matvec_enum_get: get an enum field from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_get_set.h"

GrB_Info GB_matvec_enum_get (GrB_Matrix A, int32_t *value, int field)
{
    switch (field)
    {
        case GrB_STORAGE_ORIENTATION_HINT : 

            (*value) = (A->is_csc) ? GrB_COLMAJOR : GrB_ROWMAJOR ;
            break ;

        case GrB_EL_TYPE_CODE : 

            (*value) = GB_type_code_get (A->type->code) ;
            break ;

        case GxB_SPARSITY_CONTROL : 

            (*value) = A->sparsity_control ;
            break ;

        case GxB_SPARSITY_STATUS : 

            (*value) = GB_sparsity (A) ;
            break ;

        case GxB_FORMAT : 

            (*value) = (A->is_csc) ? GxB_BY_COL : GxB_BY_ROW ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

