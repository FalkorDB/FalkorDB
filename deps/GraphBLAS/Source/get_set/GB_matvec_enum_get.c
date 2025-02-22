//------------------------------------------------------------------------------
// GB_matvec_enum_get: get an enum field from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"

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

        case GxB_IS_READONLY : 

            (*value) = GB_is_shallow (A) ;
            break ;

        case GxB_ISO : 

            (*value) = A->iso ;
            break ;

        case GxB_HYPER_HASH : 

            (*value) = !(A->no_hyper_hash) ;
            break ;

        case GxB_FORMAT : 

            (*value) = (A->is_csc) ? GxB_BY_COL : GxB_BY_ROW ;
            break ;

        case GxB_OFFSET_INTEGER_HINT : 

            (*value) = A->p_control ;
            break ;

        case GxB_OFFSET_INTEGER_BITS : 

            (*value) = (A->p_is_32) ? 32 : 64 ;
            break ;

        case GxB_COLINDEX_INTEGER_HINT : 

            (*value) = (A->is_csc) ? A->j_control : A->i_control ;
            break ;

        case GxB_COLINDEX_INTEGER_BITS : 

            (*value) = ((A->is_csc) ? A->j_is_32 : A->i_is_32) ? 32 : 64 ;
            break ;

        case GxB_ROWINDEX_INTEGER_HINT : 

            (*value) = (A->is_csc) ? A->i_control : A->j_control ;
            break ;

        case GxB_ROWINDEX_INTEGER_BITS : 

            (*value) = ((A->is_csc) ? A->i_is_32 : A->j_is_32) ? 32 : 64 ;
            break ;

        case GxB_WILL_WAIT : 

            (*value) = GB_ANY_PENDING_WORK (A) || GB_hyper_hash_need (A) ;
            break ;

        default : 
            return (GrB_INVALID_VALUE) ;
    }
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

