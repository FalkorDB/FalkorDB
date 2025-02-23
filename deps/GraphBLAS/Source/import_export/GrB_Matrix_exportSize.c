//------------------------------------------------------------------------------
// GrB_Matrix_exportSize: determine sizes of arrays for GrB_Matrix_export
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "transpose/GB_transpose.h"
#define GB_FREE_ALL ;

GrB_Info GrB_Matrix_exportSize  // determine sizes of user arrays for export
(
    uint64_t *Ap_len,       // # of entries required for Ap (not # of bytes)
    uint64_t *Ai_len,       // # of entries required for Ai (not # of bytes)
    uint64_t *Ax_len,       // # of entries required for Ax (not # of bytes)
    int format,             // export format                   
    GrB_Matrix A            // matrix to export
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE_1 (A, "GrB_Matrix_exportSize "
        "(&Ap_len, &Ai_len, &Ax_len, format, A)") ;
    GB_RETURN_IF_NULL (A) ;
    GB_RETURN_IF_NULL (Ap_len) ;
    GB_RETURN_IF_NULL (Ai_len) ;
    GB_RETURN_IF_NULL (Ax_len) ;

    uint64_t nvals ;
    GB_OK (GB_nvals (&nvals, A, Werk)) ;
    (*Ax_len) = nvals ;

    //--------------------------------------------------------------------------
    // determine the sizes of Ap and Ai for each format
    //--------------------------------------------------------------------------

    switch (format)
    {
        case GrB_CSR_FORMAT :
            (*Ap_len) = GB_NROWS (A) + 1 ;
            (*Ai_len) = nvals ;
            break ;

        case GrB_CSC_FORMAT :
            (*Ap_len) = GB_NCOLS (A) + 1 ;
            (*Ai_len) = nvals ;
            break ;

//      case GrB_DENSE_ROW_FORMAT :
//      case GrB_DENSE_COL_FORMAT :
//          (*Ap_len) = 0 ;
//          (*Ai_len) = 0 ;
//          if (!GB_is_dense (A))
//          {
//              // A must dense or full
//              return (GrB_INVALID_VALUE) ;
//          }
//          break ;

        case GrB_COO_FORMAT : 
            (*Ap_len) = nvals ;
            (*Ai_len) = nvals ;
            break ;

        default :
            // unknown format
            return (GrB_INVALID_VALUE) ;
    }

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

