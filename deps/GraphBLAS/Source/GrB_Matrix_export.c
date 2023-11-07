//------------------------------------------------------------------------------
// GrB_Matrix_export: export a matrix in CSR, CSC, FullC, FullR, or COO format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Exports the contents of a matrix in one of 3 formats: CSR, CSC, or COO
// (triplet format).  The exported matrix is not modified.  No typecast is
// performed; the output array Ax must be of the same type as the input matrix
// A.

// The required sizes of the Ap, Ai, and Ax arrays are given by
// GrB_Matrix_exportSize.

// The GraphBLAS C API does not have a GrB* method to query the type of a
// GrB_Matrix or the size of a type.  SuiteSparse:GraphBLAS provides
// GxB_Matrix_type_name to query the type of a matrix (returning a string),
// which can be converted into a GrB_Type with GxB_Type_from_name.  The size of
// a type can be queried with GxB_Type_size.  Using these methods, a user
// application can ensure that its Ax array has the correct size for any
// given GrB_Matrix it wishes to export, regardless of its type.

#define GB_FREE_ALL                 \
{                                   \
    GB_Matrix_free (&T) ;           \
}

#include "GB_transpose.h"

//------------------------------------------------------------------------------
// GB_export_worker: export a matrix of any type
//------------------------------------------------------------------------------

static GrB_Info GB_export_worker  // export a matrix
(
    GrB_Index *Ap,          // pointers for CSR, CSC, row indices for COO
    GrB_Index *Ai,          // row indices for CSR, CSC, col indices for COO
    void *Ax,               // values (must match the type of A_input)
    GrB_Index *Ap_len,      // number of entries in Ap (not # of bytes)
    GrB_Index *Ai_len,      // number of entries in Ai (not # of bytes)
    GrB_Index *Ax_len,      // number of entries in Ax (not # of bytes)
    GrB_Format format,      // export format
    GrB_Matrix A_input,     // matrix to export
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;

    GrB_Matrix A = A_input ;
    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = NULL ;

    switch (format)
    {
        case GrB_CSR_FORMAT :
        case GrB_CSC_FORMAT :
        case GrB_COO_FORMAT :
            GB_RETURN_IF_NULL (Ap) ; GB_RETURN_IF_NULL (Ap_len) ;
            GB_RETURN_IF_NULL (Ai) ; GB_RETURN_IF_NULL (Ai_len) ;
        default:
            GB_RETURN_IF_NULL (Ax) ; GB_RETURN_IF_NULL (Ax_len) ;
    }

    // finish any pending work
    GB_MATRIX_WAIT (A) ;

    //--------------------------------------------------------------------------
    // determine current format of A and if a copy is needed
    //--------------------------------------------------------------------------

    int sparsity = GB_sparsity (A) ;
    bool is_csc = A->is_csc ;
    bool make_copy ;
    bool csc_requested ;

    switch (format)
    {
        case GrB_CSR_FORMAT :
            make_copy = !(sparsity == GxB_SPARSE && !is_csc) ;
            csc_requested = false ;
            break ;

        case GrB_CSC_FORMAT :
            make_copy = !(sparsity == GxB_SPARSE && is_csc) ;
            csc_requested = true ;
            break ;

//      case GrB_DENSE_ROW_FORMAT :
//          if (!GB_is_dense (A))
//          {
//              // A must dense or full
//              return (GrB_INVALID_VALUE) ;
//          }
//          make_copy = !(sparsity == GxB_FULL && !is_csc) ;
//          csc_requested = false ;
//          break ;

//      case GrB_DENSE_COL_FORMAT :
//          if (!GB_is_dense (A))
//          {
//              // A must dense or full
//              return (GrB_INVALID_VALUE) ;
//          }
//          make_copy = !(sparsity == GxB_FULL && is_csc) ;
//          csc_requested = true ;
//          break ;

        case GrB_COO_FORMAT : 
            // never make a copy to export in tuple format
            make_copy = false ;
            csc_requested = is_csc ;
            break ;

        default : 
            // unknown format
            return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // create a copy if the matrix is not in the requested format
    //--------------------------------------------------------------------------

    if (make_copy)
    { 
        GB_CLEAR_STATIC_HEADER (T, &T_header) ;
        if (is_csc != csc_requested)
        { 
            // T = A'
            GB_OK (GB_transpose_cast (T, A->type, csc_requested, A, false,
                Werk)) ;
        }
        else
        { 
            // T = A
            GB_OK (GB_dup_worker (&T, A->iso, A, true, A->type)) ;
        }

        switch (format)
        {
            case GrB_CSR_FORMAT :
            case GrB_CSC_FORMAT :
                GB_OK (GB_convert_any_to_sparse (T, Werk)) ;
                break ;
//          case GrB_DENSE_ROW_FORMAT :
//          case GrB_DENSE_COL_FORMAT :
//              GB_convert_any_to_full (T) ;
//              break ;
            default :
                break ;
        }
        A = T ;
    }

    //--------------------------------------------------------------------------
    // export the contents of the matrix
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;

    GrB_Index nvals = GB_nnz (A) ;
    int64_t plen = A->vdim+1 ; 

    switch (format)
    {
        case GrB_CSR_FORMAT : 
        case GrB_CSC_FORMAT : 
            if (plen > (*Ap_len) || nvals > (*Ai_len))
            { 
                GB_FREE_ALL ;
                return (GrB_INSUFFICIENT_SPACE) ;
            }
            GB_memcpy (Ap, A->p, plen  * sizeof (GrB_Index), nthreads_max) ;
            GB_memcpy (Ai, A->i, nvals * sizeof (GrB_Index), nthreads_max) ;
            (*Ap_len) = plen ;
            (*Ai_len) = nvals ;

//      case GrB_DENSE_ROW_FORMAT :
//      case GrB_DENSE_COL_FORMAT :
            if (nvals > (*Ax_len))
            { 
                GB_FREE_ALL ;
                return (GrB_INSUFFICIENT_SPACE) ;
            }
            (*Ax_len) = nvals ;
            ASSERT (csc_requested == A->is_csc) ;
            if (A->iso)
            { 
                // expand the iso A->x into the non-iso array Ax
                ASSERT (nvals > 0) ;
                GB_expand_iso (Ax, nvals, A->x, A->type->size) ;
            }
            else
            { 
                GB_memcpy (Ax, A->x, nvals * A->type->size, nthreads_max) ;
            }
            break ;

        default:
        case GrB_COO_FORMAT : 
            if (nvals > (*Ap_len) || nvals > (*Ai_len) || nvals > (*Ax_len))
            { 
                GB_FREE_ALL ;
                return (GrB_INSUFFICIENT_SPACE) ;
            }
            GB_OK (GB_extractTuples (Ap, Ai, Ax, &nvals, A->type->code, A,
                Werk)) ;
            (*Ap_len) = nvals ;
            (*Ai_len) = nvals ;
            (*Ax_len) = nvals ;
            break ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_export_*: export a matrix of a given type
//------------------------------------------------------------------------------

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

#define GB_EXPORT(prefix,ctype,Type,acode)                                     \
GrB_Info GB_EVAL3 (prefix, _Matrix_export_, Type) /* export a matrix */        \
(                                                                              \
    GrB_Index *Ap,          /* pointers for CSR, CSC, row indices for COO    */\
    GrB_Index *Ai,          /* row indices for CSR, CSC, col indices for COO */\
    ctype *Ax,              /* values (must match the type of A)             */\
    GrB_Index *Ap_len,      /* number of entries in Ap (not # of bytes)      */\
    GrB_Index *Ai_len,      /* number of entries in Ai (not # of bytes)      */\
    GrB_Index *Ax_len,      /* number of entries in Ax (not # of bytes)      */\
    GrB_Format format,      /* export format                                 */\
    GrB_Matrix A            /* matrix to export                              */\
)                                                                              \
{                                                                              \
    GB_WHERE1 (GB_STR(prefix) "_Matrix_export_" GB_STR(Type)                   \
        " (Ap, Ai, Ax, &Ap_len, &Ai_len, &Ax_len, format, A)") ;               \
/*  GB_BURBLE_START (GB_STR(prefix) "_Matrix_export_" GB_STR(Type)) ; */       \
    GB_RETURN_IF_NULL_OR_FAULTY (A) ;                                          \
    if (A->type->code != acode) return (GrB_DOMAIN_MISMATCH) ;                 \
    GrB_Info info = GB_export_worker (Ap, Ai, (void *) Ax,                     \
        Ap_len, Ai_len, Ax_len, format, A, Werk) ;                             \
/*  GB_BURBLE_END ;  */                                                        \
    return (info) ;                                                            \
}

GB_EXPORT (GrB, bool      , BOOL   , GB_BOOL_code  )
GB_EXPORT (GrB, int8_t    , INT8   , GB_INT8_code  )
GB_EXPORT (GrB, int16_t   , INT16  , GB_INT16_code )
GB_EXPORT (GrB, int32_t   , INT32  , GB_INT32_code )
GB_EXPORT (GrB, int64_t   , INT64  , GB_INT64_code )
GB_EXPORT (GrB, uint8_t   , UINT8  , GB_UINT8_code )
GB_EXPORT (GrB, uint16_t  , UINT16 , GB_UINT16_code)
GB_EXPORT (GrB, uint32_t  , UINT32 , GB_UINT32_code)
GB_EXPORT (GrB, uint64_t  , UINT64 , GB_UINT64_code)
GB_EXPORT (GrB, float     , FP32   , GB_FP32_code  )
GB_EXPORT (GrB, double    , FP64   , GB_FP64_code  )
GB_EXPORT (GxB, GxB_FC32_t, FC32   , GB_FC32_code  )
GB_EXPORT (GxB, GxB_FC64_t, FC64   , GB_FC64_code  )
GB_EXPORT (GrB, void      , UDT    , GB_UDT_code   )

