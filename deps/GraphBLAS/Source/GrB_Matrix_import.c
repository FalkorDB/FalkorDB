//------------------------------------------------------------------------------
// GrB_Matrix_import: import a matrix in CSR, CSC, FullC, FullR, or COO format
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// No typecasting is done.  The type of entries in the Ax array must match
// the GrB_Type type parameter.

#include "GB_export.h"
#include "GB_build.h"

#define GB_FREE_ALL                 \
{                                   \
    GB_Matrix_free (A) ;            \
    GB_FREE (&Ap_copy, Ap_size) ;   \
    GB_FREE (&Ai_copy, Ai_size) ;   \
    GB_FREE (&Ax_copy, Ax_size) ;   \
}

//------------------------------------------------------------------------------
// GB_import_worker: import a matrix of any type
//------------------------------------------------------------------------------

static GrB_Info GB_import_worker   // import a matrix of any type
(
    GrB_Matrix *A,          // handle of matrix to create
    GrB_Type type,          // type of matrix to create
    GrB_Index nrows,        // number of rows of the matrix
    GrB_Index ncols,        // number of columns of the matrix
    const GrB_Index *Ap,    // pointers for CSR, CSC, row indices for COO
    const GrB_Index *Ai,    // row indices for CSR, CSC, col indices for COO
    const void *Ax,         // values (must match the GrB_Type type parameter)
    GrB_Index Ap_len,       // number of entries in Ap (not # of bytes)
    GrB_Index Ai_len,       // number of entries in Ai (not # of bytes)
    GrB_Index Ax_len,       // number of entries in Ax (not # of bytes)
    GrB_Format format,      // import format
    GB_Werk Werk
)
{ 

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_RETURN_IF_NULL (A) ;
    (*A) = NULL ;
    GB_RETURN_IF_NULL (Ax) ;
    ASSERT_TYPE_OK (type, "type for GrB_Matrix_import", GB0) ;
    GrB_Info info ;

    // GrB_Matrix_import has no descritptor so it only supports a secure import
    bool fast_import = false ;

    if (nrows > GB_NMAX || ncols > GB_NMAX || Ap_len > GB_NMAX
        || Ai_len > GB_NMAX || Ax_len > GB_NMAX)
    { 
        // problem is too large
        return (GrB_INVALID_VALUE) ;
    }

    GrB_Index nvals = 0 ;
    bool ok = true ;
    int64_t plen = (format == GrB_CSR_FORMAT) ? (nrows+1) : (ncols+1) ;

    switch (format)
    {

        case GrB_CSR_FORMAT :
        case GrB_CSC_FORMAT :

            GB_RETURN_IF_NULL (Ap) ;
            GB_RETURN_IF_NULL (Ai) ;
            if (Ap_len < plen)
            { 
                // Ap is too small
                return (GrB_INVALID_VALUE) ;
            }
            nvals = Ap [plen-1] ;
            if (Ai_len < nvals || Ax_len < nvals || nvals > GB_NMAX)
            { 
                // Ai and/or Ax are too small or problem is too large
                return (GrB_INVALID_VALUE) ;
            }
            break ;

//      case GrB_DENSE_ROW_FORMAT :
//      case GrB_DENSE_COL_FORMAT :
//
//          ok = GB_uint64_multiply (&nvals, nrows, ncols) ;
//          if (!ok || Ax_len < nvals)
//          {
//              // Ap, Ai, and Ax must all have the same size
//              return (GrB_INVALID_VALUE) ;
//          }
//          break ;

        case GrB_COO_FORMAT :

            GB_RETURN_IF_NULL (Ap) ;
            GB_RETURN_IF_NULL (Ai) ;
            nvals = Ap_len ;
            if (Ai_len != nvals || Ax_len != nvals)
            { 
                // Ap, Ai, and Ax must all have the same size
                return (GrB_INVALID_VALUE) ;
            }
            break ;

        default :

            // unknown format
            return (GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // allocate copies of Ap, Ai, and Ax to be imported
    //--------------------------------------------------------------------------

    GrB_Index *Ap_copy = NULL ; size_t Ap_size = 0 ;
    GrB_Index *Ai_copy = NULL ; size_t Ai_size = 0 ;
    GB_void   *Ax_copy = NULL ; size_t Ax_size = 0 ;
    size_t typesize = type->size ;

    // Ap_copy, Ai_copy, Ax_copy are GB_MALLOC'ed so they are already in the
    // debug memtable.  Thus, GB_import does not add them again to the
    // memtable (with add_to_memtable set to false).

    switch (format)
    {
        case GrB_CSR_FORMAT : 
        case GrB_CSC_FORMAT : 
            Ap_copy = GB_MALLOC (plen,           GrB_Index, &Ap_size) ;
            Ai_copy = GB_MALLOC (nvals,          GrB_Index, &Ai_size) ;
            Ax_copy = GB_MALLOC (nvals*typesize, GB_void,   &Ax_size) ; // x:OK
            ok = (Ap_copy != NULL && Ai_copy != NULL && Ax_copy != NULL) ;
            break ;

//      case GrB_DENSE_ROW_FORMAT :
//      case GrB_DENSE_COL_FORMAT :
//          Ax_copy = GB_MALLOC (nvals*typesize, GB_void,   &Ax_size) ; // x:OK
//          ok = (Ax_copy != NULL) ;
//          break ;

        default : // GrB_COO_FORMAT, nothing to allocate
            break ;
    }

    if (!ok)
    { 
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    //--------------------------------------------------------------------------
    // determine the # of threads to use
    //--------------------------------------------------------------------------

    int nthreads_max = GB_Context_nthreads_max ( ) ;

    //--------------------------------------------------------------------------
    // copy the user input arrays
    //--------------------------------------------------------------------------

    switch (format)
    {
        case GrB_CSR_FORMAT : 
        case GrB_CSC_FORMAT : 
            GB_memcpy (Ap_copy, Ap, plen  * sizeof (GrB_Index), nthreads_max) ;
            GB_memcpy (Ai_copy, Ai, nvals * sizeof (GrB_Index), nthreads_max) ;
//      case GrB_DENSE_ROW_FORMAT :
//      case GrB_DENSE_COL_FORMAT :
            GB_memcpy (Ax_copy, Ax, nvals * typesize          , nthreads_max) ;
            break ;
        default : // GrB_COO_FORMAT, nothing to copy
            break ;
    }

    //--------------------------------------------------------------------------
    // import the matrix
    //--------------------------------------------------------------------------

    switch (format)
    {

        case GrB_CSR_FORMAT : 

            GB_OK (GB_import (false, A, type, ncols, nrows, false,
                &Ap_copy, Ap_size,  // Ap
                NULL, 0,            // Ah
                NULL, 0,            // Ab
                &Ai_copy, Ai_size,  // Ai
                (void **)
                &Ax_copy, Ax_size,  // Ax
                0, true, 0,         // CSR format may be jumbled
                GxB_SPARSE, false,  // sparse by row
                false,              // not iso
                fast_import,
                false,              // do not add to memtable
                Werk)) ;
            break ;

        case GrB_CSC_FORMAT : 

            GB_OK (GB_import (false, A, type, nrows, ncols, false,
                &Ap_copy, Ap_size,  // Ap
                NULL, 0,            // Ah
                NULL, 0,            // Ab
                &Ai_copy, Ai_size,  // Ai
                (void **)
                &Ax_copy, Ax_size,  // Ax
                0, true, 0,         // CSC format may be jumbled
                GxB_SPARSE, true,   // sparse by column
                false,              // not iso
                fast_import,
                false,              // do not add to memtable
                Werk)) ;
            break ;

//      case GrB_DENSE_ROW_FORMAT :
//
//          GB_OK (GB_import (false, A, type, ncols, nrows, false,
//              NULL, 0,            // Ap
//              NULL, 0,            // Ah
//              NULL, 0,            // Ab
//              NULL, 0,            // Ai
//              (void **)
//              &Ax_copy, Ax_size,  // Ax
//              0, false, 0,        // cannot be jumbled
//              GxB_FULL, false,    // full by row
//              false,              // not iso
//              fast_import,
//              false,              // do not add to memtable
//              Werk)) ;
//          break ;

//      case GrB_DENSE_COL_FORMAT :
//
//          GB_OK (GB_import (false, A, type, nrows, ncols, false,
//              NULL, 0,            // Ap
//              NULL, 0,            // Ah
//              NULL, 0,            // Ab
//              NULL, 0,            // Ai
//              (void **)
//              &Ax_copy, Ax_size,  // Ax
//              0, false, 0,        // cannot be jumbled
//              GxB_FULL, true,     // full by column
//              false,              // not iso
//              fast_import,
//              false,              // do not add to memtable
//              Werk)) ;
//          break ;

        default : // GrB_COO_FORMAT
            {
                // build A as hypersparse by row or by column
                int64_t *no_I_work = NULL ; size_t I_work_size = 0 ;
                int64_t *no_J_work = NULL ; size_t J_work_size = 0 ;
                GB_void *no_X_work = NULL ; size_t X_work_size = 0 ;
                bool is_csc = GB_Global_is_csc_get ( ) ;
                int64_t vlen = is_csc ? nrows : ncols ;
                int64_t vdim = is_csc ? ncols : nrows ;

                // allocate the header for A
                GB_OK (GB_new (A, // new header
                    type, vlen, vdim, GB_Ap_null, is_csc, GxB_AUTO_SPARSITY,
                    GB_Global_hyper_switch_get ( ), 0)) ;

                // build A from the input triplets
                GB_OK (GB_builder (
                    *A,             // create A using a dynamic header
                    type,           // the type of A
                    vlen,
                    vdim,
                    is_csc,         // CSR/CSC format
                    &no_I_work,     // I_work_handle, not used here
                    &I_work_size,
                    &no_J_work,     // J_work_handle, not used here
                    &J_work_size,
                    &no_X_work,     // X_work_handle, not used here
                    &X_work_size,
                    false,          // known_sorted: not yet known
                    false,          // known_no_duplicates: not yet known
                    0,              // I_work, J_work, and X_work not used here
                    true,           // A is a GrB_Matrix
                    (int64_t *) (is_csc ? Ap : Ai),     // row/col indices
                    (int64_t *) (is_csc ? Ai : Ap),     // col/row indices
                    (const GB_void *) Ax,               // values
                    false,          // matrix is not iso
                    nvals,          // number of tuples
                    NULL,           // implicit SECOND operator for duplicates
                    type,           // type of the X array
                    true,           // burble is allowed
                    Werk
                )) ;
            }
            break ;
    }

    ASSERT_MATRIX_OK (*A, "A imported", GB0) ;

    //--------------------------------------------------------------------------
    // determine if A is iso
    //--------------------------------------------------------------------------

    if (GB_check_if_iso (*A))
    { 
        // All entries in A are the same; convert A to iso
        GBURBLE ("(import post iso) ") ;
        (*A)->iso = true ;
        GB_OK (GB_convert_any_to_iso (*A, NULL)) ;
    }

    //--------------------------------------------------------------------------
    // conform the matrix to its desired sparsity and return result
    //--------------------------------------------------------------------------

    GB_OK (GB_conform (*A, Werk)) ;
    ASSERT_MATRIX_OK (*A, "final A imported", GB0) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GrB_Matrix_import_*: import a matrix of a given type
//------------------------------------------------------------------------------

#define GB_IMPORT(prefix,ctype,T,acode)                                        \
GrB_Info GB_EVAL3 (prefix, _Matrix_import_, T) /* import a matrix */           \
(                                                                              \
    GrB_Matrix *A,          /* handle of matrix to create                    */\
    GrB_Type type,          /* type of matrix to create                      */\
    GrB_Index nrows,        /* number of rows of the matrix                  */\
    GrB_Index ncols,        /* number of columns of the matrix               */\
    const GrB_Index *Ap,    /* pointers for CSR, CSC, row indices for COO    */\
    const GrB_Index *Ai,    /* row indices for CSR, CSC, col indices for COO */\
    const ctype *Ax,        /* values (must match GrB_Type type parameter)   */\
    GrB_Index Ap_len,       /* number of entries in Ap (not # of bytes)      */\
    GrB_Index Ai_len,       /* number of entries in Ai (not # of bytes)      */\
    GrB_Index Ax_len,       /* number of entries in Ax (not # of bytes)      */\
    GrB_Format format       /* import format                                 */\
)                                                                              \
{                                                                              \
    GB_WHERE1 (GB_STR(prefix) "_Matrix_import_" GB_STR(T) " (&A, type, nrows," \
        " ncols, Ap, Ai, Ax, Ap_len, Ai_len, Ax_len, format)") ;               \
/*  GB_BURBLE_START (GB_STR(prefix) "_Matrix_import_" GB_STR(T)) ;  */         \
    GB_RETURN_IF_NULL_OR_FAULTY (type) ;                                       \
    if (type->code != acode) return (GrB_DOMAIN_MISMATCH) ;                    \
    GrB_Info info = GB_import_worker (A, type, nrows, ncols, Ap, Ai,           \
        (const void *) Ax, Ap_len, Ai_len, Ax_len, format, Werk) ;             \
/*  GB_BURBLE_END ; */                                                         \
    return (info) ;                                                            \
}

GB_IMPORT (GrB, bool      , BOOL   , GB_BOOL_code  )
GB_IMPORT (GrB, int8_t    , INT8   , GB_INT8_code  )
GB_IMPORT (GrB, int16_t   , INT16  , GB_INT16_code )
GB_IMPORT (GrB, int32_t   , INT32  , GB_INT32_code )
GB_IMPORT (GrB, int64_t   , INT64  , GB_INT64_code )
GB_IMPORT (GrB, uint8_t   , UINT8  , GB_UINT8_code )
GB_IMPORT (GrB, uint16_t  , UINT16 , GB_UINT16_code)
GB_IMPORT (GrB, uint32_t  , UINT32 , GB_UINT32_code)
GB_IMPORT (GrB, uint64_t  , UINT64 , GB_UINT64_code)
GB_IMPORT (GrB, float     , FP32   , GB_FP32_code  )
GB_IMPORT (GrB, double    , FP64   , GB_FP64_code  )
GB_IMPORT (GxB, GxB_FC32_t, FC32   , GB_FC32_code  )
GB_IMPORT (GxB, GxB_FC64_t, FC64   , GB_FC64_code  )
GB_IMPORT (GrB, void      , UDT    , GB_UDT_code   )

