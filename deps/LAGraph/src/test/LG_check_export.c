//------------------------------------------------------------------------------
// LG_check_export: export G->A for testing
//------------------------------------------------------------------------------

// LAGraph, (c) 2019-2022 by The LAGraph Contributors, All Rights Reserved.
// SPDX-License-Identifier: BSD-2-Clause
//
// For additional details (including references to third party source code and
// other files) see the LICENSE file or contact permission@sei.cmu.edu. See
// Contributors.txt for a full list of contributors. Created, in part, with
// funding and support from the U.S. Government (see Acknowledgments.txt file).
// DM22-0790

// Contributed by Timothy A. Davis, Texas A&M University

//------------------------------------------------------------------------------

// Export G->A in CSR format, for testing only.
// See test_export for a brutal memory test of this method.

#define LG_FREE_ALL                             \
{                                               \
    LAGraph_Free ((void **) Ap_handle, NULL) ;  \
    LAGraph_Free ((void **) Aj_handle, NULL) ;  \
    LAGraph_Free ((void **) Ax_handle, NULL) ;  \
}

#include "LG_internal.h"
#include "LG_test.h"

int LG_check_export
(
    // input
    LAGraph_Graph G,        // export G->A in CSR format
    // output
    GrB_Index **Ap_handle,  // size Ap_len on output
    GrB_Index **Aj_handle,  // size Aj_len on output
    void **Ax_handle,       // size Ax_len * typesize on output
    GrB_Index *Ap_len,
    GrB_Index *Aj_len,
    GrB_Index *Ax_len,
    size_t *typesize,       // size of the type of A
    char *msg
)
{
    LG_CLEAR_MSG ;

    GrB_Index *Ap = NULL, *Aj = NULL ;
    void *Ax = NULL ;
    LG_TRY (LAGraph_CheckGraph (G, msg)) ;

    LG_ASSERT_MSG (Ap_handle != NULL, GrB_NULL_POINTER, "&Ap is NULL") ;
    LG_ASSERT_MSG (Aj_handle != NULL, GrB_NULL_POINTER, "&Aj is NULL") ;
    LG_ASSERT_MSG (Ax_handle != NULL, GrB_NULL_POINTER, "&Ax is NULL") ;
    LG_ASSERT_MSG (Ap_len != NULL, GrB_NULL_POINTER, "&Ap_len is NULL") ;
    LG_ASSERT_MSG (Aj_len != NULL, GrB_NULL_POINTER, "&Aj_len is NULL") ;
    LG_ASSERT_MSG (Ax_len != NULL, GrB_NULL_POINTER, "&Ax_len is NULL") ;
    LG_ASSERT_MSG (typesize != NULL, GrB_NULL_POINTER, "&typesize is NULL") ;

    // get the type of G->A
    GrB_Type atype = NULL ;
    char atype_name [LAGRAPH_MAX_NAME_LEN] ;
    LG_TRY (LAGraph_Matrix_TypeName (atype_name, G->A, msg)) ;
    LG_TRY (LAGraph_TypeFromName (&atype, atype_name, msg)) ;

    size_t s = 0 ;
    if      (atype == GrB_BOOL  ) s = sizeof (bool    ) ;
    else if (atype == GrB_INT8  ) s = sizeof (int8_t  ) ;
    else if (atype == GrB_INT16 ) s = sizeof (int16_t ) ;
    else if (atype == GrB_INT32 ) s = sizeof (int32_t ) ;
    else if (atype == GrB_INT64 ) s = sizeof (int64_t ) ;
    else if (atype == GrB_UINT8 ) s = sizeof (uint8_t ) ;
    else if (atype == GrB_UINT16) s = sizeof (uint16_t) ;
    else if (atype == GrB_UINT32) s = sizeof (uint32_t) ;
    else if (atype == GrB_UINT64) s = sizeof (uint64_t) ;
    else if (atype == GrB_FP32  ) s = sizeof (float   ) ;
    else if (atype == GrB_FP64  ) s = sizeof (double  ) ;
    LG_ASSERT_MSG (s != 0, GrB_NOT_IMPLEMENTED, "unsupported type") ;
    (*typesize) = s ;

    GRB_TRY (GrB_Matrix_exportSize (Ap_len, Aj_len, Ax_len, GrB_CSR_FORMAT,
        G->A)) ;
    LG_TRY (LAGraph_Malloc ((void **) Ap_handle, *Ap_len, sizeof (GrB_Index), msg)) ;
    LG_TRY (LAGraph_Malloc ((void **) Aj_handle, *Aj_len, sizeof (GrB_Index), msg)) ;
    LG_TRY (LAGraph_Malloc ((void **) Ax_handle, *Ax_len, s, msg)) ;
    Ap = (*Ap_handle) ;
    Aj = (*Aj_handle) ;
    Ax = (*Ax_handle) ;

    if      (atype == GrB_BOOL  )
    {
        GRB_TRY (GrB_Matrix_export (Ap, Aj, (bool     *) Ax,
            Ap_len, Aj_len, Ax_len, GrB_CSR_FORMAT, G->A)) ;
    }
    else if (atype == GrB_INT8  )
    {
        GRB_TRY (GrB_Matrix_export (Ap, Aj, (int8_t   *) Ax,
            Ap_len, Aj_len, Ax_len, GrB_CSR_FORMAT, G->A)) ;
    }
    else if (atype == GrB_INT16 )
    {
        GRB_TRY (GrB_Matrix_export (Ap, Aj, (int16_t  *) Ax,
            Ap_len, Aj_len, Ax_len, GrB_CSR_FORMAT, G->A)) ;
    }
    else if (atype == GrB_INT32 )
    {
        GRB_TRY (GrB_Matrix_export (Ap, Aj, (int32_t  *) Ax,
            Ap_len, Aj_len, Ax_len, GrB_CSR_FORMAT, G->A)) ;
    }
    else if (atype == GrB_INT64 )
    {
        GRB_TRY (GrB_Matrix_export (Ap, Aj, (int64_t  *) Ax,
            Ap_len, Aj_len, Ax_len, GrB_CSR_FORMAT, G->A)) ;
    }
    else if (atype == GrB_UINT8 )
    {
        GRB_TRY (GrB_Matrix_export (Ap, Aj, (uint8_t  *) Ax,
            Ap_len, Aj_len, Ax_len, GrB_CSR_FORMAT, G->A)) ;
    }
    else if (atype == GrB_UINT16)
    {
        GRB_TRY (GrB_Matrix_export (Ap, Aj, (uint16_t *) Ax,
            Ap_len, Aj_len, Ax_len, GrB_CSR_FORMAT, G->A)) ;
    }
    else if (atype == GrB_UINT32)
    {
        GRB_TRY (GrB_Matrix_export (Ap, Aj, (uint32_t *) Ax,
            Ap_len, Aj_len, Ax_len, GrB_CSR_FORMAT, G->A)) ;
    }
    else if (atype == GrB_UINT64)
    {
        GRB_TRY (GrB_Matrix_export (Ap, Aj, (uint64_t *) Ax,
            Ap_len, Aj_len, Ax_len, GrB_CSR_FORMAT, G->A)) ;
    }
    else if (atype == GrB_FP32  )
    {
        GRB_TRY (GrB_Matrix_export (Ap, Aj, (float    *) Ax,
            Ap_len, Aj_len, Ax_len, GrB_CSR_FORMAT, G->A)) ;
    }
    else if (atype == GrB_FP64  )
    {
        GRB_TRY (GrB_Matrix_export (Ap, Aj, (double   *) Ax,
            Ap_len, Aj_len, Ax_len, GrB_CSR_FORMAT, G->A)) ;
    }

    return (GrB_SUCCESS) ;
}
