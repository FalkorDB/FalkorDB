//------------------------------------------------------------------------------
// GB_mex_export_import: export and then reimport a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// format_matrix for matrices and vectors:

//      case 1 :    // standard CSR
//      case 2 :    // standard CSC
//      case 3 :    // hypersparse CSR
//      case 4 :    // hypersparse CSC
//      case 5 :    // bitmapR
//      case 6 :    // bitmapC
//      case 7 :    // FullR
//      case 8 :    // FullC
//      case 9 :    // to control == 11, then bitmap

// format_export: for export/import

//      case 0 :    // standard CSR
//      case 1 :    // standard CSC
//      case 2 :    // hypersparse CSR
//      case 3 :    // hypersparse CSC
//      case 4 :    // bitmapR
//      case 5 :    // bitmapC
//      case 6 :    // FullR
//      case 7 :    // FullC
//      case 8 :    // standard CSR, not jumbled
//      case 9 :    // standard CSC, not jumbled
//      case 10 :   // hypersparse CSR, not jumbled
//      case 11 :   // hypersparse CSC, not jumbled

//      case 12 :   // CSR using GrB_Matrix_export/import    (matrices only)
//      case 13 :   // CSC using GrB_Matrix_export/import    (matrices only)
//      case 14 :   // COO using GrB_Matrix_export/import    (matrices only)
//      case 15 :   // FullR using GrB_Matrix_export/import  (matrices only)
//      case 16 :   // FullC using GrB_Matrix_export/import  (matrices only)

#include "GB_mex.h"

#define USAGE "C = GB_mex_export_import (A, format_matrix, format_export, mode)"

#define FREE_WORK                                   \
{                                                   \
    if (Cp != NULL) { mxFree (Cp) ; Cp = NULL ; }   \
    if (Ch != NULL) { mxFree (Ch) ; Ch = NULL ; }   \
    if (Cb != NULL) { mxFree (Cb) ; Cb = NULL ; }   \
    if (Ci != NULL) { mxFree (Ci) ; Ci = NULL ; }   \
    if (Cx != NULL) { mxFree (Cx) ; Cx = NULL ; }   \
    GrB_Matrix_free_(&C) ;                          \
}

#define FREE_ALL                                \
{                                               \
    FREE_WORK ;                                 \
    GrB_Matrix_free_(&A) ;                      \
    GrB_Descriptor_free_(&desc) ;               \
    GB_mx_put_global (true) ;                   \
}

#define OK(method)                              \
{                                               \
    info = method ;                             \
    if (info != GrB_SUCCESS)                    \
    {                                           \
        FREE_WORK ;                             \
        return (info) ;                         \
    }                                           \
}

GrB_Descriptor desc = NULL ;
GrB_Matrix A = NULL ;
GrB_Matrix C = NULL ;
GrB_Index *Cp = NULL, *Ch = NULL, *Ci = NULL, *Tp = NULL, *Ti = NULL ;
void *Cx = NULL, *Tx = NULL ;
int8_t *Cb = NULL ;
GrB_Index nvec = 0, nvals = 0, nrows = 0, ncols = 0 ;
GrB_Type type2 = NULL ;
GrB_Index nrows2, ncols2 ;
size_t typesize ;

GrB_Index Cp_size = 0, Tp_len = 0 ;
GrB_Index Ch_size = 0 ;
GrB_Index Cb_size = 0 ;
GrB_Index Ci_size = 0, Ti_len = 0 ;
GrB_Index Cx_size = 0, Tx_len = 0 ;
bool iso = false ;

int64_t ignore = -1 ;
bool jumbled = false ;
GrB_Type type = NULL ;
GrB_Info info = GrB_SUCCESS ;

GrB_Info export_import ( int format_matrix, int format_export) ;
GrB_Info vector_export_import ( int format_matrix, int format_export) ;

//------------------------------------------------------------------------------
// GB_exporter: export a matrix with GrB_Matrix_export_T
//------------------------------------------------------------------------------

static GrB_Info GB_exporter (GrB_Index *Ap, GrB_Index *Ai, void *Ax,
    GrB_Index *Ap_len, GrB_Index *Ai_len, GrB_Index *Ax_len, GrB_Format format,
    GrB_Matrix A)
{
    switch (A->type->code)
    {
        case GB_BOOL_code   : return (GrB_Matrix_export_BOOL_  (Ap, Ai, (bool       *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_INT8_code   : return (GrB_Matrix_export_INT8_  (Ap, Ai, (int8_t     *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_INT16_code  : return (GrB_Matrix_export_INT16_ (Ap, Ai, (int16_t    *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_INT32_code  : return (GrB_Matrix_export_INT32_ (Ap, Ai, (int32_t    *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_INT64_code  : return (GrB_Matrix_export_INT64_ (Ap, Ai, (int64_t    *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_UINT8_code  : return (GrB_Matrix_export_UINT8_ (Ap, Ai, (uint8_t    *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_UINT16_code : return (GrB_Matrix_export_UINT16_(Ap, Ai, (uint16_t   *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_UINT32_code : return (GrB_Matrix_export_UINT32_(Ap, Ai, (uint32_t   *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_UINT64_code : return (GrB_Matrix_export_UINT64_(Ap, Ai, (uint64_t   *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_FP32_code   : return (GrB_Matrix_export_FP32_  (Ap, Ai, (float      *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_FP64_code   : return (GrB_Matrix_export_FP64_  (Ap, Ai, (double     *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_FC32_code   : return (GxB_Matrix_export_FC32_  (Ap, Ai, (GxB_FC32_t *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_FC64_code   : return (GxB_Matrix_export_FC64_  (Ap, Ai, (GxB_FC64_t *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        case GB_UDT_code    : return (GrB_Matrix_export_UDT_   (Ap, Ai, (void       *) Ax, Ap_len, Ai_len, Ax_len, format, A)) ;
        default             : ;
    }
    mexErrMsgTxt ("unknown type") ;
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GB_importer: import a matrix with GrB_Matrix_import_T
//------------------------------------------------------------------------------

static GrB_Info GB_importer (GrB_Matrix *A, GrB_Type type, GrB_Index nrows,
    GrB_Index ncols, const GrB_Index *Ap, const GrB_Index *Ai, const void *Ax,
    GrB_Index Ap_len, GrB_Index Ai_len, GrB_Index Ax_len, GrB_Format format)
{
    switch (type->code)
    {
        case GB_BOOL_code   : return (GrB_Matrix_import_BOOL_  (A, type, nrows, ncols, Ap, Ai, (const bool       *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_INT8_code   : return (GrB_Matrix_import_INT8_  (A, type, nrows, ncols, Ap, Ai, (const int8_t     *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_INT16_code  : return (GrB_Matrix_import_INT16_ (A, type, nrows, ncols, Ap, Ai, (const int16_t    *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_INT32_code  : return (GrB_Matrix_import_INT32_ (A, type, nrows, ncols, Ap, Ai, (const int32_t    *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_INT64_code  : return (GrB_Matrix_import_INT64_ (A, type, nrows, ncols, Ap, Ai, (const int64_t    *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_UINT8_code  : return (GrB_Matrix_import_UINT8_ (A, type, nrows, ncols, Ap, Ai, (const uint8_t    *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_UINT16_code : return (GrB_Matrix_import_UINT16_(A, type, nrows, ncols, Ap, Ai, (const uint16_t   *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_UINT32_code : return (GrB_Matrix_import_UINT32_(A, type, nrows, ncols, Ap, Ai, (const uint32_t   *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_UINT64_code : return (GrB_Matrix_import_UINT64_(A, type, nrows, ncols, Ap, Ai, (const uint64_t   *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_FP32_code   : return (GrB_Matrix_import_FP32_  (A, type, nrows, ncols, Ap, Ai, (const float      *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_FP64_code   : return (GrB_Matrix_import_FP64_  (A, type, nrows, ncols, Ap, Ai, (const double     *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_FC32_code   : return (GxB_Matrix_import_FC32_  (A, type, nrows, ncols, Ap, Ai, (const GxB_FC32_t *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_FC64_code   : return (GxB_Matrix_import_FC64_  (A, type, nrows, ncols, Ap, Ai, (const GxB_FC64_t *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        case GB_UDT_code    : return (GrB_Matrix_import_UDT_   (A, type, nrows, ncols, Ap, Ai, (const void       *) Ax, Ap_len, Ai_len, Ax_len, format)) ;
        default             : ;
    }
    mexErrMsgTxt ("unknown type") ;
    return (GrB_NOT_IMPLEMENTED) ;
}

//------------------------------------------------------------------------------
// GB_mex_export_import mexFunction
//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    bool malloc_debug = GB_mx_get_global (true) ;

    // check inputs
    if (nargout > 1 || nargin < 1 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get A (shallow copy)
    {
        A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
        if (A == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("A failed") ;
        }
    }

    // get matrix format (1 to 8, and -1 to -8)
    int GET_SCALAR (1, int, format_matrix, 0) ;
    bool do_matrix = (format_matrix > 0) ;
    if (format_matrix < 0)
    {
        format_matrix = -format_matrix ;
    }

    // get export/import format (0 to 11)
    int GET_SCALAR (2, int, format_export, 0) ;

    // get mode: 0:default, 502:secure, else:fast
    int GET_SCALAR (3, int, mode, 0) ;
    if (mode == GxB_DEFAULT)
    {
        desc = NULL ;
    }
    else if (mode == GxB_SECURE_IMPORT)
    {
        GrB_Descriptor_new (&desc) ;
        GrB_Descriptor_set (desc, GxB_IMPORT, GxB_SECURE_IMPORT) ;
    }
    else // mode is GxB_FAST_IMPORT)
    {
        GrB_Descriptor_new (&desc) ;
        GrB_Descriptor_set (desc, GxB_IMPORT, GxB_FAST_IMPORT) ;
    }

    #define GET_DEEP_COPY   GrB_Matrix_dup (&C, A) ;
    #define FREE_DEEP_COPY  GrB_Matrix_free (&C) ;

    // C = deep copy of A
    GET_DEEP_COPY ;

    // convert matrix, export, then import
    if (do_matrix)
    {
        METHOD (export_import (format_matrix, format_export)) ;
    }

    FREE_DEEP_COPY ;
    GET_DEEP_COPY ;

    // convert vector, export, then import, if C can be cast as a GrB_Vector
    if (GB_VECTOR_OK (C) && format_export <= 11)
    {
        METHOD (vector_export_import (format_matrix, format_export)) ;
    }

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;
    FREE_ALL ;
}



//------------------------------------------------------------------------------

GrB_Info export_import
(
    int format_matrix,
    int format_export
)
{

    //--------------------------------------------------------------------------
    // convert C to the requested format
    //--------------------------------------------------------------------------

    switch (format_matrix)
    {

        //----------------------------------------------------------------------
        case 1 :    // standard CSR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_Option_set_(C, GxB_HYPER_SWITCH, GxB_NEVER_HYPER)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_FORMAT, GxB_BY_ROW)) ;
            break ;

        //----------------------------------------------------------------------
        case 2 :    // standard CSC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_Option_set_(C, GxB_HYPER_SWITCH, GxB_NEVER_HYPER)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_FORMAT, GxB_BY_COL)) ;
            break ;

        //----------------------------------------------------------------------
        case 3 :    // hypersparse CSR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_Option_set_(C, GxB_HYPER_SWITCH, GxB_ALWAYS_HYPER)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL,
                GxB_HYPERSPARSE)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_FORMAT, GxB_BY_ROW)) ;
            break ;

        //----------------------------------------------------------------------
        case 4 :    // hypersparse CSC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_Option_set_(C, GxB_HYPER_SWITCH, GxB_ALWAYS_HYPER)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL,
                GxB_HYPERSPARSE)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_FORMAT, GxB_BY_COL)) ;
            break ;

        //----------------------------------------------------------------------
        case 5 :    // bitmapR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_FORMAT, GxB_BY_ROW)) ;
            break ;

        //----------------------------------------------------------------------
        case 6 :    // bitmapC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_FORMAT, GxB_BY_COL)) ;
            break ;

        //----------------------------------------------------------------------
        case 7 :    // FullR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL, GxB_FULL)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_FORMAT, GxB_BY_ROW)) ;
            break ;

        //----------------------------------------------------------------------
        case 8 :    // FullC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL, GxB_FULL)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_FORMAT, GxB_BY_COL)) ;
            break ;

        //----------------------------------------------------------------------
        case 9 :    // to control == 11, then bitmap
        //----------------------------------------------------------------------

            OK (GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL,
               GxB_HYPERSPARSE + GxB_SPARSE + GxB_FULL)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_FORMAT, GxB_BY_COL)) ;
            break ;

        default : mexErrMsgTxt ("invalid mtx format") ;
    }

    //--------------------------------------------------------------------------
    // export then import
    //--------------------------------------------------------------------------

    OK (GxB_Matrix_type (&type2, C)) ;
    OK (GrB_Matrix_nrows (&nrows2, C)) ;
    OK (GrB_Matrix_ncols (&ncols2, C)) ;
    OK (GxB_Type_size (&typesize, type2)) ;

    switch (format_export)
    {

        //----------------------------------------------------------------------
        case 0 :    // standard CSR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_CSR (&C, &type, &nrows, &ncols,
                &Cp, &Ci, &Cx, &Cp_size, &Ci_size, &Cx_size, &iso,
                &jumbled, NULL)) ;

            OK (GxB_Matrix_import_CSR (&C, type, nrows, ncols,
                &Cp, &Ci, &Cx, Cp_size, Ci_size, Cx_size, iso,
                jumbled, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 1 :    // standard CSC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_CSC (&C, &type, &nrows, &ncols,
                &Cp, &Ci, &Cx, &Cp_size, &Ci_size, &Cx_size, &iso,
                &jumbled, NULL)) ;

            OK (GxB_Matrix_import_CSC (&C, type, nrows, ncols,
                &Cp, &Ci, &Cx, Cp_size, Ci_size, Cx_size, iso,
                jumbled, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 2 :    // hypersparse CSR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_HyperCSR (&C, &type, &nrows, &ncols,
                &Cp, &Ch, &Ci, &Cx,
                &Cp_size, &Ch_size, &Ci_size, &Cx_size, &iso,
                &nvec, &jumbled, NULL)) ;

            OK (GxB_Matrix_import_HyperCSR (&C, type, nrows, ncols,
                &Cp, &Ch, &Ci, &Cx,
                Cp_size, Ch_size, Ci_size, Cx_size, iso,
                nvec, jumbled, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 3 :    // hypersparse CSC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_HyperCSC (&C, &type, &nrows, &ncols,
                &Cp, &Ch, &Ci, &Cx,
                &Cp_size, &Ch_size, &Ci_size, &Cx_size, &iso,
                &nvec, &jumbled, NULL)) ;

            OK (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols,
                &Cp, &Ch, &Ci, &Cx,
                Cp_size, Ch_size, Ci_size, Cx_size, iso,
                nvec, jumbled, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 4 :    // bitmapR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_BitmapR (&C, &type, &nrows, &ncols,
                &Cb, &Cx, &Cb_size, &Cx_size, &iso, &nvals, NULL)) ;

            OK (GxB_Matrix_import_BitmapR (&C, type, nrows, ncols,
                &Cb, &Cx, Cb_size, Cx_size, iso, nvals, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 5 :    // bitmapC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_BitmapC (&C, &type, &nrows, &ncols,
                &Cb, &Cx, &Cb_size, &Cx_size, &iso, &nvals, NULL)) ;

            OK (GxB_Matrix_import_BitmapC (&C, type, nrows, ncols,
                &Cb, &Cx, Cb_size, Cx_size, iso, nvals, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 6 :    // FullR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_FullR (&C, &type, &nrows, &ncols,
                &Cx, &Cx_size, &iso, NULL)) ;

            OK (GxB_Matrix_import_FullR (&C, type, nrows, ncols,
                &Cx, Cx_size, iso, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 7 :    // FullC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_FullC (&C, &type, &nrows, &ncols,
                &Cx, &Cx_size, &iso, NULL)) ;

            OK (GxB_Matrix_import_FullC (&C, type, nrows, ncols,
                &Cx, Cx_size, iso, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 8 :    // standard CSR, not jumbled
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_CSR (&C, &type, &nrows, &ncols,
                &Cp, &Ci, &Cx, &Cp_size, &Ci_size, &Cx_size, &iso,
                NULL, NULL)) ;

            OK (GxB_Matrix_import_CSR (&C, type, nrows, ncols,
                &Cp, &Ci, &Cx, Cp_size, Ci_size, Cx_size, iso,
                false, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 9 :    // standard CSC, not jumbled
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_CSC (&C, &type, &nrows, &ncols,
                &Cp, &Ci, &Cx, &Cp_size, &Ci_size, &Cx_size, &iso,
                NULL, NULL)) ;

            OK (GxB_Matrix_import_CSC (&C, type, nrows, ncols,
                &Cp, &Ci, &Cx, Cp_size, Ci_size, Cx_size, iso,
                false, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 10 :    // hypersparse CSR, not jumbled
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_HyperCSR (&C, &type, &nrows, &ncols,
                &Cp, &Ch, &Ci, &Cx,
                &Cp_size, &Ch_size, &Ci_size, &Cx_size, &iso,
                &nvec, NULL, NULL)) ;

            OK (GxB_Matrix_import_HyperCSR (&C, type, nrows, ncols,
                &Cp, &Ch, &Ci, &Cx,
                Cp_size, Ch_size, Ci_size, Cx_size, iso,
                nvec, false, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 11 :    // hypersparse CSC, not jumbled
        //----------------------------------------------------------------------

            OK (GxB_Matrix_export_HyperCSC (&C, &type, &nrows, &ncols,
                &Cp, &Ch, &Ci, &Cx,
                &Cp_size, &Ch_size, &Ci_size, &Cx_size, &iso,
                &nvec, NULL, NULL)) ;

            OK (GxB_Matrix_import_HyperCSC (&C, type, nrows, ncols,
                &Cp, &Ch, &Ci, &Cx,
                Cp_size, Ch_size, Ci_size, Cx_size, iso,
                nvec, false, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 12 :   // CSR using GrB_Matrix_export/import    (matrices only)
        //----------------------------------------------------------------------

            // export in CSR format, then free C
            OK (GrB_Matrix_exportSize (&Tp_len, &Ti_len, &Tx_len,
                GrB_CSR_FORMAT, C)) ;
            Tp = mxMalloc ((Tp_len+1) * sizeof (GrB_Index)) ;
            Ti = mxMalloc ((Ti_len+1) * sizeof (GrB_Index)) ;
            Tx = mxMalloc ((Tx_len+1) * typesize) ;
            OK (GB_exporter (Tp, Ti, Tx, &Tp_len, &Ti_len, &Tx_len, GrB_CSR_FORMAT, C)) ;
            OK (GrB_Matrix_free (&C)) ;
            // import in CSR format, then free Tp, Ti, Tx
            OK (GB_importer (&C, type2, nrows2, ncols2, Tp, Ti, Tx,
                Tp_len, Ti_len, Tx_len, GrB_CSR_FORMAT)) ;
            mxFree (Tp) ;
            mxFree (Ti) ;
            mxFree (Tx) ;
            break ;

        //----------------------------------------------------------------------
        case 13 :   // CSC using GrB_Matrix_export/import    (matrices only)
        //----------------------------------------------------------------------

            // export in CSC format, then free C
            OK (GrB_Matrix_exportSize (&Tp_len, &Ti_len, &Tx_len,
                GrB_CSC_FORMAT, C)) ;
            Tp = mxMalloc ((Tp_len+1) * sizeof (GrB_Index)) ;
            Ti = mxMalloc ((Ti_len+1) * sizeof (GrB_Index)) ;
            Tx = mxMalloc ((Tx_len+1) * typesize) ;
            OK (GB_exporter (Tp, Ti, Tx, &Tp_len, &Ti_len, &Tx_len, GrB_CSC_FORMAT, C)) ;
            OK (GrB_Matrix_free (&C)) ;
            // import in CSC format, then free Tp, Ti, Tx
            OK (GB_importer (&C, type2, nrows2, ncols2, Tp, Ti, Tx,
                Tp_len, Ti_len, Tx_len, GrB_CSC_FORMAT)) ;
            mxFree (Tp) ;
            mxFree (Ti) ;
            mxFree (Tx) ;
            break ;

//      //----------------------------------------------------------------------
//      case 15 :   // FullR using GrB_Matrix_export/import  (matrices only)
//      //----------------------------------------------------------------------
//
//          // export in FullR format, then free C
//          OK (GrB_Matrix_exportSize (&Tp_len, &Ti_len, &Tx_len,
//              GrB_DENSE_ROW_FORMAT, C)) ;
//          Tx = mxMalloc ((Tx_len+1) * typesize) ;
//          OK (GB_exporter (Tp, Ti, Tx, &Tp_len, &Ti_len, &Tx_len, GrB_DENSE_ROW_FORMAT, C)) ;
//          OK (GrB_Matrix_free (&C)) ;
//          // import in FullR format, then free Tx
//          OK (GB_importer (&C, type2, nrows2, ncols2, Tp, Ti, Tx,
//              Tp_len, Ti_len, Tx_len, GrB_DENSE_ROW_FORMAT)) ;
//          mxFree (Tx) ;
//          break ;

//      //----------------------------------------------------------------------
//      case 16 :   // FullC using GrB_Matrix_export/import  (matrices only)
//      //----------------------------------------------------------------------
//
//          // export in FullC format, then free C
//          OK (GrB_Matrix_exportSize (&Tp_len, &Ti_len, &Tx_len,
//              GrB_DENSE_COL_FORMAT, C)) ;
//          Tx = mxMalloc ((Tx_len+1) * typesize) ;
//          OK (GB_exporter (Tp, Ti, Tx, &Tp_len, &Ti_len, &Tx_len, GrB_DENSE_COL_FORMAT, C)) ;
//          OK (GrB_Matrix_free (&C)) ;
//          // import in CSR format, then free Tx
//          OK (GB_importer (&C, type2, nrows2, ncols2, Tp, Ti, Tx,
//              Tp_len, Ti_len, Tx_len, GrB_DENSE_COL_FORMAT)) ;
//          mxFree (Tx) ;
//          break ;

        //----------------------------------------------------------------------
        case 14 :   // COO using GrB_Matrix_export/import    (matrices only)
        //----------------------------------------------------------------------

            // export in COO format, then free C
            OK (GrB_Matrix_exportSize (&Tp_len, &Ti_len, &Tx_len,
                GrB_COO_FORMAT, C)) ;
            Tp = mxMalloc ((Tp_len+1) * sizeof (GrB_Index)) ;
            Ti = mxMalloc ((Ti_len+1) * sizeof (GrB_Index)) ;
            Tx = mxMalloc ((Tx_len+1) * typesize) ;
            info = GB_exporter (Tp, Ti, Tx, &Tp_len, &Ti_len, &Tx_len, GrB_COO_FORMAT, C) ;
            OK (info) ;
            OK (GrB_Matrix_free (&C)) ;
            // import in COO format, then free Tp, Ti, Tx
            info = (GB_importer (&C, type2, nrows2, ncols2, Tp, Ti, Tx,
                Tp_len, Ti_len, Tx_len, GrB_COO_FORMAT)) ;
            OK (info) ;
            mxFree (Tp) ;
            mxFree (Ti) ;
            mxFree (Tx) ;
            break ;

        default : mexErrMsgTxt ("invalid export format") ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------

GrB_Info vector_export_import
(
    int format_matrix,
    int format_export
)
{

    //--------------------------------------------------------------------------
    // convert C as a vector to the requested format, if available
    //--------------------------------------------------------------------------

    switch (format_matrix)
    {

        //----------------------------------------------------------------------
        case 1 :    // standard CSR
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 2 :    // standard CSC
        //----------------------------------------------------------------------

            OK (GxB_Vector_Option_set_((GrB_Vector) C,
                GxB_SPARSITY_CONTROL, GxB_SPARSE)) ;
            break ;

        //----------------------------------------------------------------------
        case 3 :    // hypersparse CSR 
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 4 :    // hypersparse CSC (will be sparse, not hypersparse)
        //----------------------------------------------------------------------

            OK (GxB_Vector_Option_set_((GrB_Vector) C,
                GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE)) ;
            break ;

        //----------------------------------------------------------------------
        case 5 :    // bitmapR
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 6 :    // bitmapC
        case 9 :    // bitmapC
        //----------------------------------------------------------------------

            OK (GxB_Vector_Option_set_((GrB_Vector) C,
                GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
            break ;

        //----------------------------------------------------------------------
        case 7 :    // FullR
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 8 :    // FullC
        //----------------------------------------------------------------------

            OK (GxB_Vector_Option_set_((GrB_Vector) C,
                GxB_SPARSITY_CONTROL, GxB_FULL)) ;
            break ;

        default : mexErrMsgTxt ("invalid format") ;
    }

    //--------------------------------------------------------------------------
    // export then import
    //--------------------------------------------------------------------------

    switch (format_export)
    {

        //----------------------------------------------------------------------
        case 0 :    // standard CSR
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 1 :    // standard CSC
        //----------------------------------------------------------------------

            OK (GxB_Vector_export_CSC ((GrB_Vector *) &C, &type, &nrows,
                &Ci, &Cx, &Ci_size, &Cx_size, &iso,
                &nvals, &jumbled, NULL)) ;

            OK (GxB_Vector_import_CSC ((GrB_Vector *) &C, type, nrows,
                &Ci, &Cx, Ci_size, Cx_size, iso,
                nvals, jumbled, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 2 :    // hypersparse CSR
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 3 :    // hypersparse CSC
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 4 :    // bitmapR
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 5 :    // bitmapC
        //----------------------------------------------------------------------

            OK (GxB_Vector_export_Bitmap ((GrB_Vector *) &C, &type, &nrows,
                &Cb, &Cx, &Cb_size, &Cx_size, &iso, &nvals, NULL)) ;

            OK (GxB_Vector_import_Bitmap ((GrB_Vector *) &C, type, nrows,
                &Cb, &Cx, Cb_size, Cx_size, iso, nvals, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 6 :    // FullR
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 7 :    // FullC
        //----------------------------------------------------------------------

            OK (GxB_Vector_export_Full ((GrB_Vector *) &C, &type, &nrows,
                &Cx, &Cx_size, &iso, NULL)) ;

            OK (GxB_Vector_import_Full ((GrB_Vector *) &C, type, nrows,
                &Cx, Cx_size, iso, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 8 :    // standard CSR, not jumbled
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 9 :    // standard CSC, not jumbled
        //----------------------------------------------------------------------

            OK (GxB_Vector_export_CSC ((GrB_Vector *) &C, &type, &nrows,
                &Ci, &Cx, &Ci_size, &Cx_size, &iso,
                &nvals, NULL, NULL)) ;

            OK (GxB_Vector_import_CSC ((GrB_Vector *) &C, type, nrows,
                &Ci, &Cx, Ci_size, Cx_size, iso,
                nvals, false, desc)) ;

            break ;

        //----------------------------------------------------------------------
        case 10 :    // hypersparse CSR, not jumbled
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 11 :    // hypersparse CSC, not jumbled
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        default : mexErrMsgTxt ("invalid format") ;
    }

    return (GrB_SUCCESS) ;
}

