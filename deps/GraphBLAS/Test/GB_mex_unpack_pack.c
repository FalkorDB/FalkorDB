//------------------------------------------------------------------------------
// GB_mex_unpack_pack: unpack and then pack a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// For matrices:

    // format_matrix:
    //      case 1 :    // standard CSR
    //      case 2 :    // standard CSC
    //      case 3 :    // hypersparse CSR
    //      case 4 :    // hypersparse CSC
    //      case 5 :    // bitmapR
    //      case 6 :    // bitmapC
    //      case 7 :    // FullR
    //      case 8 :    // FullC
    //      case 9 :    // to bitmap then (hyper+sparse+full)

    // format_export:
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

#include "GB_mex.h"

#define USAGE "C = GB_mex_unpack_pack (A, format_matrix, format_export)"

#define FREE_WORK                                   \
{                                                   \
    if (Cp != NULL) { mxFree (Cp) ; Cp = NULL ; }   \
    if (Ch != NULL) { mxFree (Ch) ; Ch = NULL ; }   \
    if (Cb != NULL) { mxFree (Cb) ; Cb = NULL ; }   \
    if (Ci != NULL) { mxFree (Ci) ; Ci = NULL ; }   \
    if (Cx != NULL) { mxFree (Cx) ; Cx = NULL ; }   \
    GrB_Matrix_free_(&C) ;                          \
}

#define FREE_ALL                        \
{                                       \
    FREE_WORK ;                         \
    GrB_Matrix_free_(&A) ;              \
    GB_mx_put_global (true) ;           \
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

GrB_Matrix A = NULL ;
GrB_Matrix C = NULL ;
GrB_Matrix Y = NULL ;
GrB_Index *Cp = NULL, *Ch = NULL, *Ci = NULL ;
void *Cx = NULL ;
int8_t *Cb = NULL ;
GrB_Index nvec = 0, nvals = 0 ;

GrB_Index Cp_size = 0 ;
GrB_Index Ch_size = 0 ;
GrB_Index Cb_size = 0 ;
GrB_Index Ci_size = 0 ;
GrB_Index Cx_size = 0 ;
bool iso = false ;

int64_t ignore = -1 ;
bool jumbled = false ;
GrB_Info info = GrB_SUCCESS ;

GrB_Info unpack_pack ( int format_matrix, int format_export) ;
GrB_Info vector_unpack_pack ( int format_matrix, int format_export) ;

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
    if (nargout > 1 || nargin != 3)
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

    #define GET_DEEP_COPY           \
        GrB_Matrix_dup (&C, A) ;    \
        GrB_Matrix_wait (C, GrB_MATERIALIZE) ;

    #define FREE_DEEP_COPY  GrB_Matrix_free (&C) ;

    // C = deep copy of A
    GET_DEEP_COPY ;

    // convert matrix, unpack, then import
    if (do_matrix)
    {
        METHOD (unpack_pack (format_matrix, format_export)) ;
    }

    FREE_DEEP_COPY ;
    GET_DEEP_COPY ;

    // convert vector, unpack, then import, if C can be cast as a GrB_Vector
    if (GB_VECTOR_OK (C))
    {
        METHOD (vector_unpack_pack (format_matrix, format_export)) ;
    }

    // return C as a struct and free the GraphBLAS C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;
    FREE_ALL ;
}



//------------------------------------------------------------------------------

GrB_Info unpack_pack
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
        case 9 :    // to bitmap then (hyper+sparse+full)
        //----------------------------------------------------------------------

            OK (GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL, GxB_BITMAP)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL,
               GxB_HYPERSPARSE + GxB_SPARSE + GxB_FULL)) ;
            OK (GxB_Matrix_Option_set_(C, GxB_FORMAT, GxB_BY_COL)) ;
            break ;

        default : mexErrMsgTxt ("invalid mtx format") ;
    }

    //--------------------------------------------------------------------------
    // unpack then pack
    //--------------------------------------------------------------------------

    switch (format_export)
    {

        //----------------------------------------------------------------------
        case 0 :    // standard CSR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_unpack_CSR (C,
                &Cp, &Ci, &Cx, &Cp_size, &Ci_size, &Cx_size, &iso,
                &jumbled, NULL)) ;

            OK (GxB_Matrix_pack_CSR (C,
                &Cp, &Ci, &Cx, Cp_size, Ci_size, Cx_size, iso,
                jumbled, NULL)) ;

            break ;

        //----------------------------------------------------------------------
        case 1 :    // standard CSC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_unpack_CSC (C,
                &Cp, &Ci, &Cx, &Cp_size, &Ci_size, &Cx_size, &iso,
                &jumbled, NULL)) ;

            OK (GxB_Matrix_pack_CSC (C,
                &Cp, &Ci, &Cx, Cp_size, Ci_size, Cx_size, iso,
                jumbled, NULL)) ;

            break ;

        //----------------------------------------------------------------------
        case 2 :    // hypersparse CSR
        //----------------------------------------------------------------------

            OK (GxB_unpack_HyperHash (C, &Y, NULL)) ;

            OK (GxB_Matrix_unpack_HyperCSR (C,
                &Cp, &Ch, &Ci, &Cx,
                &Cp_size, &Ch_size, &Ci_size, &Cx_size, &iso,
                &nvec, &jumbled, NULL)) ;

            OK (GxB_Matrix_pack_HyperCSR (C,
                &Cp, &Ch, &Ci, &Cx,
                Cp_size, Ch_size, Ci_size, Cx_size, iso,
                nvec, jumbled, NULL)) ;

            OK (GxB_pack_HyperHash (C, &Y, NULL)) ;

            // silently check the result to make sure the hyper_hash is valid
            OK (GxB_Matrix_fprint (C, "C", GxB_SILENT, stdout)) ;

            break ;

        //----------------------------------------------------------------------
        case 3 :    // hypersparse CSC
        //----------------------------------------------------------------------

            OK (GxB_unpack_HyperHash (C, &Y, NULL)) ;

            OK (GxB_Matrix_unpack_HyperCSC (C,
                &Cp, &Ch, &Ci, &Cx,
                &Cp_size, &Ch_size, &Ci_size, &Cx_size, &iso,
                &nvec, &jumbled, NULL)) ;

            OK (GxB_Matrix_pack_HyperCSC (C,
                &Cp, &Ch, &Ci, &Cx,
                Cp_size, Ch_size, Ci_size, Cx_size, iso,
                nvec, jumbled, NULL)) ;

            // silently check the result to make sure the hyper_hash is valid
            OK (GxB_pack_HyperHash (C, &Y, NULL)) ;

            OK (GxB_Matrix_fprint (C, "C", GxB_SILENT, stdout)) ;
            break ;

        //----------------------------------------------------------------------
        case 4 :    // bitmapR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_unpack_BitmapR (C,
                &Cb, &Cx, &Cb_size, &Cx_size, &iso, &nvals, NULL)) ;

            OK (GxB_Matrix_pack_BitmapR (C,
                &Cb, &Cx, Cb_size, Cx_size, iso, nvals, NULL)) ;

            break ;

        //----------------------------------------------------------------------
        case 5 :    // bitmapC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_unpack_BitmapC (C,
                &Cb, &Cx, &Cb_size, &Cx_size, &iso, &nvals, NULL)) ;

            OK (GxB_Matrix_pack_BitmapC (C,
                &Cb, &Cx, Cb_size, Cx_size, iso, nvals, NULL)) ;

            break ;

        //----------------------------------------------------------------------
        case 6 :    // FullR
        //----------------------------------------------------------------------

            OK (GxB_Matrix_unpack_FullR (C, &Cx, &Cx_size, &iso, NULL)) ;
            OK (GxB_Matrix_pack_FullR (C, &Cx, Cx_size, iso, NULL)) ;
            break ;

        //----------------------------------------------------------------------
        case 7 :    // FullC
        //----------------------------------------------------------------------

            OK (GxB_Matrix_unpack_FullC (C, &Cx, &Cx_size, &iso, NULL)) ;
            OK (GxB_Matrix_pack_FullC (C, &Cx, Cx_size, iso, NULL)) ;
            break ;

        //----------------------------------------------------------------------
        case 8 :    // standard CSR, not jumbled
        //----------------------------------------------------------------------

            OK (GxB_Matrix_unpack_CSR (C,
                &Cp, &Ci, &Cx, &Cp_size, &Ci_size, &Cx_size, &iso,
                NULL, NULL)) ;

            OK (GxB_Matrix_pack_CSR (C,
                &Cp, &Ci, &Cx, Cp_size, Ci_size, Cx_size, iso, false, NULL)) ;

            break ;

        //----------------------------------------------------------------------
        case 9 :    // standard CSC, not jumbled
        //----------------------------------------------------------------------

            OK (GxB_Matrix_unpack_CSC (C,
                &Cp, &Ci, &Cx, &Cp_size, &Ci_size, &Cx_size, &iso,
                NULL, NULL)) ;

            OK (GxB_Matrix_pack_CSC (C,
                &Cp, &Ci, &Cx, Cp_size, Ci_size, Cx_size, iso, false, NULL)) ;

            break ;

        //----------------------------------------------------------------------
        case 10 :    // hypersparse CSR, not jumbled
        //----------------------------------------------------------------------

            OK (GxB_unpack_HyperHash (C, &Y, NULL)) ;

            OK (GxB_Matrix_unpack_HyperCSR (C,
                &Cp, &Ch, &Ci, &Cx,
                &Cp_size, &Ch_size, &Ci_size, &Cx_size, &iso,
                &nvec, NULL, NULL)) ;

            OK (GxB_Matrix_pack_HyperCSR (C,
                &Cp, &Ch, &Ci, &Cx, Cp_size, Ch_size, Ci_size, Cx_size,
                iso, nvec, false, NULL)) ;

            OK (GxB_pack_HyperHash (C, &Y, NULL)) ;

            break ;

        //----------------------------------------------------------------------
        case 11 :    // hypersparse CSC, not jumbled
        //----------------------------------------------------------------------

            OK (GxB_unpack_HyperHash (C, &Y, NULL)) ;

            OK (GxB_Matrix_unpack_HyperCSC (C,
                &Cp, &Ch, &Ci, &Cx,
                &Cp_size, &Ch_size, &Ci_size, &Cx_size, &iso,
                &nvec, NULL, NULL)) ;

            OK (GxB_Matrix_pack_HyperCSC (C,
                &Cp, &Ch, &Ci, &Cx, Cp_size, Ch_size, Ci_size, Cx_size, iso,
                nvec, false, NULL)) ;

            OK (GxB_pack_HyperHash (C, &Y, NULL)) ;

            break ;


        default : mexErrMsgTxt ("invalid export format") ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------

GrB_Info vector_unpack_pack
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

            OK (GxB_Vector_unpack_CSC ((GrB_Vector) C,
                &Ci, &Cx, &Ci_size, &Cx_size, &iso,
                &nvals, &jumbled, NULL)) ;

            OK (GxB_Vector_pack_CSC ((GrB_Vector) C,
                &Ci, &Cx, Ci_size, Cx_size, iso,
                nvals, jumbled, NULL)) ;

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

            OK (GxB_Vector_unpack_Bitmap ((GrB_Vector) C,
                &Cb, &Cx, &Cb_size, &Cx_size, &iso, &nvals, NULL)) ;

            OK (GxB_Vector_pack_Bitmap ((GrB_Vector) C,
                &Cb, &Cx, Cb_size, Cx_size, iso, nvals, NULL)) ;

            break ;

        //----------------------------------------------------------------------
        case 6 :    // FullR
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 7 :    // FullC
        //----------------------------------------------------------------------

            OK (GxB_Vector_unpack_Full ((GrB_Vector) C,
                &Cx, &Cx_size, &iso, NULL)) ;

            OK (GxB_Vector_pack_Full ((GrB_Vector) C,
                &Cx, Cx_size, iso, NULL)) ;

            break ;

        //----------------------------------------------------------------------
        case 8 :    // standard CSR, not jumbled
        //----------------------------------------------------------------------

            return (GrB_SUCCESS) ;

        //----------------------------------------------------------------------
        case 9 :    // standard CSC, not jumbled
        //----------------------------------------------------------------------

            OK (GxB_Vector_unpack_CSC ((GrB_Vector) C,
                &Ci, &Cx, &Ci_size, &Cx_size, &iso, &nvals, NULL, NULL)) ;

            OK (GxB_Vector_pack_CSC ((GrB_Vector) C,
                &Ci, &Cx, Ci_size, Cx_size, iso, nvals, false, NULL)) ;

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

