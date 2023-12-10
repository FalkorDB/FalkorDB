//------------------------------------------------------------------------------
// GB_mex_AdotB: compute C=spones(Mask).*(A'*B)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns a plain built-in sparse matrix, not a struct.  Only works in double
// and complex.  Input matrices must be built-in sparse matrices, or GraphBLAS
// structs in CSC format.

#include "GB_mex.h"

#define USAGE "C = GB_mex_AdotB (A,B,Mask,flipxy)"

#define FREE_ALL                            \
{                                           \
    GrB_Matrix_free_(&A) ;                  \
    GrB_Matrix_free_(&Aconj) ;              \
    GrB_Matrix_free_(&B) ;                  \
    GrB_Matrix_free_(&C) ;                  \
    GrB_Matrix_free_(&Mask) ;               \
    GrB_Monoid_free_(&add) ;                \
    GrB_Semiring_free_(&semiring) ;         \
    GB_mx_put_global (true) ;               \
}

GrB_Matrix A = NULL, B = NULL, C = NULL, Aconj = NULL, Mask = NULL ;
GrB_Monoid add = NULL ;
GrB_Semiring semiring = NULL ;
GrB_Info adotb_complex (GB_Werk Werk) ;
GrB_Info adotb (GB_Werk Werk) ;
GrB_Index anrows, ancols, bnrows, bncols, mnrows, mncols ;
bool flipxy = false ;
struct GB_Matrix_opaque C_header ;

//------------------------------------------------------------------------------

GrB_Info adotb_complex (GB_Werk Werk)
{
    GrB_Info info = GrB_Matrix_new (&Aconj, Complex, anrows, ancols) ;
    if (info != GrB_SUCCESS) return (info) ;
    info = GxB_Matrix_Option_set (Aconj, GxB_FORMAT, GxB_BY_COL) ;
    if (info != GrB_SUCCESS)
    { 
        GrB_Matrix_free_(&Aconj) ;
        return (info) ;
    }
    info = GrB_Matrix_apply_(Aconj, NULL, NULL, Complex_conj, A, NULL) ;
    if (info != GrB_SUCCESS)
    {
        GrB_Matrix_free_(&Aconj) ;
        return (info) ;
    }

    // force completion
    info = GrB_Matrix_wait_(Aconj, GrB_MATERIALIZE) ;
    if (info != GrB_SUCCESS)
    {
        GrB_Matrix_free_(&Aconj) ;
        return (info) ;
    }

    bool mask_applied = false ;

    GrB_Semiring semiring = Complex_plus_times ;

    if (Mask != NULL)
    {
        // C<M> = A'*B using dot product method
        info = GB_AxB_dot3 (C, false, NULL, Mask, false, Aconj, B, semiring,
            flipxy, Werk) ;
        mask_applied = true ;
    }
    else
    {
        // C = A'*B using dot product method
        mask_applied = false ;  // no mask to apply
        info = GB_AxB_dot2 (C, false, NULL, NULL, false, false,
            false, Aconj, B, semiring, flipxy, Werk) ;
    }

    GrB_Matrix_free_(&Aconj) ;
    return (info) ;
}

//------------------------------------------------------------------------------

GrB_Info adotb (GB_Werk Werk) 
{
    // create the Semiring for regular z += x*y
    GrB_Info info = GrB_Monoid_new_FP64_(&add, GrB_PLUS_FP64, (double) 0) ;
    if (info != GrB_SUCCESS) return (info) ;
    info = GrB_Semiring_new (&semiring, add, GrB_TIMES_FP64) ;
    if (info != GrB_SUCCESS)
    {
        GrB_Monoid_free_(&add) ;
        return (info) ;
    }
    // C = A'*B
    bool mask_applied = false ;

    if (Mask != NULL)
    {
        // C<M> = A'*B using dot product method
        info = GB_AxB_dot3 (C, false, NULL, Mask, false, A, B,
            semiring /* GxB_PLUS_TIMES_FP64 */,
            flipxy, Werk) ;
        mask_applied = true ;
    }
    else
    {
        mask_applied = false ;  // no mask to apply
        info = GB_AxB_dot2 (C, false, NULL, NULL, false, false,
            false, A, B, semiring /* GxB_PLUS_TIMES_FP64 */, flipxy, Werk) ;
    }

    GrB_Monoid_free_(&add) ;
    GrB_Semiring_free_(&semiring) ;
    return (info) ;
}

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

    GB_WERK (USAGE) ;

    // check inputs
    if (nargout > 1 || nargin < 2 || nargin > 4)
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    #define GET_DEEP_COPY ;
    #define FREE_DEEP_COPY ;

    GET_DEEP_COPY ;
    // get A and B (shallow copies)
    A = GB_mx_mxArray_to_Matrix (pargin [0], "A input", false, true) ;
    B = GB_mx_mxArray_to_Matrix (pargin [1], "B input", false, true) ;
    if (A == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("A failed") ;
    }
    if (B == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("B failed") ;
    }

    GrB_Matrix_nrows (&anrows, A) ;
    GrB_Matrix_ncols (&ancols, A) ;

    GrB_Matrix_nrows (&bnrows, B) ;
    GrB_Matrix_ncols (&bncols, B) ;

    if (!A->is_csc || !B->is_csc)
    {
        FREE_ALL ;
        mexErrMsgTxt ("matrices must be CSC only") ;
    }

    if (A->iso || B->iso)
    {
        FREE_ALL ;
        mexErrMsgTxt ("matrices must be non-iso only") ;
    }

    // get Mask (shallow copy)
    if (nargin > 2)
    {
        Mask = GB_mx_mxArray_to_Matrix (pargin [2], "Mask input", false, false);

        GrB_Matrix_nrows (&mnrows, Mask) ;
        GrB_Matrix_ncols (&mncols, Mask) ;

        if (!Mask->is_csc)
        {
            FREE_ALL ;
            mexErrMsgTxt ("matrices must be CSC only") ;
        }

        if (mnrows != ancols || mncols != bncols)
        {
            FREE_ALL ;
            mexErrMsgTxt ("mask wrong dimension") ;
        }
    }

    if (anrows != bnrows)
    {
        FREE_ALL ;
        mexErrMsgTxt ("inner dimensions of A'*B do not match") ;
    }

    if (anrows == 0)
    {
        FREE_ALL ;
        mexErrMsgTxt ("inner dimensions of A'*B must be > 0") ;
    }

    // get flipxy
    GET_SCALAR (3, bool, flipxy, false) ;

    struct GB_Matrix_opaque C_header ;
    C = GB_clear_static_header (&C_header) ;

    if (A->type == Complex)
    {
        // C = A'*B, complex case
        METHOD (adotb_complex (Werk)) ;
    }
    else
    {
        METHOD (adotb (Werk)) ;
    }

    // return C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C AdotB result", false) ;

    FREE_ALL ;
}

