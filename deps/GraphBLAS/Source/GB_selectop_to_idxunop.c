//------------------------------------------------------------------------------
// GB_selectop_to_idxunop: convert a GxB_SelectOp to a GrB_IndexUnaryOp
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The GxB_SelectOp is historical, and part of its functionality is now
// deprecated.  In particular, user-defined GxB_SelectOps are not supported.
// The GxB_EQ* and GxB_NE* operators are not supported for user-defined types.
// Otherwise, this function converts a supported GxB_SelectOp and its optional
// Thunk scalar into its corresponding GrB_IndexUnaryOp and its required
// NewThunk scalar.

#define GB_FREE_ALL                             \
{                                               \
    GB_Matrix_free ((GrB_Matrix *) &NewThunk) ; \
}

#include "GB_select.h"

GrB_Info GB_selectop_to_idxunop
(
    // output:
    GrB_IndexUnaryOp *idxunop_handle,
    GrB_Scalar *NewThunk_handle,
    // input:
    GxB_SelectOp selectop,
    GrB_Scalar Thunk,
    GrB_Type atype,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Scalar NewThunk = NULL ;
    GrB_IndexUnaryOp idxunop = NULL ;
    GB_RETURN_IF_NULL_OR_FAULTY (selectop) ;
    GB_RETURN_IF_FAULTY (Thunk) ;

    ASSERT (idxunop_handle != NULL) ;
    ASSERT (NewThunk_handle != NULL) ;
    ASSERT_OP_OK ((GB_Operator) selectop, "selop to convert to idxunop", GB0) ;
    ASSERT_TYPE_OK (atype, "atype for selectop_to_idxunop", GB0) ;
    ASSERT (GB_IS_SELECTOP_CODE (selectop->opcode)) ;

    (*idxunop_handle) = NULL ;
    (*NewThunk_handle) = NULL ;
    GB_Type_code acode = atype->code ;

    //--------------------------------------------------------------------------
    // find the new GrB_IndexUnaryOp
    //--------------------------------------------------------------------------

    switch (selectop->opcode)
    {
        // built-in positional select operators: thunk optional; defaults to 0
        case GB_TRIL_selop_code      : idxunop = GrB_TRIL    ; break ;
        case GB_TRIU_selop_code      : idxunop = GrB_TRIU    ; break ;
        case GB_DIAG_selop_code      : idxunop = GrB_DIAG    ; break ;
        case GB_OFFDIAG_selop_code   : idxunop = GrB_OFFDIAG ; break ;

        // built-in select comparators, thunk defaults to zero
        case GB_NONZERO_selop_code   : Thunk = NULL ;
        case GB_NE_THUNK_selop_code  : 

            switch (acode)
            {
                case GB_BOOL_code    : idxunop = GrB_VALUENE_BOOL   ; break ;
                case GB_INT8_code    : idxunop = GrB_VALUENE_INT8   ; break ;
                case GB_INT16_code   : idxunop = GrB_VALUENE_INT16  ; break ;
                case GB_INT32_code   : idxunop = GrB_VALUENE_INT32  ; break ;
                case GB_INT64_code   : idxunop = GrB_VALUENE_INT64  ; break ;
                case GB_UINT8_code   : idxunop = GrB_VALUENE_UINT8  ; break ;
                case GB_UINT16_code  : idxunop = GrB_VALUENE_UINT16 ; break ;
                case GB_UINT32_code  : idxunop = GrB_VALUENE_UINT32 ; break ;
                case GB_UINT64_code  : idxunop = GrB_VALUENE_UINT64 ; break ;
                case GB_FP32_code    : idxunop = GrB_VALUENE_FP32   ; break ;
                case GB_FP64_code    : idxunop = GrB_VALUENE_FP64   ; break ;
                case GB_FC32_code    : idxunop = GxB_VALUENE_FC32   ; break ;
                case GB_FC64_code    : idxunop = GxB_VALUENE_FC64   ; break ;
                default:;
            }
            break ;

        case GB_EQ_ZERO_selop_code   : Thunk = NULL ;
        case GB_EQ_THUNK_selop_code  : 

            switch (acode)
            {
                case GB_BOOL_code    : idxunop = GrB_VALUEEQ_BOOL   ; break ;
                case GB_INT8_code    : idxunop = GrB_VALUEEQ_INT8   ; break ;
                case GB_INT16_code   : idxunop = GrB_VALUEEQ_INT16  ; break ;
                case GB_INT32_code   : idxunop = GrB_VALUEEQ_INT32  ; break ;
                case GB_INT64_code   : idxunop = GrB_VALUEEQ_INT64  ; break ;
                case GB_UINT8_code   : idxunop = GrB_VALUEEQ_UINT8  ; break ;
                case GB_UINT16_code  : idxunop = GrB_VALUEEQ_UINT16 ; break ;
                case GB_UINT32_code  : idxunop = GrB_VALUEEQ_UINT32 ; break ;
                case GB_UINT64_code  : idxunop = GrB_VALUEEQ_UINT64 ; break ;
                case GB_FP32_code    : idxunop = GrB_VALUEEQ_FP32   ; break ;
                case GB_FP64_code    : idxunop = GrB_VALUEEQ_FP64   ; break ;
                case GB_FC32_code    : idxunop = GxB_VALUEEQ_FC32   ; break ;
                case GB_FC64_code    : idxunop = GxB_VALUEEQ_FC64   ; break ;
                default:;
            }
            break ;

        case GB_GT_ZERO_selop_code   : Thunk = NULL ;
        case GB_GT_THUNK_selop_code  : 

            switch (acode)
            {
                case GB_BOOL_code    : idxunop = GrB_VALUEGT_BOOL   ; break ;
                case GB_INT8_code    : idxunop = GrB_VALUEGT_INT8   ; break ;
                case GB_INT16_code   : idxunop = GrB_VALUEGT_INT16  ; break ;
                case GB_INT32_code   : idxunop = GrB_VALUEGT_INT32  ; break ;
                case GB_INT64_code   : idxunop = GrB_VALUEGT_INT64  ; break ;
                case GB_UINT8_code   : idxunop = GrB_VALUEGT_UINT8  ; break ;
                case GB_UINT16_code  : idxunop = GrB_VALUEGT_UINT16 ; break ;
                case GB_UINT32_code  : idxunop = GrB_VALUEGT_UINT32 ; break ;
                case GB_UINT64_code  : idxunop = GrB_VALUEGT_UINT64 ; break ;
                case GB_FP32_code    : idxunop = GrB_VALUEGT_FP32   ; break ;
                case GB_FP64_code    : idxunop = GrB_VALUEGT_FP64   ; break ;
                default:;
            }
            break ;

        case GB_GE_ZERO_selop_code   : Thunk = NULL ;
        case GB_GE_THUNK_selop_code  : 

            switch (acode)
            {
                case GB_BOOL_code    : idxunop = GrB_VALUEGE_BOOL   ; break ;
                case GB_INT8_code    : idxunop = GrB_VALUEGE_INT8   ; break ;
                case GB_INT16_code   : idxunop = GrB_VALUEGE_INT16  ; break ;
                case GB_INT32_code   : idxunop = GrB_VALUEGE_INT32  ; break ;
                case GB_INT64_code   : idxunop = GrB_VALUEGE_INT64  ; break ;
                case GB_UINT8_code   : idxunop = GrB_VALUEGE_UINT8  ; break ;
                case GB_UINT16_code  : idxunop = GrB_VALUEGE_UINT16 ; break ;
                case GB_UINT32_code  : idxunop = GrB_VALUEGE_UINT32 ; break ;
                case GB_UINT64_code  : idxunop = GrB_VALUEGE_UINT64 ; break ;
                case GB_FP32_code    : idxunop = GrB_VALUEGE_FP32   ; break ;
                case GB_FP64_code    : idxunop = GrB_VALUEGE_FP64   ; break ;
                default:;
            }
            break ;

        case GB_LT_ZERO_selop_code   : Thunk = NULL ;
        case GB_LT_THUNK_selop_code  : 

            switch (acode)
            {
                case GB_BOOL_code    : idxunop = GrB_VALUELT_BOOL   ; break ;
                case GB_INT8_code    : idxunop = GrB_VALUELT_INT8   ; break ;
                case GB_INT16_code   : idxunop = GrB_VALUELT_INT16  ; break ;
                case GB_INT32_code   : idxunop = GrB_VALUELT_INT32  ; break ;
                case GB_INT64_code   : idxunop = GrB_VALUELT_INT64  ; break ;
                case GB_UINT8_code   : idxunop = GrB_VALUELT_UINT8  ; break ;
                case GB_UINT16_code  : idxunop = GrB_VALUELT_UINT16 ; break ;
                case GB_UINT32_code  : idxunop = GrB_VALUELT_UINT32 ; break ;
                case GB_UINT64_code  : idxunop = GrB_VALUELT_UINT64 ; break ;
                case GB_FP32_code    : idxunop = GrB_VALUELT_FP32   ; break ;
                case GB_FP64_code    : idxunop = GrB_VALUELT_FP64   ; break ;
                default:;
            }
            break ;

        case GB_LE_ZERO_selop_code   : Thunk = NULL ;
        case GB_LE_THUNK_selop_code  : 

            switch (acode)
            {
                case GB_BOOL_code    : idxunop = GrB_VALUELE_BOOL   ; break ;
                case GB_INT8_code    : idxunop = GrB_VALUELE_INT8   ; break ;
                case GB_INT16_code   : idxunop = GrB_VALUELE_INT16  ; break ;
                case GB_INT32_code   : idxunop = GrB_VALUELE_INT32  ; break ;
                case GB_INT64_code   : idxunop = GrB_VALUELE_INT64  ; break ;
                case GB_UINT8_code   : idxunop = GrB_VALUELE_UINT8  ; break ;
                case GB_UINT16_code  : idxunop = GrB_VALUELE_UINT16 ; break ;
                case GB_UINT32_code  : idxunop = GrB_VALUELE_UINT32 ; break ;
                case GB_UINT64_code  : idxunop = GrB_VALUELE_UINT64 ; break ;
                case GB_FP32_code    : idxunop = GrB_VALUELE_FP32   ; break ;
                case GB_FP64_code    : idxunop = GrB_VALUELE_FP64   ; break ;
                default:;
            }
            break ;

        default: ;
    }

    if (idxunop == NULL)
    { 
        // user-defined GxB_SelectOps and the NONZOMBIE opcode are not supported
        return (GrB_NOT_IMPLEMENTED) ;
    }

    //--------------------------------------------------------------------------
    // create the new Thunk
    //--------------------------------------------------------------------------

    // finish any pending work on the Thunk
    GB_MATRIX_WAIT (Thunk) ;

    // allocate the NewThunk as a full scalar
    GB_OK (GB_new_bix ((GrB_Matrix *) &NewThunk, idxunop->ytype, 1, 1,
        GB_Ap_calloc, true, GxB_FULL, false, GB_Global_hyper_switch_get ( ),
        1, 1, true, false)) ;

    // NewThunk = 0
    memset (NewThunk->x, 0, idxunop->ytype->size) ;
    NewThunk->magic = GB_MAGIC ;

    // copy/typecast Thunk into NewThunk
    if (Thunk != NULL && GB_nnz ((GrB_Matrix) Thunk) == 1)
    {
        if (!GB_Type_compatible (idxunop->ytype, Thunk->type))
        { 
            GB_FREE_ALL ;
            return (GrB_DOMAIN_MISMATCH) ;
        }
        GB_cast_matrix ((GrB_Matrix) NewThunk, (GrB_Matrix) Thunk) ;
    }
    
    ASSERT_MATRIX_OK ((GrB_Matrix) NewThunk, "new thunk", GB0) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*idxunop_handle) = idxunop ;
    (*NewThunk_handle) = NewThunk ;
    return (GrB_SUCCESS) ;
}

