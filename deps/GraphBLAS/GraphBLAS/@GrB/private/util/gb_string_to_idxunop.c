//------------------------------------------------------------------------------
// gb_string_to_idxunop: get a GrB_IndexUnaryOp from a string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

// GrB_IndexUnaryOp operators, with their equivalent aliases

void gb_string_to_idxunop
(
    // outputs: one of the outputs is non-NULL and the other NULL
    GrB_IndexUnaryOp *op,       // GrB_IndexUnaryOp, if found
    bool *thunk_zero,           // true if op requires a thunk zero
    bool *op_is_positional,     // true if op is positional
    // input/output:
    int64_t *ithunk,
    // inputs:
    char *opstring,             // string defining the operator
    const GrB_Type atype        // type of A, or NULL if not present
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (opstring == NULL || opstring [0] == '\0', "invalid op") ;

    //--------------------------------------------------------------------------
    // get the opstring and parse it
    //--------------------------------------------------------------------------

    int32_t position [2] ;
    gb_find_dot (position, opstring) ;

    char *op_name = opstring ;
    char *op_typename = NULL ;
    if (position [0] >= 0)
    { 
        opstring [position [0]] = '\0' ;
        op_typename = opstring + position [0] + 1 ;
    }

    //--------------------------------------------------------------------------
    // get the operator type for VALUE* operators
    //--------------------------------------------------------------------------

    GrB_Type type ;
    if (op_typename == NULL)
    { 
        // no type in the opstring; select the type from A
        type = atype ;
    }
    else
    { 
        // type is explicitly present in the opstring
        type = gb_string_to_type (op_typename) ;
    }

    if (type == NULL)
    {
        // type may still be NULL, which is OK for positional ops since the
        // ignore the type.  But a placeholder type is needed for VALUE ops.
        type = GrB_FP64 ;
    }
    GB_Type_code typecode = type->code ;

    //--------------------------------------------------------------------------
    // convert the string to a GrB_IndexUnaryOp
    //--------------------------------------------------------------------------

    bool is_nonzero = MATCH (opstring, "nonzero") || MATCH (opstring, "~=0") ;
    bool is_zero = MATCH (opstring, "zero") || MATCH (opstring, "==0") ;
    bool is_positive = MATCH (opstring, "positive") || MATCH (opstring, ">0") ;
    bool is_nonneg = MATCH (opstring, "nonnegative") || MATCH (opstring, ">=0");
    bool is_negative = MATCH (opstring, "negative") || MATCH (opstring, "<0") ;
    bool is_nonpos = MATCH (opstring, "nonpositive") || MATCH (opstring, "<=0");
    (*thunk_zero) = (is_nonzero || is_zero || is_positive || is_nonneg
        || is_negative || is_nonpos) ;

    (*op) = NULL ;
    (*op_is_positional) = false ;

    if (MATCH (opstring, "tril"))
    { 
        (*op) = GrB_TRIL ;
        (*op_is_positional) = true ;
    }
    else if (MATCH (opstring, "triu"))
    { 
        (*op) = GrB_TRIU ;
        (*op_is_positional) = true ;
    }
    else if (MATCH (opstring, "diag"))
    { 
        (*op) = GrB_DIAG ;
        (*op_is_positional) = true ;
    }
    else if (MATCH (opstring, "offdiag"))
    { 
        (*op) = GrB_OFFDIAG ;
        (*op_is_positional) = true ;
    }
    else if (MATCH (opstring, "rowne"))
    { 
        (*op) = GrB_ROWINDEX_INT64 ;
        (*ithunk) = - (*ithunk - 1) ;
        (*op_is_positional) = true ;
    }
    else if (MATCH (opstring, "rowle"))
    { 
        (*op) = GrB_ROWLE ;
        (*ithunk)-- ;
        (*op_is_positional) = true ;
    }
    else if (MATCH (opstring, "rowgt"))
    { 
        (*op) = GrB_ROWGT ;
        (*ithunk)-- ;
        (*op_is_positional) = true ;
    }
    else if (MATCH (opstring, "colne"))
    { 
        (*op) = GrB_COLINDEX_INT64 ;
        (*ithunk) = - (*ithunk - 1) ;
        (*op_is_positional) = true ;
    }
    else if (MATCH (opstring, "colle"))
    { 
        (*op) = GrB_COLLE ;
        (*ithunk)-- ;
        (*op_is_positional) = true ;
    }
    else if (MATCH (opstring, "colgt"))
    { 
        (*op) = GrB_COLGT ;
        (*ithunk)-- ;
        (*op_is_positional) = true ;
    }
    else if (MATCH (opstring, "~=") || is_nonzero)
    { 
        switch (typecode)
        {
            case GB_BOOL_code   : (*op) = GrB_VALUENE_BOOL    ; break ;
            case GB_INT8_code   : (*op) = GrB_VALUENE_INT8    ; break ;
            case GB_INT16_code  : (*op) = GrB_VALUENE_INT16   ; break ;
            case GB_INT32_code  : (*op) = GrB_VALUENE_INT32   ; break ;
            case GB_INT64_code  : (*op) = GrB_VALUENE_INT64   ; break ;
            case GB_UINT8_code  : (*op) = GrB_VALUENE_UINT8   ; break ;
            case GB_UINT16_code : (*op) = GrB_VALUENE_UINT16  ; break ;
            case GB_UINT32_code : (*op) = GrB_VALUENE_UINT32  ; break ;
            case GB_UINT64_code : (*op) = GrB_VALUENE_UINT64  ; break ;
            case GB_FP32_code   : (*op) = GrB_VALUENE_FP32    ; break ;
            case GB_FP64_code   : (*op) = GrB_VALUENE_FP64    ; break ;
            case GB_FC32_code   : (*op) = GxB_VALUENE_FC32    ; break ;
            case GB_FC64_code   : (*op) = GxB_VALUENE_FC64    ; break ;
            default             : ;
        }
    }
    else if (MATCH (opstring, "==") || is_zero)
    { 
        switch (typecode)
        {
            case GB_BOOL_code   : (*op) = GrB_VALUEEQ_BOOL    ; break ;
            case GB_INT8_code   : (*op) = GrB_VALUEEQ_INT8    ; break ;
            case GB_INT16_code  : (*op) = GrB_VALUEEQ_INT16   ; break ;
            case GB_INT32_code  : (*op) = GrB_VALUEEQ_INT32   ; break ;
            case GB_INT64_code  : (*op) = GrB_VALUEEQ_INT64   ; break ;
            case GB_UINT8_code  : (*op) = GrB_VALUEEQ_UINT8   ; break ;
            case GB_UINT16_code : (*op) = GrB_VALUEEQ_UINT16  ; break ;
            case GB_UINT32_code : (*op) = GrB_VALUEEQ_UINT32  ; break ;
            case GB_UINT64_code : (*op) = GrB_VALUEEQ_UINT64  ; break ;
            case GB_FP32_code   : (*op) = GrB_VALUEEQ_FP32    ; break ;
            case GB_FP64_code   : (*op) = GrB_VALUEEQ_FP64    ; break ;
            case GB_FC32_code   : (*op) = GxB_VALUEEQ_FC32    ; break ;
            case GB_FC64_code   : (*op) = GxB_VALUEEQ_FC64    ; break ;
            default             : ;
        }
    }
    else if (MATCH (opstring, ">") || is_positive)
    { 
        switch (typecode)
        {
            case GB_BOOL_code   : (*op) = GrB_VALUEGT_BOOL    ; break ;
            case GB_INT8_code   : (*op) = GrB_VALUEGT_INT8    ; break ;
            case GB_INT16_code  : (*op) = GrB_VALUEGT_INT16   ; break ;
            case GB_INT32_code  : (*op) = GrB_VALUEGT_INT32   ; break ;
            case GB_INT64_code  : (*op) = GrB_VALUEGT_INT64   ; break ;
            case GB_UINT8_code  : (*op) = GrB_VALUEGT_UINT8   ; break ;
            case GB_UINT16_code : (*op) = GrB_VALUEGT_UINT16  ; break ;
            case GB_UINT32_code : (*op) = GrB_VALUEGT_UINT32  ; break ;
            case GB_UINT64_code : (*op) = GrB_VALUEGT_UINT64  ; break ;
            case GB_FP32_code   : (*op) = GrB_VALUEGT_FP32    ; break ;
            case GB_FP64_code   : (*op) = GrB_VALUEGT_FP64    ; break ;
            default             : ;
        }
    }
    else if (MATCH (opstring, ">=") || is_nonneg)
    { 
        switch (typecode)
        {
            case GB_BOOL_code   : (*op) = GrB_VALUEGE_BOOL    ; break ;
            case GB_INT8_code   : (*op) = GrB_VALUEGE_INT8    ; break ;
            case GB_INT16_code  : (*op) = GrB_VALUEGE_INT16   ; break ;
            case GB_INT32_code  : (*op) = GrB_VALUEGE_INT32   ; break ;
            case GB_INT64_code  : (*op) = GrB_VALUEGE_INT64   ; break ;
            case GB_UINT8_code  : (*op) = GrB_VALUEGE_UINT8   ; break ;
            case GB_UINT16_code : (*op) = GrB_VALUEGE_UINT16  ; break ;
            case GB_UINT32_code : (*op) = GrB_VALUEGE_UINT32  ; break ;
            case GB_UINT64_code : (*op) = GrB_VALUEGE_UINT64  ; break ;
            case GB_FP32_code   : (*op) = GrB_VALUEGE_FP32    ; break ;
            case GB_FP64_code   : (*op) = GrB_VALUEGE_FP64    ; break ;
            default             : ;
        }
    }
    else if (MATCH (opstring, "<") || is_negative)
    { 
        switch (typecode)
        {
            case GB_BOOL_code   : (*op) = GrB_VALUELT_BOOL    ; break ;
            case GB_INT8_code   : (*op) = GrB_VALUELT_INT8    ; break ;
            case GB_INT16_code  : (*op) = GrB_VALUELT_INT16   ; break ;
            case GB_INT32_code  : (*op) = GrB_VALUELT_INT32   ; break ;
            case GB_INT64_code  : (*op) = GrB_VALUELT_INT64   ; break ;
            case GB_UINT8_code  : (*op) = GrB_VALUELT_UINT8   ; break ;
            case GB_UINT16_code : (*op) = GrB_VALUELT_UINT16  ; break ;
            case GB_UINT32_code : (*op) = GrB_VALUELT_UINT32  ; break ;
            case GB_UINT64_code : (*op) = GrB_VALUELT_UINT64  ; break ;
            case GB_FP32_code   : (*op) = GrB_VALUELT_FP32    ; break ;
            case GB_FP64_code   : (*op) = GrB_VALUELT_FP64    ; break ;
            default             : ;
        }
    }
    else if (MATCH (opstring, "<=") || is_nonpos)
    { 
        switch (typecode)
        {
            case GB_BOOL_code   : (*op) = GrB_VALUELE_BOOL    ; break ;
            case GB_INT8_code   : (*op) = GrB_VALUELE_INT8    ; break ;
            case GB_INT16_code  : (*op) = GrB_VALUELE_INT16   ; break ;
            case GB_INT32_code  : (*op) = GrB_VALUELE_INT32   ; break ;
            case GB_INT64_code  : (*op) = GrB_VALUELE_INT64   ; break ;
            case GB_UINT8_code  : (*op) = GrB_VALUELE_UINT8   ; break ;
            case GB_UINT16_code : (*op) = GrB_VALUELE_UINT16  ; break ;
            case GB_UINT32_code : (*op) = GrB_VALUELE_UINT32  ; break ;
            case GB_UINT64_code : (*op) = GrB_VALUELE_UINT64  ; break ;
            case GB_FP32_code   : (*op) = GrB_VALUELE_FP32    ; break ;
            case GB_FP64_code   : (*op) = GrB_VALUELE_FP64    ; break ;
            default             : ;
        }
    }

    if ((*op) == NULL)
    { 
        ERROR2 ("op unknown: %s\n", opstring) ;
    }
}

