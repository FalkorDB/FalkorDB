//------------------------------------------------------------------------------
// GB_mx_string_to_IndexUnaryOp.c: get GraphBLAS operator from built-in strings
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

// opname_mx: a built-in string defining the operator name:
// rowindex, colindex, diagindex, tril, triu, diag, offdiag, colle, colgt,
// rowle, rowgt, valueeq, valuene, valuelt, valuele, valuegt, valuege

// optype_mx: a built-in string defining one of 11 operator types:
//  'logical', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64',
//  'uint64', 'single', 'double', 'single complex', or 'double complex'

// default_optype: default operator type if optype_mx is NULL or if
// the type is not part of the string

bool GB_mx_string_to_IndexUnaryOp       // true if successful, false otherwise
(
    GrB_IndexUnaryOp *op_handle,    // the op
    const GrB_Type default_optype,  // default operator type
    const mxArray *opname_mx,       // built-in string with operator name
    const mxArray *optype_mx        // built-in string with operator type
)
{

    (*op_handle) = NULL ;
    GrB_IndexUnaryOp op = NULL ;

    //--------------------------------------------------------------------------
    // get the string
    //--------------------------------------------------------------------------

    #define LEN 256
    char opname [LEN+2] ;
    int len = GB_mx_mxArray_to_string (opname, LEN, opname_mx) ;
    if (len < 0)
    {
        return (false) ;
    }
    if (len == 0)
    {
        op = NULL ;                 // op is not present
        return (true) ;
    }

    // get the optype from the optype_mx string, if present
    GrB_Type optype = GB_mx_string_to_Type (optype_mx, default_optype) ;
    if (optype == NULL)
    {
        mexWarnMsgIdAndTxt ("GB:warn", "unrecognized op type") ;
        return (false) ;
    }

    if (optype == Complex) optype = GxB_FC64 ;

    //--------------------------------------------------------------------------
    // convert the string to a builtin GraphBLAS idxunop operator
    //--------------------------------------------------------------------------

    GB_Opcode opcode ; 

// rowindex, colindex, diagindex, tril, triu, diag, offdiag, colle, colgt,
// rowle, rowgt, valueeq, valuene, valuelt, valuele, valuegt, valuege

         if (MATCH (opname, "rowindex" )) { opcode = GB_ROWINDEX_idxunop_code ; }
    else if (MATCH (opname, "colindex" )) { opcode = GB_COLINDEX_idxunop_code ; }
    else if (MATCH (opname, "diagindex")) { opcode = GB_DIAGINDEX_idxunop_code; }
    else if (MATCH (opname, "tril"     )) { opcode = GB_TRIL_idxunop_code ; }
    else if (MATCH (opname, "triu"     )) { opcode = GB_TRIU_idxunop_code ; }
    else if (MATCH (opname, "diag"     )) { opcode = GB_DIAG_idxunop_code ; }
    else if (MATCH (opname, "offdiag"  )) { opcode = GB_OFFDIAG_idxunop_code ; }
    else if (MATCH (opname, "colle"    )) { opcode = GB_COLLE_idxunop_code ; }
    else if (MATCH (opname, "colgt"    )) { opcode = GB_COLGT_idxunop_code ; }
    else if (MATCH (opname, "rowle"    )) { opcode = GB_ROWLE_idxunop_code ; }
    else if (MATCH (opname, "rowgt"    )) { opcode = GB_ROWGT_idxunop_code ; }
    else if (MATCH (opname, "valueeq"  )) { opcode = GB_VALUEEQ_idxunop_code ; }
    else if (MATCH (opname, "valuene"  )) { opcode = GB_VALUENE_idxunop_code ; }
    else if (MATCH (opname, "valuegt"  )) { opcode = GB_VALUEGT_idxunop_code ; }
    else if (MATCH (opname, "valuege"  )) { opcode = GB_VALUEGE_idxunop_code ; }
    else if (MATCH (opname, "valuelt"  )) { opcode = GB_VALUELT_idxunop_code ; }
    else if (MATCH (opname, "valuele"  )) { opcode = GB_VALUELE_idxunop_code ; }
    else
    {
        mexWarnMsgIdAndTxt ("GB:warn", "unrecognized function name") ;
        return (false) ;
    }

    GB_Type_code xcode = optype->code ;

    switch (opcode)
    {

        // Result is INT32 or INT64, depending on i and/or j, and thunk:
        case GB_ROWINDEX_idxunop_code  : // (i+thunk): row index - thunk

            switch (xcode)
            {
                case GB_INT32_code   : op = GrB_ROWINDEX_INT32 ; break ;
                case GB_INT64_code   : op = GrB_ROWINDEX_INT64 ; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_COLINDEX_idxunop_code  : // (j+thunk): col index - thunk

            switch (xcode)
            {
                case GB_INT32_code   : op = GrB_COLINDEX_INT32 ; break ;
                case GB_INT64_code   : op = GrB_COLINDEX_INT64 ; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_DIAGINDEX_idxunop_code : // (j-(i+thunk)): diag index + thunk

            switch (xcode)
            {
                case GB_INT32_code   : op = GrB_DIAGINDEX_INT32 ; break ;
                case GB_INT64_code   : op = GrB_DIAGINDEX_INT64 ; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        // Result is BOOL, depending on i and/or j, and thunk:
        case GB_TRIL_idxunop_code      : // (j <= (i+thunk)): tril (A,thunk)

            switch (xcode)
            {
                case GB_INT64_code   : op = GrB_TRIL ; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_TRIU_idxunop_code      : // (j >= (i+thunk)): triu (A,thunk)

            switch (xcode)
            {
                case GB_INT64_code   : op = GrB_TRIU ; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_DIAG_idxunop_code      : // (j == (i+thunk)): diag(A,thunk)

            switch (xcode)
            {
                case GB_INT64_code   : op = GrB_DIAG; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_OFFDIAG_idxunop_code   : // (j != (i+thunk)): offdiag(A,thunk)

            switch (xcode)
            {
                case GB_INT64_code   : op = GrB_OFFDIAG; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_COLLE_idxunop_code     : // (j <= thunk): A (:,0:thunk)

            switch (xcode)
            {
                case GB_INT64_code   : op = GrB_COLLE; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_COLGT_idxunop_code     : // (j > thunk): A (:,thunk+1:ncols-1)

            switch (xcode)
            {
                case GB_INT64_code   : op = GrB_COLGT; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_ROWLE_idxunop_code     : // (i <= thunk): A (0:thunk,:)

            switch (xcode)
            {
                case GB_INT64_code   : op = GrB_ROWLE; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_ROWGT_idxunop_code     : // (i > thunk): A (thunk+1:nrows-1,:)

            switch (xcode)
            {
                case GB_INT64_code   : op = GrB_ROWGT; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        // Result is BOOL, depending on the value aij and thunk:
        case GB_VALUENE_idxunop_code   : // (aij != thunk)

            switch (xcode)
            {
                case GB_BOOL_code    : op = GrB_VALUENE_BOOL   ; break ;
                case GB_INT8_code    : op = GrB_VALUENE_INT8   ; break ;
                case GB_INT16_code   : op = GrB_VALUENE_INT16  ; break ;
                case GB_INT32_code   : op = GrB_VALUENE_INT32  ; break ;
                case GB_INT64_code   : op = GrB_VALUENE_INT64  ; break ;
                case GB_UINT8_code   : op = GrB_VALUENE_UINT8  ; break ;
                case GB_UINT16_code  : op = GrB_VALUENE_UINT16 ; break ;
                case GB_UINT32_code  : op = GrB_VALUENE_UINT32 ; break ;
                case GB_UINT64_code  : op = GrB_VALUENE_UINT64 ; break ;
                case GB_FP32_code    : op = GrB_VALUENE_FP32   ; break ;
                case GB_FP64_code    : op = GrB_VALUENE_FP64   ; break ;
                case GB_FC32_code    : op = GxB_VALUENE_FC32   ; break ;
                case GB_FC64_code    : op = GxB_VALUENE_FC64   ; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_VALUEEQ_idxunop_code   : // (aij == thunk)

            switch (xcode)
            {
                case GB_BOOL_code    : op = GrB_VALUEEQ_BOOL   ; break ;
                case GB_INT8_code    : op = GrB_VALUEEQ_INT8   ; break ;
                case GB_INT16_code   : op = GrB_VALUEEQ_INT16  ; break ;
                case GB_INT32_code   : op = GrB_VALUEEQ_INT32  ; break ;
                case GB_INT64_code   : op = GrB_VALUEEQ_INT64  ; break ;
                case GB_UINT8_code   : op = GrB_VALUEEQ_UINT8  ; break ;
                case GB_UINT16_code  : op = GrB_VALUEEQ_UINT16 ; break ;
                case GB_UINT32_code  : op = GrB_VALUEEQ_UINT32 ; break ;
                case GB_UINT64_code  : op = GrB_VALUEEQ_UINT64 ; break ;
                case GB_FP32_code    : op = GrB_VALUEEQ_FP32   ; break ;
                case GB_FP64_code    : op = GrB_VALUEEQ_FP64   ; break ;
                case GB_FC32_code    : op = GxB_VALUEEQ_FC32   ; break ;
                case GB_FC64_code    : op = GxB_VALUEEQ_FC64   ; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_VALUEGT_idxunop_code   : // (aij > thunk)

            switch (xcode)
            {
                case GB_BOOL_code    : op = GrB_VALUEGT_BOOL   ; break ;
                case GB_INT8_code    : op = GrB_VALUEGT_INT8   ; break ;
                case GB_INT16_code   : op = GrB_VALUEGT_INT16  ; break ;
                case GB_INT32_code   : op = GrB_VALUEGT_INT32  ; break ;
                case GB_INT64_code   : op = GrB_VALUEGT_INT64  ; break ;
                case GB_UINT8_code   : op = GrB_VALUEGT_UINT8  ; break ;
                case GB_UINT16_code  : op = GrB_VALUEGT_UINT16 ; break ;
                case GB_UINT32_code  : op = GrB_VALUEGT_UINT32 ; break ;
                case GB_UINT64_code  : op = GrB_VALUEGT_UINT64 ; break ;
                case GB_FP32_code    : op = GrB_VALUEGT_FP32   ; break ;
                case GB_FP64_code    : op = GrB_VALUEGT_FP64   ; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_VALUEGE_idxunop_code   : // (aij >= thunk)

            switch (xcode)
            {
                case GB_BOOL_code    : op = GrB_VALUEGE_BOOL   ; break ;
                case GB_INT8_code    : op = GrB_VALUEGE_INT8   ; break ;
                case GB_INT16_code   : op = GrB_VALUEGE_INT16  ; break ;
                case GB_INT32_code   : op = GrB_VALUEGE_INT32  ; break ;
                case GB_INT64_code   : op = GrB_VALUEGE_INT64  ; break ;
                case GB_UINT8_code   : op = GrB_VALUEGE_UINT8  ; break ;
                case GB_UINT16_code  : op = GrB_VALUEGE_UINT16 ; break ;
                case GB_UINT32_code  : op = GrB_VALUEGE_UINT32 ; break ;
                case GB_UINT64_code  : op = GrB_VALUEGE_UINT64 ; break ;
                case GB_FP32_code    : op = GrB_VALUEGE_FP32   ; break ;
                case GB_FP64_code    : op = GrB_VALUEGE_FP64   ; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_VALUELT_idxunop_code   : // (aij < thunk)

            switch (xcode)
            {
                case GB_BOOL_code    : op = GrB_VALUELT_BOOL   ; break ;
                case GB_INT8_code    : op = GrB_VALUELT_INT8   ; break ;
                case GB_INT16_code   : op = GrB_VALUELT_INT16  ; break ;
                case GB_INT32_code   : op = GrB_VALUELT_INT32  ; break ;
                case GB_INT64_code   : op = GrB_VALUELT_INT64  ; break ;
                case GB_UINT8_code   : op = GrB_VALUELT_UINT8  ; break ;
                case GB_UINT16_code  : op = GrB_VALUELT_UINT16 ; break ;
                case GB_UINT32_code  : op = GrB_VALUELT_UINT32 ; break ;
                case GB_UINT64_code  : op = GrB_VALUELT_UINT64 ; break ;
                case GB_FP32_code    : op = GrB_VALUELT_FP32   ; break ;
                case GB_FP64_code    : op = GrB_VALUELT_FP64   ; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

        case GB_VALUELE_idxunop_code   : // (aij <= thunk)

            switch (xcode)
            {
                case GB_BOOL_code    : op = GrB_VALUELE_BOOL   ; break ;
                case GB_INT8_code    : op = GrB_VALUELE_INT8   ; break ;
                case GB_INT16_code   : op = GrB_VALUELE_INT16  ; break ;
                case GB_INT32_code   : op = GrB_VALUELE_INT32  ; break ;
                case GB_INT64_code   : op = GrB_VALUELE_INT64  ; break ;
                case GB_UINT8_code   : op = GrB_VALUELE_UINT8  ; break ;
                case GB_UINT16_code  : op = GrB_VALUELE_UINT16 ; break ;
                case GB_UINT32_code  : op = GrB_VALUELE_UINT32 ; break ;
                case GB_UINT64_code  : op = GrB_VALUELE_UINT64 ; break ;
                case GB_FP32_code    : op = GrB_VALUELE_FP32   ; break ;
                case GB_FP64_code    : op = GrB_VALUELE_FP64   ; break ;
                default              : 
                    mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                    return (false) ;
            }
            break ;

            default : 
                mexWarnMsgIdAndTxt ("GB:warn","unknown idxunop operator") ;
                return (false) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // return the idxunop operator to the caller
    ASSERT_INDEXUNARYOP_OK (op, "got idxunop op", GB0) ;
    (*op_handle) = op ;
    return (true) ;
}

