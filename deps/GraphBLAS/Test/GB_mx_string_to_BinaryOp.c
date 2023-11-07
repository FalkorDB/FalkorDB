//------------------------------------------------------------------------------
// GB_mx_string_to_BinaryOp.c: get a GraphBLAS operator from built-in strings
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB_mex.h"

// opname_mx: a built-in string defining the operator name:
// 10: first, second, pair, min, max, plus, minus, rminus, times, div, rdiv
//  6: iseq, isne, isgt, islt, isge, isle,
//  6: eq, ne, gt, lt, ge, le,
//  3: or, and, xor
//  ... and more

// optype_mx: a built-in string defining one of 11 operator types:
//  'logical', 'int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64',
//  'uint64', 'single', 'double', 'single complex', or 'double complex'

// default_optype: default operator type if optype_mx is NULL or if
// the type is not part of the string

bool GB_mx_string_to_BinaryOp       // true if successful, false otherwise
(
    GrB_BinaryOp *op_handle,        // the binary op
    const GrB_Type default_optype,  // default operator type
    const mxArray *opname_mx,       // built-in string with operator name
    const mxArray *optype_mx,       // built-in string with operator type
    const bool user_complex         // if true, use user-defined Complex op
)
{

    (*op_handle) = NULL ;
    GrB_BinaryOp op = NULL ;

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

    bool cmplx_op = MATCH (opname, "cmplx") || MATCH (opname, "complex" ) ;

    // get the optype from the optype_mx string, if present
    GrB_Type optype = GB_mx_string_to_Type (optype_mx, default_optype) ;
    if (optype == NULL)
    {
        mexWarnMsgIdAndTxt ("GB:warn", "unrecognized op type") ;
        return (false) ;
    }

    //--------------------------------------------------------------------------
    // convert the string to a GraphBLAS binary operator, built-in or Complex
    //--------------------------------------------------------------------------

    if (user_complex && (optype == Complex || cmplx_op))
    {

        //----------------------------------------------------------------------
        // user-defined operators for the Complex type
        //----------------------------------------------------------------------

        // 12 binary operators z=f(x,y), all x,y,z are Complex
             if (MATCH (opname, "first"   )) { op = Complex_first  ; }
        else if (MATCH (opname, "second"  )) { op = Complex_second ; }
        else if (MATCH (opname, "pair"    )) { op = Complex_pair   ; }
        else if (MATCH (opname, "oneb"    )) { op = Complex_pair   ; }
        else if (MATCH (opname, "any"     )) { op = Complex_second ; }
        else if (MATCH (opname, "min"     )) { op = Complex_min    ; }
        else if (MATCH (opname, "max"     )) { op = Complex_max    ; }
        else if (MATCH (opname, "plus"    )) { op = Complex_plus   ; }
        else if (MATCH (opname, "minus"   )) { op = Complex_minus  ; }
        else if (MATCH (opname, "rminus"  )) { op = Complex_rminus ; }
        else if (MATCH (opname, "times"   )) { op = Complex_times  ; }
        else if (MATCH (opname, "div"     )) { op = Complex_div    ; }
        else if (MATCH (opname, "rdiv"    )) { op = Complex_rdiv   ; }

        // 6 ops z=f(x,y), where x,y are Complex, z = (1,0) or (0,0)
        else if (MATCH (opname, "iseq"    )) { op = Complex_iseq   ; }
        else if (MATCH (opname, "isne"    )) { op = Complex_isne   ; }
        else if (MATCH (opname, "isgt"    )) { op = Complex_isgt   ; }
        else if (MATCH (opname, "islt"    )) { op = Complex_islt   ; }
        else if (MATCH (opname, "isge"    )) { op = Complex_isge   ; }
        else if (MATCH (opname, "isle"    )) { op = Complex_isle   ; }

        // 3 binary operators z=f(x,y), all x,y,x the same type
        else if (MATCH (opname, "or"      )) { op = Complex_or ; }
        else if (MATCH (opname, "and"     )) { op = Complex_and ; }
        else if (MATCH (opname, "xor"     )) { op = Complex_xor ; }

        // 6 ops z=f(x,y), where x,y are Complex type but z is boolean
        else if (MATCH (opname, "eq"      )) { op = Complex_eq     ; }
        else if (MATCH (opname, "ne"      )) { op = Complex_ne     ; }
        else if (MATCH (opname, "gt"      )) { op = Complex_gt     ; }
        else if (MATCH (opname, "lt"      )) { op = Complex_lt     ; }
        else if (MATCH (opname, "ge"      )) { op = Complex_ge     ; }
        else if (MATCH (opname, "le"      )) { op = Complex_le     ; }

        // z is complex, x and y are real
        else if (cmplx_op) { op = Complex_complex ; }

        else
        {
            mexWarnMsgIdAndTxt ("GB:warn", "Complex op unrecognized") ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // built-in binary operator
        //----------------------------------------------------------------------

        GB_Opcode opcode ; 

        // 12 binary operators z=f(x,y), all x,y,z of the same type
        // (pair and oneb are identical)
             if (MATCH (opname, "first"   )) { opcode = GB_FIRST_binop_code ; }
        else if (MATCH (opname, "second"  )) { opcode = GB_SECOND_binop_code ; }
        else if (MATCH (opname, "pair"    )) { opcode = GB_PAIR_binop_code ; }
        else if (MATCH (opname, "oneb"    )) { opcode = GB_PAIR_binop_code ; }
        else if (MATCH (opname, "any"     )) { opcode = GB_ANY_binop_code ; }
        else if (MATCH (opname, "min"     )) { opcode = GB_MIN_binop_code ; }
        else if (MATCH (opname, "max"     )) { opcode = GB_MAX_binop_code ; }
        else if (MATCH (opname, "plus"    )) { opcode = GB_PLUS_binop_code ; }
        else if (MATCH (opname, "minus"   )) { opcode = GB_MINUS_binop_code ; }
        else if (MATCH (opname, "rminus"  )) { opcode = GB_RMINUS_binop_code ; }
        else if (MATCH (opname, "times"   )) { opcode = GB_TIMES_binop_code ; }
        else if (MATCH (opname, "div"     )) { opcode = GB_DIV_binop_code ; }
        else if (MATCH (opname, "rdiv"    )) { opcode = GB_RDIV_binop_code ; }

        // 6 ops z=f(x,y), all x,y,z the same type
        else if (MATCH (opname, "iseq"    )) { opcode = GB_ISEQ_binop_code ; }
        else if (MATCH (opname, "isne"    )) { opcode = GB_ISNE_binop_code ; }
        else if (MATCH (opname, "isgt"    )) { opcode = GB_ISGT_binop_code ; }
        else if (MATCH (opname, "islt"    )) { opcode = GB_ISLT_binop_code ; }
        else if (MATCH (opname, "isge"    )) { opcode = GB_ISGE_binop_code ; }
        else if (MATCH (opname, "isle"    )) { opcode = GB_ISLE_binop_code ; }

        // 3 binary operators z=f(x,y), all x,y,x the same type
        else if (MATCH (opname, "or"      ) ||
                 MATCH (opname, "lor"     )) { opcode = GB_LOR_binop_code ; }
        else if (MATCH (opname, "and"     ) ||
                 MATCH (opname, "land"    )) { opcode = GB_LAND_binop_code ; }
        else if (MATCH (opname, "xor"     ) ||
                 MATCH (opname, "lxor"    )) { opcode = GB_LXOR_binop_code ; }

        // 6 ops z=f(x,y), where x,y are the requested type but z is boolean
        else if (MATCH (opname, "eq"      )) { opcode = GB_EQ_binop_code ; }
        else if (MATCH (opname, "ne"      )) { opcode = GB_NE_binop_code ; }
        else if (MATCH (opname, "gt"      )) { opcode = GB_GT_binop_code ; }
        else if (MATCH (opname, "lt"      )) { opcode = GB_LT_binop_code ; }
        else if (MATCH (opname, "ge"      )) { opcode = GB_GE_binop_code ; }
        else if (MATCH (opname, "le"      )) { opcode = GB_LE_binop_code ; }

        else if (MATCH (opname, "atan2"   )) { opcode = GB_ATAN2_binop_code ; }
        else if (MATCH (opname, "hypot"   )) { opcode = GB_HYPOT_binop_code ; }
        else if (MATCH (opname, "fmod"    )) { opcode = GB_FMOD_binop_code ; }
        else if (MATCH (opname,"remainder")) { opcode = GB_REMAINDER_binop_code ; }
        else if (MATCH (opname, "copysign")) { opcode = GB_COPYSIGN_binop_code ; }
        else if (MATCH (opname, "ldexp"   )) { opcode = GB_LDEXP_binop_code ; }
        else if (MATCH (opname, "pow"     )) { opcode = GB_POW_binop_code ; }

        // positional ops
        else if (MATCH (opname, "firsti"  ) ||
                 MATCH (opname, "1sti"    )) { opcode = GB_FIRSTI_binop_code ; }
        else if (MATCH (opname, "firsti1" ) ||
                 MATCH (opname, "1sti1"   )) { opcode = GB_FIRSTI1_binop_code ; }
        else if (MATCH (opname, "firstj"  ) ||
                 MATCH (opname, "1stj"    )) { opcode = GB_FIRSTJ_binop_code ; }
        else if (MATCH (opname, "firstj1" ) ||
                 MATCH (opname, "1stj1"   )) { opcode = GB_FIRSTJ1_binop_code ; }
        else if (MATCH (opname, "secondi" ) ||
                 MATCH (opname, "2ndi"    )) { opcode = GB_SECONDI_binop_code ; }
        else if (MATCH (opname, "secondi1") ||
                 MATCH (opname, "2ndi1"   )) { opcode = GB_SECONDI1_binop_code ; }
        else if (MATCH (opname, "secondj" ) ||
                 MATCH (opname, "2ndj"    )) { opcode = GB_SECONDJ_binop_code ; }
        else if (MATCH (opname, "secondj1") ||
                 MATCH (opname, "2ndj1"   )) { opcode = GB_SECONDJ1_binop_code ; }

        // z is complex, x and y are real
        else if (cmplx_op                  ) { opcode = GB_CMPLX_binop_code ; }

        // bitwise operators
        else if (MATCH (opname, "bitor"   ) ||
                 MATCH (opname, "bor"     )) { opcode = GB_BOR_binop_code ; }
        else if (MATCH (opname, "bitand"  ) ||
                 MATCH (opname, "band"    )) { opcode = GB_BAND_binop_code ; }
        else if (MATCH (opname, "bitxor"  ) ||
                 MATCH (opname, "bxor"    )) { opcode = GB_BXOR_binop_code ; }
        else if (MATCH (opname, "bitxnor" ) ||
                 MATCH (opname, "bxnor"   )) { opcode = GB_BXNOR_binop_code ; }
        else if (MATCH (opname, "bitget"  ) ||
                 MATCH (opname, "bget"    )) { opcode = GB_BGET_binop_code ; }
        else if (MATCH (opname, "bitset"  ) ||
                 MATCH (opname, "bset"    )) { opcode = GB_BSET_binop_code ; }
        else if (MATCH (opname, "bitclr"  ) ||
                 MATCH (opname, "bclr"    )) { opcode = GB_BCLR_binop_code ; }
        else if (MATCH (opname, "bitshift") ||
                 MATCH (opname, "bshift"  )) { opcode = GB_BSHIFT_binop_code ; }

        else
        {
            mexWarnMsgIdAndTxt ("GB:warn", "unrecognized function name") ;
            return (false) ;
        }

        GB_Type_code xcode = optype->code ;
        bool is64 = (xcode == GB_INT64_code) ;

        if (GB_OPCODE_IS_POSITIONAL (opcode))
        {
            if (! (xcode == GB_INT64_code || xcode == GB_INT32_code))
            {
                mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                return (false) ;
            }
        }

        switch (opcode)
        {

            case GB_FIRST_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_FIRST_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_FIRST_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_FIRST_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_FIRST_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_FIRST_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_FIRST_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_FIRST_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_FIRST_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_FIRST_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_FIRST_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_FIRST_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_FIRST_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_FIRST_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_SECOND_binop_code:

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_SECOND_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_SECOND_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_SECOND_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_SECOND_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_SECOND_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_SECOND_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_SECOND_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_SECOND_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_SECOND_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_SECOND_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_SECOND_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_SECOND_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_SECOND_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ANY_binop_code:

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GxB_ANY_BOOL   ; break ;
                    case GB_INT8_code    : op = GxB_ANY_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_ANY_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_ANY_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_ANY_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_ANY_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_ANY_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_ANY_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_ANY_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_ANY_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ANY_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ANY_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ANY_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_PAIR_binop_code:

                switch (xcode)
                {
                    // GrB_ONEB_T is the new name for GxB_PAIR_T
                    case GB_BOOL_code    : op = GrB_ONEB_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_ONEB_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_ONEB_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_ONEB_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_ONEB_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_ONEB_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_ONEB_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_ONEB_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_ONEB_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_ONEB_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_ONEB_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ONEB_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ONEB_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_MIN_binop_code   :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_MIN_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_MIN_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_MIN_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_MIN_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_MIN_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_MIN_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_MIN_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_MIN_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_MIN_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_MIN_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_MIN_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        break ;
                }
                break ;

            case GB_MAX_binop_code   :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_MAX_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_MAX_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_MAX_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_MAX_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_MAX_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_MAX_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_MAX_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_MAX_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_MAX_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_MAX_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_MAX_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_PLUS_binop_code  :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_PLUS_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_PLUS_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_PLUS_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_PLUS_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_PLUS_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_PLUS_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_PLUS_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_PLUS_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_PLUS_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_PLUS_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_PLUS_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_PLUS_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_PLUS_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_MINUS_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_MINUS_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_MINUS_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_MINUS_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_MINUS_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_MINUS_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_MINUS_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_MINUS_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_MINUS_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_MINUS_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_MINUS_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_MINUS_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_MINUS_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_MINUS_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_RMINUS_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GxB_RMINUS_BOOL   ; break ;
                    case GB_INT8_code    : op = GxB_RMINUS_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_RMINUS_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_RMINUS_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_RMINUS_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_RMINUS_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_RMINUS_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_RMINUS_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_RMINUS_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_RMINUS_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_RMINUS_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_RMINUS_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_RMINUS_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_TIMES_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_TIMES_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_TIMES_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_TIMES_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_TIMES_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_TIMES_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_TIMES_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_TIMES_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_TIMES_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_TIMES_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_TIMES_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_TIMES_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_TIMES_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_TIMES_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_DIV_binop_code   :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_DIV_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_DIV_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_DIV_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_DIV_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_DIV_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_DIV_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_DIV_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_DIV_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_DIV_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_DIV_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_DIV_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_DIV_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_DIV_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_RDIV_binop_code   :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GxB_RDIV_BOOL   ; break ;
                    case GB_INT8_code    : op = GxB_RDIV_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_RDIV_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_RDIV_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_RDIV_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_RDIV_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_RDIV_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_RDIV_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_RDIV_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_RDIV_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_RDIV_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_RDIV_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_RDIV_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_POW_binop_code :     // z = pow (x,y)

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GxB_POW_BOOL   ; break ;
                    case GB_INT8_code    : op = GxB_POW_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_POW_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_POW_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_POW_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_POW_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_POW_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_POW_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_POW_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_POW_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_POW_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_POW_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_POW_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ISEQ_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GxB_ISEQ_BOOL   ; break ;
                    case GB_INT8_code    : op = GxB_ISEQ_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_ISEQ_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_ISEQ_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_ISEQ_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_ISEQ_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_ISEQ_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_ISEQ_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_ISEQ_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_ISEQ_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ISEQ_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ISEQ_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ISEQ_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ISNE_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GxB_ISNE_BOOL   ; break ;
                    case GB_INT8_code    : op = GxB_ISNE_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_ISNE_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_ISNE_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_ISNE_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_ISNE_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_ISNE_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_ISNE_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_ISNE_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_ISNE_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ISNE_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_ISNE_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_ISNE_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ISGT_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GxB_ISGT_BOOL   ; break ;
                    case GB_INT8_code    : op = GxB_ISGT_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_ISGT_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_ISGT_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_ISGT_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_ISGT_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_ISGT_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_ISGT_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_ISGT_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_ISGT_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ISGT_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ISLT_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GxB_ISLT_BOOL   ; break ;
                    case GB_INT8_code    : op = GxB_ISLT_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_ISLT_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_ISLT_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_ISLT_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_ISLT_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_ISLT_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_ISLT_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_ISLT_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_ISLT_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ISLT_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ISGE_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GxB_ISGE_BOOL   ; break ;
                    case GB_INT8_code    : op = GxB_ISGE_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_ISGE_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_ISGE_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_ISGE_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_ISGE_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_ISGE_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_ISGE_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_ISGE_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_ISGE_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ISGE_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ISLE_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GxB_ISLE_BOOL   ; break ;
                    case GB_INT8_code    : op = GxB_ISLE_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_ISLE_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_ISLE_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_ISLE_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_ISLE_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_ISLE_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_ISLE_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_ISLE_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_ISLE_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ISLE_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;


            case GB_EQ_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_EQ_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_EQ_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_EQ_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_EQ_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_EQ_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_EQ_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_EQ_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_EQ_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_EQ_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_EQ_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_EQ_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_EQ_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_EQ_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_NE_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_NE_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_NE_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_NE_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_NE_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_NE_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_NE_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_NE_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_NE_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_NE_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_NE_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_NE_FP64   ; break ;
                    case GB_FC32_code    : op = GxB_NE_FC32   ; break ;
                    case GB_FC64_code    : op = GxB_NE_FC64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_GT_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_GT_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_GT_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_GT_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_GT_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_GT_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_GT_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_GT_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_GT_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_GT_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_GT_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_GT_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_LT_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_LT_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_LT_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_LT_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_LT_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_LT_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_LT_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_LT_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_LT_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_LT_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_LT_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_LT_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_GE_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_GE_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_GE_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_GE_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_GE_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_GE_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_GE_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_GE_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_GE_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_GE_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_GE_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_GE_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_LE_binop_code :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_LE_BOOL   ; break ;
                    case GB_INT8_code    : op = GrB_LE_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_LE_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_LE_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_LE_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_LE_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_LE_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_LE_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_LE_UINT64 ; break ;
                    case GB_FP32_code    : op = GrB_LE_FP32   ; break ;
                    case GB_FP64_code    : op = GrB_LE_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;


            case GB_LOR_binop_code   :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_LOR        ; break ;
                    case GB_INT8_code    : op = GxB_LOR_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_LOR_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_LOR_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_LOR_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_LOR_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_LOR_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_LOR_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_LOR_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_LOR_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_LOR_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_LAND_binop_code   :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_LAND        ; break ;
                    case GB_INT8_code    : op = GxB_LAND_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_LAND_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_LAND_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_LAND_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_LAND_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_LAND_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_LAND_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_LAND_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_LAND_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_LAND_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_LXOR_binop_code   :

                switch (xcode)
                {
                    case GB_BOOL_code    : op = GrB_LXOR        ; break ;
                    case GB_INT8_code    : op = GxB_LXOR_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_LXOR_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_LXOR_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_LXOR_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_LXOR_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_LXOR_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_LXOR_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_LXOR_UINT64 ; break ;
                    case GB_FP32_code    : op = GxB_LXOR_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_LXOR_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_ATAN2_binop_code :       // z = atan2 (x,y)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_ATAN2_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_ATAN2_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_HYPOT_binop_code :       // z = hypot (x,y)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_HYPOT_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_HYPOT_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_FMOD_binop_code :        // z = fmod (x,y)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_FMOD_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_FMOD_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_REMAINDER_binop_code :   // z = remainder (x,y)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_REMAINDER_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_REMAINDER_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_COPYSIGN_binop_code :    // z = copysign (x,y)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_COPYSIGN_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_COPYSIGN_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_LDEXP_binop_code :       // z = ldexp (x,y)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_LDEXP_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_LDEXP_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_CMPLX_binop_code :       // z = cmplx (x,y)

                switch (xcode)
                {
                    case GB_FP32_code    : op = GxB_CMPLX_FP32   ; break ;
                    case GB_FP64_code    : op = GxB_CMPLX_FP64   ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_BOR_binop_code :     // z = (x | y), bitwise or

                switch (xcode)
                {
                    case GB_INT8_code    : op = GrB_BOR_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_BOR_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_BOR_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_BOR_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_BOR_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_BOR_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_BOR_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_BOR_UINT64 ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_BAND_binop_code :    // z = (x & y), bitwise and

                switch (xcode)
                {
                    case GB_INT8_code    : op = GrB_BAND_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_BAND_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_BAND_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_BAND_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_BAND_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_BAND_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_BAND_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_BAND_UINT64 ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_BXOR_binop_code :    // z = (x ^ y), bitwise xor

                switch (xcode)
                {
                    case GB_INT8_code    : op = GrB_BXOR_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_BXOR_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_BXOR_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_BXOR_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_BXOR_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_BXOR_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_BXOR_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_BXOR_UINT64 ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_BXNOR_binop_code :   // z = ~(x ^ y), bitwise xnor

                switch (xcode)
                {
                    case GB_INT8_code    : op = GrB_BXNOR_INT8   ; break ;
                    case GB_INT16_code   : op = GrB_BXNOR_INT16  ; break ;
                    case GB_INT32_code   : op = GrB_BXNOR_INT32  ; break ;
                    case GB_INT64_code   : op = GrB_BXNOR_INT64  ; break ;
                    case GB_UINT8_code   : op = GrB_BXNOR_UINT8  ; break ;
                    case GB_UINT16_code  : op = GrB_BXNOR_UINT16 ; break ;
                    case GB_UINT32_code  : op = GrB_BXNOR_UINT32 ; break ;
                    case GB_UINT64_code  : op = GrB_BXNOR_UINT64 ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_BGET_binop_code :    // z = bitget (x,y)

                switch (xcode)
                {
                    case GB_INT8_code    : op = GxB_BGET_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_BGET_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_BGET_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_BGET_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_BGET_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_BGET_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_BGET_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_BGET_UINT64 ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_BSET_binop_code :    // z = bitset (x,y)

                switch (xcode)
                {
                    case GB_INT8_code    : op = GxB_BSET_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_BSET_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_BSET_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_BSET_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_BSET_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_BSET_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_BSET_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_BSET_UINT64 ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_BCLR_binop_code :    // z = bitclr (x,y)

                switch (xcode)
                {
                    case GB_INT8_code    : op = GxB_BCLR_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_BCLR_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_BCLR_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_BCLR_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_BCLR_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_BCLR_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_BCLR_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_BCLR_UINT64 ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_BSHIFT_binop_code :  // z = bitshift (x,y)

                switch (xcode)
                {
                    case GB_INT8_code    : op = GxB_BSHIFT_INT8   ; break ;
                    case GB_INT16_code   : op = GxB_BSHIFT_INT16  ; break ;
                    case GB_INT32_code   : op = GxB_BSHIFT_INT32  ; break ;
                    case GB_INT64_code   : op = GxB_BSHIFT_INT64  ; break ;
                    case GB_UINT8_code   : op = GxB_BSHIFT_UINT8  ; break ;
                    case GB_UINT16_code  : op = GxB_BSHIFT_UINT16 ; break ;
                    case GB_UINT32_code  : op = GxB_BSHIFT_UINT32 ; break ;
                    case GB_UINT64_code  : op = GxB_BSHIFT_UINT64 ; break ;
                    default              : 
                        mexWarnMsgIdAndTxt ("GB:warn","unknown operator") ;
                        return (false) ;
                }
                break ;

            case GB_FIRSTI_binop_code   : op = is64 ? GxB_FIRSTI_INT64   : GxB_FIRSTI_INT32   ; break ;
            case GB_FIRSTI1_binop_code  : op = is64 ? GxB_FIRSTI1_INT64  : GxB_FIRSTI1_INT32  ; break ;
            case GB_FIRSTJ_binop_code   : op = is64 ? GxB_FIRSTJ_INT64   : GxB_FIRSTJ_INT32   ; break ;
            case GB_FIRSTJ1_binop_code  : op = is64 ? GxB_FIRSTJ1_INT64  : GxB_FIRSTJ1_INT32  ; break ;
            case GB_SECONDI_binop_code  : op = is64 ? GxB_SECONDI_INT64  : GxB_SECONDI_INT32  ; break ;
            case GB_SECONDI1_binop_code : op = is64 ? GxB_SECONDI1_INT64 : GxB_SECONDI1_INT32 ; break ;
            case GB_SECONDJ_binop_code  : op = is64 ? GxB_SECONDJ_INT64  : GxB_SECONDJ_INT32  ; break ;
            case GB_SECONDJ1_binop_code : op = is64 ? GxB_SECONDJ1_INT64 : GxB_SECONDJ1_INT32 ; break ;

            default : 
                mexWarnMsgIdAndTxt ("GB:warn","unknown binary operator") ;
                return (false) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    // return the binary operator to the caller
    ASSERT_BINARYOP_OK_OR_NULL (op, "got binary op", GB0) ;
    (*op_handle) = op ;
    return (true) ;
}

