//------------------------------------------------------------------------------
// GB_select_entry_factory: switch factory for C=select(A,thunk)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This factory is used for phase1 and phase2 for the sparse case.  None of
// these workers are used when A is iso.  C is always iso for the VALUEEQ
// selector.

ASSERT (!GB_IS_INDEXUNARYOP_CODE_POSITIONAL (opcode)) ;

switch (opcode)
{

    //--------------------------------------------------------------------------
    // nonzombie selectors: phase2 only; not for the bitmap case
    //--------------------------------------------------------------------------

    #if defined ( GB_SELECT_PHASE2 )

    case GB_NONZOMBIE_idxunop_code     :  // C = all entries A(i,j) not a zombie

        switch (acode)
        {
            case GB_BOOL_code   : GB_SEL_WORKER (_nonzombie, _bool  )
            case GB_INT8_code   : GB_SEL_WORKER (_nonzombie, _int8  )
            case GB_INT16_code  : GB_SEL_WORKER (_nonzombie, _int16 )
            case GB_INT32_code  : GB_SEL_WORKER (_nonzombie, _int32 )
            case GB_INT64_code  : GB_SEL_WORKER (_nonzombie, _int64 )
            case GB_UINT8_code  : GB_SEL_WORKER (_nonzombie, _uint8 )
            case GB_UINT16_code : GB_SEL_WORKER (_nonzombie, _uint16)
            case GB_UINT32_code : GB_SEL_WORKER (_nonzombie, _uint32)
            case GB_UINT64_code : GB_SEL_WORKER (_nonzombie, _uint64)
            case GB_FP32_code   : GB_SEL_WORKER (_nonzombie, _fp32  )
            case GB_FP64_code   : GB_SEL_WORKER (_nonzombie, _fp64  )
            case GB_FC32_code   : GB_SEL_WORKER (_nonzombie, _fc32  )
            case GB_FC64_code   : GB_SEL_WORKER (_nonzombie, _fc64  )
            default: ;
        }
        break ;

    #endif

    //--------------------------------------------------------------------------
    // none of these selectop workers are needed when A is iso
    //--------------------------------------------------------------------------

    case GB_VALUEEQ_idxunop_code : // A(i,j) == thunk (so C is iso)

        switch (acode)
        {
            case GB_BOOL_code   : GB_SEL_WORKER (_eq_thunk, _bool  )
            case GB_INT8_code   : GB_SEL_WORKER (_eq_thunk, _int8  )
            case GB_INT16_code  : GB_SEL_WORKER (_eq_thunk, _int16 )
            case GB_INT32_code  : GB_SEL_WORKER (_eq_thunk, _int32 )
            case GB_INT64_code  : GB_SEL_WORKER (_eq_thunk, _int64 )
            case GB_UINT8_code  : GB_SEL_WORKER (_eq_thunk, _uint8 )
            case GB_UINT16_code : GB_SEL_WORKER (_eq_thunk, _uint16)
            case GB_UINT32_code : GB_SEL_WORKER (_eq_thunk, _uint32)
            case GB_UINT64_code : GB_SEL_WORKER (_eq_thunk, _uint64)
            case GB_FP32_code   : GB_SEL_WORKER (_eq_thunk, _fp32  )
            case GB_FP64_code   : GB_SEL_WORKER (_eq_thunk, _fp64  )
            case GB_FC32_code   : GB_SEL_WORKER (_eq_thunk, _fc32  )
            case GB_FC64_code   : GB_SEL_WORKER (_eq_thunk, _fc64  )
            default: ;
        }
        break ;

    case GB_VALUENE_idxunop_code : // A(i,j) != thunk

        switch (acode)
        {
            case GB_INT8_code   : GB_SEL_WORKER (_ne_thunk, _int8  )
            case GB_INT16_code  : GB_SEL_WORKER (_ne_thunk, _int16 )
            case GB_INT32_code  : GB_SEL_WORKER (_ne_thunk, _int32 )
            case GB_INT64_code  : GB_SEL_WORKER (_ne_thunk, _int64 )
            case GB_UINT8_code  : GB_SEL_WORKER (_ne_thunk, _uint8 )
            case GB_UINT16_code : GB_SEL_WORKER (_ne_thunk, _uint16)
            case GB_UINT32_code : GB_SEL_WORKER (_ne_thunk, _uint32)
            case GB_UINT64_code : GB_SEL_WORKER (_ne_thunk, _uint64)
            case GB_FP32_code   : GB_SEL_WORKER (_ne_thunk, _fp32  )
            case GB_FP64_code   : GB_SEL_WORKER (_ne_thunk, _fp64  )
            case GB_FC32_code   : GB_SEL_WORKER (_ne_thunk, _fc32  )
            case GB_FC64_code   : GB_SEL_WORKER (_ne_thunk, _fc64  )
            default: ;
        }
        break ;

    case GB_VALUEGT_idxunop_code : // A(i,j) > thunk

        switch (acode)
        {
            case GB_INT8_code   : GB_SEL_WORKER (_gt_thunk, _int8  )
            case GB_INT16_code  : GB_SEL_WORKER (_gt_thunk, _int16 )
            case GB_INT32_code  : GB_SEL_WORKER (_gt_thunk, _int32 )
            case GB_INT64_code  : GB_SEL_WORKER (_gt_thunk, _int64 )
            case GB_UINT8_code  : GB_SEL_WORKER (_gt_thunk, _uint8 )
            case GB_UINT16_code : GB_SEL_WORKER (_gt_thunk, _uint16)
            case GB_UINT32_code : GB_SEL_WORKER (_gt_thunk, _uint32)
            case GB_UINT64_code : GB_SEL_WORKER (_gt_thunk, _uint64)
            case GB_FP32_code   : GB_SEL_WORKER (_gt_thunk, _fp32  )
            case GB_FP64_code   : GB_SEL_WORKER (_gt_thunk, _fp64  )
            default: ;
        }
        break ;

    case GB_VALUEGE_idxunop_code : // A(i,j) >= thunk

        switch (acode)
        {
            case GB_INT8_code   : GB_SEL_WORKER (_ge_thunk, _int8  )
            case GB_INT16_code  : GB_SEL_WORKER (_ge_thunk, _int16 )
            case GB_INT32_code  : GB_SEL_WORKER (_ge_thunk, _int32 )
            case GB_INT64_code  : GB_SEL_WORKER (_ge_thunk, _int64 )
            case GB_UINT8_code  : GB_SEL_WORKER (_ge_thunk, _uint8 )
            case GB_UINT16_code : GB_SEL_WORKER (_ge_thunk, _uint16)
            case GB_UINT32_code : GB_SEL_WORKER (_ge_thunk, _uint32)
            case GB_UINT64_code : GB_SEL_WORKER (_ge_thunk, _uint64)
            case GB_FP32_code   : GB_SEL_WORKER (_ge_thunk, _fp32  )
            case GB_FP64_code   : GB_SEL_WORKER (_ge_thunk, _fp64  )
            default: ;
        }
        break ;

    case GB_VALUELT_idxunop_code : // A(i,j) < thunk

        switch (acode)
        {
            case GB_INT8_code   : GB_SEL_WORKER (_lt_thunk, _int8  )
            case GB_INT16_code  : GB_SEL_WORKER (_lt_thunk, _int16 )
            case GB_INT32_code  : GB_SEL_WORKER (_lt_thunk, _int32 )
            case GB_INT64_code  : GB_SEL_WORKER (_lt_thunk, _int64 )
            case GB_UINT8_code  : GB_SEL_WORKER (_lt_thunk, _uint8 )
            case GB_UINT16_code : GB_SEL_WORKER (_lt_thunk, _uint16)
            case GB_UINT32_code : GB_SEL_WORKER (_lt_thunk, _uint32)
            case GB_UINT64_code : GB_SEL_WORKER (_lt_thunk, _uint64)
            case GB_FP32_code   : GB_SEL_WORKER (_lt_thunk, _fp32  )
            case GB_FP64_code   : GB_SEL_WORKER (_lt_thunk, _fp64  )
            default: ;
        }
        break ;

    case GB_VALUELE_idxunop_code : // A(i,j) <= thunk

        switch (acode)
        {
            case GB_INT8_code   : GB_SEL_WORKER (_le_thunk, _int8  )
            case GB_INT16_code  : GB_SEL_WORKER (_le_thunk, _int16 )
            case GB_INT32_code  : GB_SEL_WORKER (_le_thunk, _int32 )
            case GB_INT64_code  : GB_SEL_WORKER (_le_thunk, _int64 )
            case GB_UINT8_code  : GB_SEL_WORKER (_le_thunk, _uint8 )
            case GB_UINT16_code : GB_SEL_WORKER (_le_thunk, _uint16)
            case GB_UINT32_code : GB_SEL_WORKER (_le_thunk, _uint32)
            case GB_UINT64_code : GB_SEL_WORKER (_le_thunk, _uint64)
            case GB_FP32_code   : GB_SEL_WORKER (_le_thunk, _fp32  )
            case GB_FP64_code   : GB_SEL_WORKER (_le_thunk, _fp64  )
            default: ;
        }
        break ;

    default: ;
}

