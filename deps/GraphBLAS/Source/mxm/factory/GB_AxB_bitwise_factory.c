//------------------------------------------------------------------------------
// GB_AxB_bitwise_factory.c: switch factory for C=A*B (bitwise monoids)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A template file #include'd in GB_AxB_factory.c, which calls up to 16
// bitwise semirings.  The multiply operators are bor, band, bxor, or bxnor,
// as defined by GB_MNAME.

#if defined (GxB_NO_UINT8)
#define GB_CASE_UINT8(op)
#else
#define GB_CASE_UINT8(op) \
    case GB_UINT8_code:  GB_AxB_WORKER (op, GB_MNAME, _uint8 )
#endif

#if defined (GxB_NO_UINT16)
#define GB_CASE_UINT16(op)
#else
#define GB_CASE_UINT16(op) \
    case GB_UINT16_code: GB_AxB_WORKER (op, GB_MNAME, _uint16)
#endif

#if defined (GxB_NO_UINT32)
#define GB_CASE_UINT32(op)
#else
#define GB_CASE_UINT32(op) \
    case GB_UINT32_code: GB_AxB_WORKER (op, GB_MNAME, _uint32)
#endif

#if defined (GxB_NO_UINT64)
#define GB_CASE_UINT64(op)
#else
#define GB_CASE_UINT64(op) \
    case GB_UINT64_code: GB_AxB_WORKER (op, GB_MNAME, _uint64)
#endif

{
    switch (add_binop_code)
    {

        //----------------------------------------------------------------------
        case GB_BOR_binop_code :     // z = (x | y), bitwise or
        //----------------------------------------------------------------------

            switch (zcode)
            {
                GB_CASE_UINT8  (_bor)
                GB_CASE_UINT16 (_bor)
                GB_CASE_UINT32 (_bor)
                GB_CASE_UINT64 (_bor)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BAND_binop_code :    // z = (x & y), bitwise and
        //----------------------------------------------------------------------

            switch (zcode)
            {
                GB_CASE_UINT8  (_band)
                GB_CASE_UINT16 (_band)
                GB_CASE_UINT32 (_band)
                GB_CASE_UINT64 (_band)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BXOR_binop_code :    // z = (x ^ y), bitwise xor
        //----------------------------------------------------------------------

            switch (zcode)
            {
                GB_CASE_UINT8  (_bxor)
                GB_CASE_UINT16 (_bxor)
                GB_CASE_UINT32 (_bxor)
                GB_CASE_UINT64 (_bxor)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BXNOR_binop_code :   // z = ~(x ^ y), bitwise xnor
        //----------------------------------------------------------------------

            switch (zcode)
            {
                GB_CASE_UINT8  (_bxnor)
                GB_CASE_UINT16 (_bxnor)
                GB_CASE_UINT32 (_bxnor)
                GB_CASE_UINT64 (_bxnor)
                default: ;
            }
            break ;

        default: ;
    }
}

#undef GB_MNAME

#undef GB_CASE_UINT8
#undef GB_CASE_UINT16
#undef GB_CASE_UINT32
#undef GB_CASE_UINT64

