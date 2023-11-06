//------------------------------------------------------------------------------
// GB_AxB_bitwise_factory.c: switch factory for C=A*B (bitwise monoids)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A template file #include'd in GB_AxB_factory.c, which calls up to 16
// bitwise semirings.  The multiply operators are bor, band, bxor, or bxnor,
// as defined by GB_MNAME.

{
    switch (add_binop_code)
    {

        //----------------------------------------------------------------------
        case GB_BOR_binop_code :     // z = (x | y), bitwise or
        //----------------------------------------------------------------------

            switch (zcode)
            {
                case GB_UINT8_code  : GB_AxB_WORKER (_bor, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_bor, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_bor, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_bor, GB_MNAME, _uint64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BAND_binop_code :    // z = (x & y), bitwise and
        //----------------------------------------------------------------------

            switch (zcode)
            {
                case GB_UINT8_code  : GB_AxB_WORKER (_band, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_band, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_band, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_band, GB_MNAME, _uint64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BXOR_binop_code :    // z = (x ^ y), bitwise xor
        //----------------------------------------------------------------------

            switch (zcode)
            {
                case GB_UINT8_code  : GB_AxB_WORKER (_bxor, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_bxor, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_bxor, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_bxor, GB_MNAME, _uint64)
                default: ;
            }
            break ;

        //----------------------------------------------------------------------
        case GB_BXNOR_binop_code :   // z = ~(x ^ y), bitwise xnor
        //----------------------------------------------------------------------

            switch (zcode)
            {
                case GB_UINT8_code  : GB_AxB_WORKER (_bxnor, GB_MNAME, _uint8 )
                case GB_UINT16_code : GB_AxB_WORKER (_bxnor, GB_MNAME, _uint16)
                case GB_UINT32_code : GB_AxB_WORKER (_bxnor, GB_MNAME, _uint32)
                case GB_UINT64_code : GB_AxB_WORKER (_bxnor, GB_MNAME, _uint64)
                default: ;
            }
            break ;

        default: ;
    }
}

#undef GB_MNAME

