//------------------------------------------------------------------------------
// gb_mxclass_to_mxstring: type of a built-in matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "gb_interface.h"

mxArray *gb_mxclass_to_mxstring (mxClassID class, bool is_complex)
{
    switch (class)
    {
        // a built-in sparse or full matrix, valid for G = GrB (X), or
        // for inputs to any GrB.method.
        case mxLOGICAL_CLASS  : return (mxCreateString ("logical")) ;
        case mxINT8_CLASS     : return (mxCreateString ("int8")) ;
        case mxINT16_CLASS    : return (mxCreateString ("int16")) ;
        case mxINT32_CLASS    : return (mxCreateString ("int32")) ;
        case mxINT64_CLASS    : return (mxCreateString ("int64")) ;
        case mxUINT8_CLASS    : return (mxCreateString ("uint8")) ;
        case mxUINT16_CLASS   : return (mxCreateString ("uint16")) ;
        case mxUINT32_CLASS   : return (mxCreateString ("uint32")) ;
        case mxUINT64_CLASS   : return (mxCreateString ("uint64")) ;

        case mxSINGLE_CLASS   :
            if (is_complex)
            { 
                return (mxCreateString ("single complex")) ;
            }
            else
            { 
                return (mxCreateString ("single")) ;
            }
            break ;

        case mxDOUBLE_CLASS   :
            if (is_complex)
            { 
                return (mxCreateString ("double complex")) ;
            }
            else
            { 
                return (mxCreateString ("double")) ;
            }
            break ;

        // a built-in struct, cell, char, void, function, or unknown
        case mxSTRUCT_CLASS   : return (mxCreateString ("struct")) ;
        case mxCELL_CLASS     : return (mxCreateString ("cell")) ;
        case mxCHAR_CLASS     : return (mxCreateString ("char")) ;
        case mxVOID_CLASS     : return (mxCreateString ("void")) ;
        case mxFUNCTION_CLASS : return (mxCreateString ("function_handle")) ;
        case mxUNKNOWN_CLASS  :
        default               : return (mxCreateString ("unknown")) ;
    }
}

