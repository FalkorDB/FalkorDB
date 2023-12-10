// SPDX-License-Identifier: Apache-2.0

#undef GBCUDA_CPLUSPLUS
#include "GB.h"

size_t GB_cuda_type_bits (GB_Type_code);

size_t GB_cuda_type_bits (GB_Type_code type_code) 
{
    switch (type_code)
    {
        case GB_BOOL_code   : return (8) ;
        case GB_INT8_code   : return (8) ;
        case GB_INT16_code  : return (16) ;
        case GB_INT32_code  : return (32) ;
        case GB_INT64_code  : return (64) ;
        case GB_UINT8_code  : return (8) ;
        case GB_UINT16_code : return (16) ;
        case GB_UINT32_code : return (32) ;
        case GB_UINT64_code : return (64) ;
        case GB_FP32_code   : return (32) ;
        case GB_FP64_code   : return (64) ;
//      case GB_FC32_code   : return (64) ;
//      case GB_FC64_code   : return (128) ;
        default             : return (0) ;
    }
}

