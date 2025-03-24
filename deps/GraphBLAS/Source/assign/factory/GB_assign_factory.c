//------------------------------------------------------------------------------
// GB_assign_factory.c:  switch factory for assign (a single type)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

    switch (ccode)
    {
        #ifndef GxB_NO_BOOL
            case GB_BOOL_code:    GB_WORKER (_bool  )
        #endif
        #ifndef GxB_NO_INT8
            case GB_INT8_code:    GB_WORKER (_int8  )
        #endif
        #ifndef GxB_NO_INT16
            case GB_INT16_code:   GB_WORKER (_int16 )
        #endif
        #ifndef GxB_NO_INT32
            case GB_INT32_code:   GB_WORKER (_int32 )
        #endif
        #ifndef GxB_NO_INT64
            case GB_INT64_code:   GB_WORKER (_int64 )
        #endif
        #ifndef GxB_NO_UINT8
            case GB_UINT8_code:   GB_WORKER (_uint8 )
        #endif
        #ifndef GxB_NO_UINT16
            case GB_UINT16_code:  GB_WORKER (_uint16)
        #endif
        #ifndef GxB_NO_UINT32
            case GB_UINT32_code:  GB_WORKER (_uint32)
        #endif
        #ifndef GxB_NO_UINT64
            case GB_UINT64_code:  GB_WORKER (_uint64)
        #endif
        #ifndef GxB_NO_FP32
            case GB_FP32_code:    GB_WORKER (_fp32  )
        #endif
        #ifndef GxB_NO_FP64
            case GB_FP64_code:    GB_WORKER (_fp64  )
        #endif
        #ifndef GxB_NO_FC32
            case GB_FC32_code:    GB_WORKER (_fc32  )
        #endif
        #ifndef GxB_NO_FC64
            case GB_FC64_code:    GB_WORKER (_fc64  )
        #endif
        default: ;
    }

