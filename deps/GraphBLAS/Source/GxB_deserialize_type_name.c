//------------------------------------------------------------------------------
// GxB_deserialize_type_name: return the JIT C name of the type of a blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "GB.h"
#include "GB_serialize.h"

// This method is historical; use GrB_get instead.

// GxB_deserialize_type_name extracts the JIT C type_name of the GrB_Type of
// the GrB_Matrix or GrB_Vector held in a serialized blob.  On input, type_name
// must point to a user-owned char array of size at least GxB_MAX_NAME_LEN (it
// must not point into the blob itself).  On output, type_name will contain a
// null-terminated string with the corresponding C type name.  If the blob
// holds a matrix of a built-in type, the name is returned as "bool" for
// GrB_BOOL, "uint8_t" for GrB_UINT8, "float complex" for GxB_FC32, etc.
// If the type is user-defined, the GxB_JIT_C_NAME of the type is returned.

GrB_Info GxB_deserialize_type_name  // return the type name of a blob
(
    // output:
    char *type_name,        // name of the type (char array of size at least
                            // GxB_MAX_NAME_LEN, owned by the user application).
    // input, not modified:
    const void *blob,       // the blob
    GrB_Index blob_size     // size of the blob
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE1 ("GxB_deserialize_type_name (type_name, blob, blob_size)") ;
    GB_RETURN_IF_NULL (type_name) ;
    GB_RETURN_IF_NULL (blob) ;

    if (blob_size < GB_BLOB_HEADER_SIZE)
    { 
        // blob is invalid
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // get the blob header
    //--------------------------------------------------------------------------

    size_t s = 0 ;
    GB_BLOB_READ (blob_size2, uint64_t) ;
    GB_BLOB_READ (typecode, int32_t) ;

    if (blob_size2 != blob_size)
    { 
        // blob is invalid
        return (GrB_INVALID_OBJECT) ;
    }

    //--------------------------------------------------------------------------
    // get the type_name from the built-in type or the blob
    //--------------------------------------------------------------------------

    if (typecode >= GB_BOOL_code && typecode < GB_UDT_code)
    { 
        // blob has a built-in type; the name is not in the blob
        GrB_Type blob_type = GB_code_type ((GB_Type_code) typecode, NULL) ;
        ASSERT (blob_type != NULL) ;
        memcpy (type_name, blob_type->name, GxB_MAX_NAME_LEN) ;
    }
    else if (typecode == GB_UDT_code)
    { 
        // blob has a user-defined type
        if (blob_size < GB_BLOB_HEADER_SIZE + GxB_MAX_NAME_LEN)
        { 
            // blob is invalid
            return (GrB_INVALID_OBJECT) ;
        }
        // get the name of the user type from the blob
        memcpy (type_name, ((GB_void *) blob) + GB_BLOB_HEADER_SIZE,
            GxB_MAX_NAME_LEN) ;
    }
    else
    { 
        // blob is invalid
        return (GrB_INVALID_OBJECT) ;
    }

    // this should already be in the blob, but set it to null just in case
    type_name [GxB_MAX_NAME_LEN-1] = '\0' ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

