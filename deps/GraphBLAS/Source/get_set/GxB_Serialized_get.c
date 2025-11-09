//------------------------------------------------------------------------------
// GxB_Serialized_get_*: query the contents of a serialized blob
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "get_set/GB_get_set.h"
#include "serialize/GB_serialize.h"
#define GB_FREE_ALL ;

//------------------------------------------------------------------------------
// GB_blob_header_get: get all properties of the blob
//------------------------------------------------------------------------------

static GrB_Info GB_blob_header_get
(
    // output:
    char *type_name,            // name of the type (char array of size at
                                // least GxB_MAX_NAME_LEN)
    int32_t *type_code,         // type code of the matrix
    int32_t *sparsity_status,   // sparsity status
    int32_t *sparsity_ctrl,     // sparsity control
    double *hyper_sw,           // hyper_switch
    double *bitmap_sw,          // bitmap_switch
    int32_t *storage,           // GrB_COLMAJOR or GrB_ROWMAJOR
    char **user_name,           // GrB_NAME of the blob
    char **eltype_string,       // GrB_EL_TYPE_STRING of the type of the blob
    bool *is_csc,
    bool *p_is_32,              // if true, A->p is 32 bit; else 64
    bool *j_is_32,              // if true, A->h and A->Y are 32 bit; else 64
    bool *i_is_32,              // if true, A->i is 32 bit; else 64
    int8_t *p_control,
    int8_t *j_control,
    int8_t *i_control,
    bool *iso,

    // input, not modified:
    const GB_void *blob,        // the blob
    uint64_t blob_size          // size of the blob
)
{

    //--------------------------------------------------------------------------
    // read the content of the header (160 bytes)
    //--------------------------------------------------------------------------

    size_t s = 0 ;

    if (blob_size < GB_BLOB_HEADER_SIZE)
    { 
        // blob is invalid
        return (GrB_INVALID_OBJECT)  ;
    }

    GB_BLOB_READ (blob_size2, uint64_t) ;

// was in v9.4.2 and earlier::
//  GB_BLOB_READ (typecode, int32_t) ;
// now in GrB v10.0.0:
    GB_BLOB_READ (encoding, uint32_t) ;
    uint32_t Cp_is_32 = GB_RSHIFT (encoding, 12, 4) ; // C->p_is_32
    uint32_t Cj_is_32 = GB_RSHIFT (encoding,  8, 4) ; // C->j_is_32
    uint32_t Ci_is_32 = GB_RSHIFT (encoding,  4, 4) ; // C->i_is_32
    uint32_t typecode = GB_RSHIFT (encoding,  0, 4) ; // 4 bit typecode

    // GrB 10.0.0 reserves 4 bits each for Cp_is_32, Cj_is_32, and Ci_is_32,
    // for future expansion.  This way, if a future GraphBLAS version needs
    // more bits to create a serialized blob, then GrB 10.0.0 will gracefully
    // fail if it attempts to deserialize the blob.

    uint64_t blob_size1 = (uint64_t) blob_size ;

    // GrB v9.4.2 has the same test below, so it will safely declare the blob
    // invalid if it sees any encoding with a 1 in bit position 4 or 5.
    if (blob_size1 != blob_size2
        || typecode < GB_BOOL_code || typecode > GB_UDT_code
        || (typecode == GB_UDT_code &&
            blob_size < GB_BLOB_HEADER_SIZE + GxB_MAX_NAME_LEN)
        // GrB v10.0.0 adds the following check, since it only supports the
        // values of 0 and 1, denoting 64-bit and 32-bit integers respectively:
        || (Cp_is_32 > 1) || (Cj_is_32 > 1) || (Ci_is_32 > 1))
    { 
        // blob is invalid
        return (GrB_INVALID_OBJECT)  ;
    }

    GB_BLOB_READ (version, int32_t) ;
    GB_BLOB_READ (vlen, int64_t) ;
    GB_BLOB_READ (vdim, int64_t) ;
    GB_BLOB_READ (nvec, int64_t) ;
    GB_BLOB_READ (nvec_nonempty, int64_t) ;     ASSERT (nvec_nonempty >= 0) ;
    GB_BLOB_READ (nvals, int64_t) ;
    GB_BLOB_READ (typesize, int64_t) ;
    GB_BLOB_READ (Cp_len, int64_t) ;
    GB_BLOB_READ (Ch_len, int64_t) ;
    GB_BLOB_READ (Cb_len, int64_t) ;
    GB_BLOB_READ (Ci_len, int64_t) ;
    GB_BLOB_READ (Cx_len, int64_t) ;
    GB_BLOB_READ (hyper_switch, float) ;
    GB_BLOB_READ (bitmap_switch, float) ;

// was in v9.4.2 and earlier::
//  GB_BLOB_READ (sparsity_control, int32_t) ;
// now in GrB v10.0.0:
    GB_BLOB_READ (control_encoding, uint32_t) ;
    uint32_t p_encoding = GB_RSHIFT (control_encoding, 16, 4) ;
    uint32_t j_encoding = GB_RSHIFT (control_encoding, 12, 4) ;
    uint32_t i_encoding = GB_RSHIFT (control_encoding,  8, 4) ;
    (*p_control) = GB_pji_control_decoding (p_encoding) ;
    (*j_control) = GB_pji_control_decoding (j_encoding) ;
    (*i_control) = GB_pji_control_decoding (i_encoding) ;
    uint32_t sparsity_control = GB_RSHIFT (control_encoding,  0, 8) ;

    GB_BLOB_READ (sparsity_iso_csc, int32_t) ;
    GB_BLOB_READ (Cp_nblocks, int32_t) ; GB_BLOB_READ (Cp_method, int32_t) ;
    GB_BLOB_READ (Ch_nblocks, int32_t) ; GB_BLOB_READ (Ch_method, int32_t) ;
    GB_BLOB_READ (Cb_nblocks, int32_t) ; GB_BLOB_READ (Cb_method, int32_t) ;
    GB_BLOB_READ (Ci_nblocks, int32_t) ; GB_BLOB_READ (Ci_method, int32_t) ;
    GB_BLOB_READ (Cx_nblocks, int32_t) ; GB_BLOB_READ (Cx_method, int32_t) ;

    (*sparsity_status) = sparsity_iso_csc / 4 ;
    (*iso) = ((sparsity_iso_csc & 2) == 2) ;
    (*is_csc) = ((sparsity_iso_csc & 1) == 1) ;

    //--------------------------------------------------------------------------

    (*sparsity_ctrl) = sparsity_control ;
    (*hyper_sw)  = (double) hyper_switch ;
    (*bitmap_sw) = (double) bitmap_switch ;
    (*storage) = (*is_csc) ? GrB_COLMAJOR : GrB_ROWMAJOR ;
    (*p_is_32) = Cp_is_32 ;
    (*j_is_32) = Cj_is_32 ;
    (*i_is_32) = Ci_is_32 ;

    //--------------------------------------------------------------------------
    // determine the matrix type_code and C type_name
    //--------------------------------------------------------------------------

    (*type_code) = GB_type_code_get (typecode) ;
    memset (type_name, 0, GxB_MAX_NAME_LEN) ;

    if (typecode >= GB_BOOL_code && typecode < GB_UDT_code)
    { 
        // blob has a built-in type; the name is not in the blob
        strcpy (type_name, GB_code_string (typecode)) ;
    }
    else if (typecode == GB_UDT_code)
    { 
        // blob has a user-defined type
        // get the GxB_JIT_C_NAME of the user type from the blob
        memcpy (type_name, ((GB_void *) blob) + GB_BLOB_HEADER_SIZE,
            GxB_MAX_NAME_LEN) ;
        s += GxB_MAX_NAME_LEN ;
    }

    // this should already be in the blob, but set it to null just in case
    type_name [GxB_MAX_NAME_LEN-1] = '\0' ;

    //--------------------------------------------------------------------------
    // get the compressed block sizes from the blob for each array
    //--------------------------------------------------------------------------

    GB_BLOB_READS (Cp_Sblocks, Cp_nblocks) ;
    GB_BLOB_READS (Ch_Sblocks, Ch_nblocks) ;
    GB_BLOB_READS (Cb_Sblocks, Cb_nblocks) ;
    GB_BLOB_READS (Ci_Sblocks, Ci_nblocks) ;
    GB_BLOB_READS (Cx_Sblocks, Cx_nblocks) ;

    //--------------------------------------------------------------------------
    // skip past each array (Cp, Ch, Cb, Ci, and Cx)
    //--------------------------------------------------------------------------

    switch (*sparsity_status)
    {
        case GxB_HYPERSPARSE : 
            // skip Cp, Ch, and Ci
            s += (Cp_nblocks > 0) ? Cp_Sblocks [Cp_nblocks-1] : 0 ;
            s += (Ch_nblocks > 0) ? Ch_Sblocks [Ch_nblocks-1] : 0 ;
            s += (Ci_nblocks > 0) ? Ci_Sblocks [Ci_nblocks-1] : 0 ;
            break ;

        case GxB_SPARSE : 
            // skip Cp and Ci
            s += (Cp_nblocks > 0) ? Cp_Sblocks [Cp_nblocks-1] : 0 ;
            s += (Ci_nblocks > 0) ? Ci_Sblocks [Ci_nblocks-1] : 0 ;
            break ;

        case GxB_BITMAP : 
            // skip Cb
            s += (Cb_nblocks > 0) ? Cb_Sblocks [Cb_nblocks-1] : 0 ;
            break ;

        case GxB_FULL : 
            break ;
        default: ;
    }

    // skip Cx
    s += (Cx_nblocks > 0) ? Cx_Sblocks [Cx_nblocks-1] : 0 ;

    //--------------------------------------------------------------------------
    // get the GrB_NAME and GrB_EL_TYPE_STRING
    //--------------------------------------------------------------------------

    // v8.1.0 adds two nul-terminated uncompressed strings to the end of the
    // blob.  If the strings are empty, the nul terminators still appear.

    (*user_name) = NULL ;
    (*eltype_string) = NULL ;

    if (version >= GxB_VERSION (8,1,0))
    { 

        //----------------------------------------------------------------------
        // look for the two nul bytes in blob [s : blob_size-1]
        //----------------------------------------------------------------------

        int nfound = 0 ;
        size_t ss [2] ;
        for (size_t p = s ; p < blob_size && nfound < 2 ; p++)
        {
            if (blob [p] == 0)
            { 
                ss [nfound++] = p ;
            }
        }

        if (nfound == 2)
        { 
            // extract the GrB_NAME and GrB_EL_TYPE_STRING from the blob
            (*user_name) = (char *) (blob + s) ;
            (*eltype_string) = (char *) (blob + ss [0] + 1) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    #pragma omp flush
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GB_Serialized_get: get an int, double, or string from a serialized blob
//------------------------------------------------------------------------------

static GrB_Info GB_Serialized_get
(
    const void * blob,
    int field,
    int32_t *ivalue,
    double *dvalue,
    char *cvalue,
    bool *is_double,
    bool *is_char,
    size_t blob_size
)
{

    //--------------------------------------------------------------------------
    // read the blob
    //--------------------------------------------------------------------------

    GrB_Info info ;
    (*ivalue) = 0 ;
    (*dvalue) = 0 ;
    (*cvalue) = '\0' ;
    (*is_double) = false ;
    (*is_char) = false ;

    char type_name [GxB_MAX_NAME_LEN], *user_name, *eltype_string ;
    int32_t sparsity_status, sparsity_ctrl, type_code, storage ;
    double hyper_sw, bitmap_sw ;
    bool is_csc, p_is_32, j_is_32, i_is_32, iso ;
    int8_t p_control, j_control, i_control ;

    GB_OK (GB_blob_header_get (type_name, &type_code,
        &sparsity_status, &sparsity_ctrl, &hyper_sw, &bitmap_sw, &storage,
        &user_name, &eltype_string, &is_csc, &p_is_32, &j_is_32, &i_is_32,
        &p_control, &j_control, &i_control, &iso, blob, blob_size)) ;

    //--------------------------------------------------------------------------
    // get the field
    //--------------------------------------------------------------------------

    switch ((int) field)
    {
        case GrB_STORAGE_ORIENTATION_HINT : 

            (*ivalue) = storage ;
            break ;

        case GrB_EL_TYPE_CODE : 

            (*ivalue) = type_code ;
            break ;

        case GxB_SPARSITY_CONTROL : 

            (*ivalue) = sparsity_ctrl ;
            break ;

        case GxB_SPARSITY_STATUS : 

            (*ivalue) = sparsity_status ;
            break ;

        case GxB_ISO : 

            (*ivalue) = iso ;
            break ;

        case GxB_FORMAT : 

            (*ivalue) = (storage == GrB_COLMAJOR) ? GxB_BY_COL : GxB_BY_ROW ;
            break ;

        case GxB_OFFSET_INTEGER_HINT : 

            (*ivalue) = p_control ;
            break ;

        case GxB_OFFSET_INTEGER_BITS : 

            (*ivalue) = (p_is_32) ? 32 : 64 ;
            break ;

        case GxB_COLINDEX_INTEGER_HINT : 

            (*ivalue) = (is_csc) ? j_control : i_control ;
            break ;

        case GxB_COLINDEX_INTEGER_BITS : 

            (*ivalue) = ((is_csc) ? j_is_32 : i_is_32) ? 32 : 64 ;
            break ;

        case GxB_ROWINDEX_INTEGER_HINT : 

            (*ivalue) = (is_csc) ? i_control : j_control ;
            break ;

        case GxB_ROWINDEX_INTEGER_BITS : 

            (*ivalue) = ((is_csc) ? i_is_32 : j_is_32) ? 32 : 64 ;
            break ;

        case GxB_HYPER_SWITCH : 

            (*dvalue) = hyper_sw ;
            (*is_double) = true ;
            break ;

        case GxB_BITMAP_SWITCH : 

            (*dvalue) = bitmap_sw ;
            (*is_double) = true ;
            break ;

        case GrB_NAME : 

            if (user_name != NULL)
            { 
                strcpy (cvalue, user_name) ;
            }
            (*is_char) = true ;
            break ;

        case GxB_JIT_C_NAME : 

            strcpy (cvalue, type_name) ;
            (*is_char) = true ;
            break ;

        case GrB_EL_TYPE_STRING : 

            if (eltype_string != NULL)
            { 
                strcpy (cvalue, eltype_string) ;
            }
            (*is_char) = true ;
            break ;

        default : 

            return (GrB_INVALID_VALUE) ;
    }

    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------
// GxB_Serialized_get_Scalar
//------------------------------------------------------------------------------

GrB_Info GxB_Serialized_get_Scalar
(
    const void * blob,
    GrB_Scalar scalar,
    int field,
    size_t blob_size
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GB_WHERE_1 (scalar, "GxB_Serialized_get_Scalar (blob, scalar, field,"
        " blob_size)") ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (scalar) ;

    //--------------------------------------------------------------------------
    // read the blob
    //--------------------------------------------------------------------------

    int32_t ivalue ;
    double dvalue ;
    char cvalue [GxB_MAX_NAME_LEN] ;
    bool is_double, is_char ;

    GB_OK (GB_Serialized_get (blob, field, &ivalue, &dvalue, cvalue,
        &is_double, &is_char, blob_size)) ;

    if (is_char)
    { 
        return (GrB_INVALID_VALUE) ;
    }
    else if (is_double)
    { 
        // field specifies a double: assign it to the scalar
        return (GB_setElement ((GrB_Matrix) scalar, NULL, &dvalue, 0, 0,
            GB_FP64_code, Werk)) ;
    }
    else
    { 
        // field specifies an int32_t: assign it to the scalar
        return (GB_setElement ((GrB_Matrix) scalar, NULL, &ivalue, 0, 0,
            GB_INT32_code, Werk)) ;
    }
}

//------------------------------------------------------------------------------
// GxB_Serialized_get_String
//------------------------------------------------------------------------------

GrB_Info GxB_Serialized_get_String
(
    const void * blob,
    char * value,
    int field,
    size_t blob_size
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // read the blob
    //--------------------------------------------------------------------------

    int32_t ivalue ;
    double dvalue ;
    bool is_double, is_char ;

    GB_OK (GB_Serialized_get (blob, field, &ivalue, &dvalue, value,
        &is_double, &is_char, blob_size)) ;

    if (is_char)
    { 
        #pragma omp flush
        return (GrB_SUCCESS) ;
    }
    else
    { 
        return (GrB_INVALID_VALUE) ;
    }
}

//------------------------------------------------------------------------------
// GxB_Serialized_get_INT32
//------------------------------------------------------------------------------

GrB_Info GxB_Serialized_get_INT32
(
    const void * blob,
    int32_t * value,
    int field,
    size_t blob_size
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    double dvalue ;
    bool is_double, is_char ;
    char cvalue [GxB_MAX_NAME_LEN] ;

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // read the blob (must be an integer value)
    //--------------------------------------------------------------------------

    GB_OK (GB_Serialized_get (blob, field, value, &dvalue, cvalue,
        &is_double, &is_char, blob_size)) ;

    if (is_double || is_char)
    { 
        return (GrB_INVALID_VALUE) ;
    }
    else
    { 
        #pragma omp flush
        return (GrB_SUCCESS) ;
    }
}

//------------------------------------------------------------------------------
// GxB_Serialized_get_SIZE
//------------------------------------------------------------------------------

GrB_Info GxB_Serialized_get_SIZE
(
    const void * blob,
    size_t * value,
    int field,
    size_t blob_size
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    int32_t ivalue ;
    double dvalue ;
    bool is_double, is_char ;
    char cvalue [GxB_MAX_NAME_LEN] ;

    GB_CHECK_INIT ;
    GB_RETURN_IF_NULL (blob) ;
    GB_RETURN_IF_NULL (value) ;

    //--------------------------------------------------------------------------
    // read the blob
    //--------------------------------------------------------------------------

    GB_OK (GB_Serialized_get (blob, field, &ivalue, &dvalue, cvalue,
        &is_double, &is_char, blob_size)) ;

    if (is_char)
    { 
        (*value) = strlen (cvalue) + 1 ;
        #pragma omp flush
        return (GrB_SUCCESS) ;
    }
    else
    { 
        return (GrB_INVALID_VALUE) ;
    }
}

//------------------------------------------------------------------------------
// GxB_Serialized_get_VOID
//------------------------------------------------------------------------------

GrB_Info GxB_Serialized_get_VOID
(
    const void * blob,
    void * value,
    int field,
    size_t blob_size
)
{ 
    return (GrB_INVALID_VALUE) ;
}

