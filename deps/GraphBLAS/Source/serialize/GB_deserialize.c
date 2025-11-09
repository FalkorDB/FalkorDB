//------------------------------------------------------------------------------
// GB_deserialize: decompress and deserialize a blob into a GrB_Matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A parallel decompression of a serialized blob into a GrB_Matrix.

#include "GB.h"
#include "get_set/GB_get_set.h"
#include "serialize/GB_serialize.h"

#define GB_FREE_ALL                         \
{                                           \
    GB_Matrix_free (&T) ;                   \
    GB_Matrix_free (&C) ;                   \
}

GrB_Info GB_deserialize             // deserialize a matrix from a blob
(
    // output:
    GrB_Matrix *Chandle,            // output matrix created from the blob
    // input:
    GrB_Type type_expected,         // type expected (NULL for any built-in)
    const GB_void *blob,            // serialized matrix 
    size_t blob_size                // size of the blob
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (blob != NULL && Chandle != NULL) ;
    (*Chandle) = NULL ;
    GrB_Matrix C = NULL, T = NULL ;

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
    int8_t p_control = GB_pji_control_decoding (p_encoding) ;
    int8_t j_control = GB_pji_control_decoding (j_encoding) ;
    int8_t i_control = GB_pji_control_decoding (i_encoding) ;
    uint32_t sparsity_control = GB_RSHIFT (control_encoding,  0, 8) ;

    GB_BLOB_READ (sparsity_iso_csc, int32_t) ;
    GB_BLOB_READ (Cp_nblocks, int32_t) ; GB_BLOB_READ (Cp_method, int32_t) ;
    GB_BLOB_READ (Ch_nblocks, int32_t) ; GB_BLOB_READ (Ch_method, int32_t) ;
    GB_BLOB_READ (Cb_nblocks, int32_t) ; GB_BLOB_READ (Cb_method, int32_t) ;
    GB_BLOB_READ (Ci_nblocks, int32_t) ; GB_BLOB_READ (Ci_method, int32_t) ;
    GB_BLOB_READ (Cx_nblocks, int32_t) ; GB_BLOB_READ (Cx_method, int32_t) ;

    int32_t sparsity = sparsity_iso_csc / 4 ;
    bool iso = ((sparsity_iso_csc & 2) == 2) ;
    bool is_csc = ((sparsity_iso_csc & 1) == 1) ;

    //--------------------------------------------------------------------------
    // determine the matrix type
    //--------------------------------------------------------------------------

    GB_Type_code ccode = (GB_Type_code) typecode ;
    GrB_Type ctype = GB_code_type (ccode, type_expected) ;

    // ensure the type has the right size
    if (ctype == NULL || ctype->size != typesize)
    { 
        // blob is invalid; type is missing or the wrong size
        return (GrB_DOMAIN_MISMATCH) ;
    }

    if (ccode == GB_UDT_code)
    {
        // user-defined name is 128 bytes, if present
        // ensure the user-defined type has the right name
        ASSERT (ctype == type_expected) ;
        if (strncmp ((const char *) (blob + s), ctype->name,
            GxB_MAX_NAME_LEN) != 0)
        { 
            // blob is invalid
            return (GrB_DOMAIN_MISMATCH) ;
        }
        s += GxB_MAX_NAME_LEN ;
    }
    else if (type_expected != NULL && ctype != type_expected)
    { 
        // built-in type must match type_expected
        // blob is invalid
        return (GrB_DOMAIN_MISMATCH) ;
    }

    //--------------------------------------------------------------------------
    // get the compressed block sizes from the blob for each array
    //--------------------------------------------------------------------------

    GB_BLOB_READS (Cp_Sblocks, Cp_nblocks) ;
    GB_BLOB_READS (Ch_Sblocks, Ch_nblocks) ;
    GB_BLOB_READS (Cb_Sblocks, Cb_nblocks) ;
    GB_BLOB_READS (Ci_Sblocks, Ci_nblocks) ;
    GB_BLOB_READS (Cx_Sblocks, Cx_nblocks) ;

    //--------------------------------------------------------------------------
    // allocate the output matrix C
    //--------------------------------------------------------------------------

    // allocate the matrix with info from the header
    GB_OK (GB_new (&C,  // new header (C is NULL on input)
        ctype, vlen, vdim, GB_ph_null, is_csc,
        sparsity, hyper_switch, nvec, Cp_is_32, Cj_is_32, Ci_is_32)) ;

    C->nvec = nvec ;
//  C->nvec_nonempty = nvec_nonempty ;
    GB_nvec_nonempty_set (C, nvec_nonempty) ;
    C->nvals = nvals ;      // revised below if version <= 7.2.0
    C->bitmap_switch = bitmap_switch ;
    C->sparsity_control = sparsity_control ;
    C->iso = iso ;

    // added for GrB v10.0.0:
    C->p_is_32 = Cp_is_32 ;
    C->j_is_32 = Cj_is_32 ;
    C->i_is_32 = Ci_is_32 ;
    C->p_control = p_control ;
    C->j_control = j_control ;
    C->i_control = i_control ;

    //--------------------------------------------------------------------------
    // decompress each array (Cp, Ch, Cb, Ci, and Cx)
    //--------------------------------------------------------------------------

    switch (sparsity)
    {
        case GxB_HYPERSPARSE : 
            // decompress Cp, Ch, and Ci
            GB_OK (GB_deserialize_from_blob ((GB_void **) &(C->p), &(C->p_size),
                Cp_len, blob, blob_size, Cp_Sblocks, Cp_nblocks, Cp_method,
                &s)) ;

            GB_OK (GB_deserialize_from_blob ((GB_void **) &(C->h), &(C->h_size),
                Ch_len, blob, blob_size, Ch_Sblocks, Ch_nblocks, Ch_method,
                &s)) ;

            GB_OK (GB_deserialize_from_blob ((GB_void **) &(C->i), &(C->i_size),
                Ci_len, blob, blob_size, Ci_Sblocks, Ci_nblocks, Ci_method,
                &s)) ;
            break ;

        case GxB_SPARSE : 

            // decompress Cp and Ci
            GB_OK (GB_deserialize_from_blob ((GB_void **) &(C->p), &(C->p_size),
                Cp_len, blob, blob_size, Cp_Sblocks, Cp_nblocks, Cp_method,
                &s)) ;

            GB_OK (GB_deserialize_from_blob ((GB_void **) &(C->i), &(C->i_size),
                Ci_len, blob, blob_size, Ci_Sblocks, Ci_nblocks, Ci_method,
                &s)) ;
            break ;

        case GxB_BITMAP : 

            // decompress Cb
            GB_OK (GB_deserialize_from_blob ((GB_void **) &(C->b), &(C->b_size),
                Cb_len, blob, blob_size, Cb_Sblocks, Cb_nblocks, Cb_method,
                &s)) ;
            break ;

        case GxB_FULL : 
            break ;
        default: ;
    }

    // decompress Cx
    GB_OK (GB_deserialize_from_blob ((GB_void **) &(C->x), &(C->x_size), Cx_len,
        blob, blob_size, Cx_Sblocks, Cx_nblocks, Cx_method, &s)) ;

    if (C->p != NULL && version <= GxB_VERSION (7,2,0))
    {
        // C is sparse or hypersparse.  v7.2.1 and later have the new C->nvals
        // value inside the blob already.  The blob prior to v7.2.1 had nvals
        // of zero for sparse and hypersparse matrices.  Set it here to the
        // correct value, so that blobs written by v7.2.0 and earlier can be
        // read by v7.2.1 and later.  For blobs written by v7.2.0 and earlier,
        // get C->nvals from Cp [nvec] when C is sparse or hypersparse.  For
        // blobs written by v7.2.1 and later, use nvals as read in from the
        // blob above.
        const uint64_t *restrict Cp = C->p ; // OK; v7.2.0 only had 64-bit Cp
        C->nvals = Cp [C->nvec] ;
    }
    C->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // get the GrB_NAME and GrB_EL_TYPE_STRING
    //--------------------------------------------------------------------------

    // v8.1.0 adds two nul-terminated uncompressed strings to the end of the
    // blob: the user name and the element type name.  If the strings are
    // empty, the nul terminators still appear.

    if (version >= GxB_VERSION (8,1,0))
    { 

        //----------------------------------------------------------------------
        // look for the two nul bytes in blob [s : blob_size-1]
        //----------------------------------------------------------------------

        int nfound = 0 ;
//      size_t ss [2] ;
        for (size_t p = s ; p < blob_size && nfound < 2 ; p++)
        {
            if (blob [p] == 0)
            {
//              ss [nfound] = p ;
                nfound++ ;
            }
        }

        if (nfound == 2)
        { 
            // extract the GrB_NAME from the blob;
            // GrB_EL_TYPE_STRING not needed
            char *user_name = (char *) (blob + s) ;
//          char *eltype_string = (char *) (blob + ss [0] + 1) ;
            GB_OK (GB_matvec_name_set (C, user_name, GrB_NAME)) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    (*Chandle) = C ;
    ASSERT_MATRIX_OK (*Chandle, "Final result from deserialize", GB0) ;
    return (GrB_SUCCESS) ;
}

