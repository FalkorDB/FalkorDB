//------------------------------------------------------------------------------
// gbmx_mxarray_to_descriptor: get the contents of a MATLAB descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// get a gb_descriptor from a built-in MATLAB struct.

static void get_desc
(
    // output:
    int *desc_field,        // field in gb_descriptor to modify
    int *desc_nondefault,   // if true, gbdesc has a nondefault option
    // input
    const mxArray *mxdesc,  // MATLAB struct with d.out, etc
    const char *fieldname   // fieldname to extract from mxdesc
)
{

    // find the field in the MATLAB struct
    int fieldnumber = mxGetFieldNumber (mxdesc, fieldname) ;
    if (fieldnumber >= 0)
    {

        // the field is present
        mxArray *value = mxGetFieldByNumber (mxdesc, 0, fieldnumber) ;

        // get the string from the MATLAB struct field
        char s [LEN+2] ;
        gbmx_mxstring_to_string (s, LEN, value, "field") ;

        // convert the string to a Descriptor value, and set the value
        if (MATCH (s, "default"))
        { 
            (*desc_field) = GxB_DEFAULT ;
        }
        else if (MATCH (s, "transpose"))
        { 
            (*desc_field) = GrB_TRAN ;
            (*desc_nondefault) = true ;
        }
        else if (MATCH (s, "complement"))
        { 
            (*desc_field) = GrB_COMP ;
            (*desc_nondefault) = true ;
        }
        else if (MATCH (s, "structure") || MATCH (s, "structural"))
        { 
            (*desc_field) = GrB_STRUCTURE ;
            (*desc_nondefault) = true ;
        }
        else if (MATCH (s, "structural complement"))
        { 
            (*desc_field) = GrB_COMP + GrB_STRUCTURE ;
            (*desc_nondefault) = true ;
        }
        else if (MATCH (s, "replace"))
        { 
            (*desc_field) = GrB_REPLACE ;
            (*desc_nondefault) = true ;
        }
        else if (MATCH (s, "gustavson"))
        { 
            (*desc_field) = GxB_AxB_GUSTAVSON ;
            (*desc_nondefault) = true ;
        }
        else if (MATCH (s, "dot"))
        { 
            (*desc_field) = GxB_AxB_DOT ;
            (*desc_nondefault) = true ;
        }
        else if (MATCH (s, "saxpy"))
        { 
            (*desc_field) = GxB_AxB_SAXPY ;
            (*desc_nondefault) = true ;
        }
        else if (MATCH (s, "hash"))
        { 
            (*desc_field) = GxB_AxB_HASH ;
            (*desc_nondefault) = true ;
        }
        else
        { 
            // the string must be one of the strings listed above
            ERROR ("unrecognized descriptor value", GrB_INVALID_VALUE) ;
        }
    }
}

//------------------------------------------------------------------------------
// gbmx_mxarray_to_descriptor
//------------------------------------------------------------------------------

bool gbmx_mxarray_to_descriptor // true if descriptor present in pargin [...]
(
    // output:
    gb_descriptor gbdesc,   // pointer to statically allocated struct
    // input:
    const mxArray *mxdesc   // MATLAB struct with possible descriptor
)
{

    //--------------------------------------------------------------------------
    // check inputs and find the descriptor
    //--------------------------------------------------------------------------

    // set all defaults in the gb_descriptor: all zero except gbdesc->fmt
    ASSERT (gbdesc != NULL) ;
    memset (gbdesc, 0, sizeof (struct gb_descriptor_struct)) ;
    gbdesc->fmt = GxB_NO_FORMAT ;

    if (mxdesc == NULL || !mxIsStruct (mxdesc)
        || (mxGetField (mxdesc, 0, "GraphBLASv10") != NULL)
        || (mxGetField (mxdesc, 0, "GraphBLASv7_3") != NULL)
        || (mxGetField (mxdesc, 0, "GraphBLASv5_1") != NULL)
        || (mxGetField (mxdesc, 0, "GraphBLASv5") != NULL)
        || (mxGetField (mxdesc, 0, "GraphBLASv4") != NULL)
        || (mxGetField (mxdesc, 0, "GraphBLAS") != NULL))
    { 
        // If present, the descriptor is a struct whose first field is not
        // "desc.GraphBLAS*" (a GrB matrix).  If not present, the GraphBLAS
        // descriptor is NULL.
        return (NULL) ;
    }

    gbdesc->is_present = true ;

    //--------------------------------------------------------------------------
    // create the GraphBLAS gb_descriptor
    //--------------------------------------------------------------------------

    // get each component for the GraphBLAS GrB_Descriptor
    get_desc (&(gbdesc->out ), &(gbdesc->nondefault), mxdesc, "out" ) ;
    get_desc (&(gbdesc->in0 ), &(gbdesc->nondefault), mxdesc, "in0" ) ;
    get_desc (&(gbdesc->in1 ), &(gbdesc->nondefault), mxdesc, "in1" ) ;
    get_desc (&(gbdesc->mask), &(gbdesc->nondefault), mxdesc, "mask") ;
    get_desc (&(gbdesc->axb ), &(gbdesc->nondefault), mxdesc, "axb" ) ;

    //--------------------------------------------------------------------------
    // get the desired kind of output: GrB, sparse, or full
    //--------------------------------------------------------------------------

    gbdesc->kind = gbmx_get_kind (mxdesc) ;

    //--------------------------------------------------------------------------
    // get the desired format and sparsity of output, if any
    //--------------------------------------------------------------------------

    mxArray *mxfmt = mxGetField (mxdesc, 0, "format") ;
    if (mxfmt != NULL)
    {
        char mxfmt_string [LEN+2] ;
        gbmx_mxstring_to_string (mxfmt_string, LEN, mxfmt, "format") ;
        bool ok = gb_string_to_format (mxfmt_string,
            &(gbdesc->fmt), NULL,
            &(gbdesc->sparsity), NULL) ;
        CHECK_ERROR (!ok, "unknown format") ;
    }

    //--------------------------------------------------------------------------
    // get the desired base
    //--------------------------------------------------------------------------

    mxArray *mxbase = mxGetField (mxdesc, 0, "base") ;
    if (mxbase != NULL)
    {
        // get the string from the struct field
        char s [LEN+2] ;
        gbmx_mxstring_to_string (s, LEN, mxbase, "base") ;
        if (MATCH (s, "default"))
        { 
            // The indices are one-based integer by default.
            gbdesc->base = BASE_DEFAULT ;
        }
        else if (MATCH (s, "zero-based") || MATCH (s, "zero-based int"))
        { 
            // zero-based indices are always uint64/uint32.  This is the
            // fastest option since GraphBLAS uses zero-based indices.
            gbdesc->base = BASE_0_INT ;
        }
        else if (MATCH (s, "one-based") || MATCH (s, "one-based int"))
        { 
            // one-based indices, but in uint64/uint32 (the default)
            gbdesc->base = BASE_1_INT ;
        }
        else if (MATCH (s, "double") || MATCH (s, "one-based double"))
        { 
            // one-based double indices
            gbdesc->base = BASE_1_DOUBLE ;
        }
        else
        { 
            ERROR ("invalid descriptor.base", GrB_INVALID_VALUE) ;
        }
    }

    //--------------------------------------------------------------------------
    // return results
    //--------------------------------------------------------------------------

    return (true) ;
}

