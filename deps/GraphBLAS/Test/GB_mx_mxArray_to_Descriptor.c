//------------------------------------------------------------------------------
// GB_mx_mxArray_to_Descriptor: get the contents of a GraphBLAS Descriptor
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// get a GraphBLAS descriptor from a built-in struct.
// D.out, D.in0, D.in0, and D.mask must all be strings.
// Fields that are not present are left unchanged.

#include "GB_mex.h"

static bool get_descriptor
(
    GrB_Descriptor D,               // GraphBLAS descriptor to modify
    const mxArray *D_builtin,       // built-in struct with D.output, etc
    const char *fieldname,          // fieldname to extract from D_builtin
    const GrB_Desc_Field field      // field to set in D
)
{

    // A NULL means the descriptor is not in the list of input parameters.
    // An empty Descriptor is OK.  Both mean the GraphBLAS descriptor is NULL,
    // which each method uses to denote default values for all parameters.
    if (D_builtin == NULL || mxIsEmpty (D_builtin))
    {
        return (true) ;
    }

    // if present, the built-in D must be a struct
    if (!mxIsStruct (D_builtin))
    {
        mexWarnMsgIdAndTxt ("GB:warn","descriptor must be a struct") ;
        return (false) ;
    }

    // find the field in the built-in struct
    int fieldnumber = mxGetFieldNumber (D_builtin, fieldname) ;
    if (fieldnumber >= 0)
    {
        // the field is present
        mxArray *value = mxGetFieldByNumber (D_builtin, 0, fieldnumber) ;

        // its value must be a string
        if (!mxIsChar (value))
        {
            mexWarnMsgIdAndTxt ("GB:warn","D.field must be a string") ;
            return (false) ;
        }

        // get the string from the built-in field
        #define LEN 100
        char s [LEN] ;
        mxGetString (value, s, LEN) ;

        // convert the string to a Descriptor value, and set the value
        GrB_Info info ;
        if (MATCH (s, "default"))
        {
            info = GxB_Desc_set (D, field, GxB_DEFAULT) ;
        }
        else if (MATCH (s, "tran") || MATCH (s, "transpose"))
        {
            info = GxB_Desc_set (D, field, GrB_TRAN) ;
        }
        else if (MATCH (s, "complement"))
        {
            info = GxB_Desc_set (D, field, GrB_COMP) ;
        }
        else if (MATCH (s, "structure") || MATCH (s, "structural"))
        {
            info = GxB_Desc_set (D, field, GrB_STRUCTURE) ;
        }
        else if (MATCH (s, "structural complement"))
        {
            info = GxB_Desc_set (D, field, GrB_COMP + GrB_STRUCTURE) ;
        }
        else if (MATCH (s, "replace"))
        {
            info = GxB_Desc_set (D, field, GrB_REPLACE) ;
        }
        else if (MATCH (s, "saxpy"))
        {
            info = GxB_Desc_set (D, field, GxB_AxB_SAXPY) ;
        }
        else if (MATCH (s, "gustavson"))
        {
            info = GxB_Desc_set (D, field, GxB_AxB_GUSTAVSON) ;
        }
        else if (MATCH (s, "dot"))
        {
            info = GxB_Desc_set (D, field, GxB_AxB_DOT) ;
        }
        else if (MATCH (s, "hash"))
        {
            info = GxB_Desc_set (D, field, GxB_AxB_HASH) ;
        }
        else
        {
            // the string must be one of the four strings listed above
            mexWarnMsgIdAndTxt ("GB:warn","unrecognized descriptor value") ;
            return (false) ;
        }

        // make sure the Descriptor field was set OK
        if (info != GrB_SUCCESS)
        {
            mexWarnMsgIdAndTxt ("GB:warn", "set D failed!") ;
            return (false) ;
        }
    }
    return (true) ;
}

//------------------------------------------------------------------------------

bool GB_mx_mxArray_to_Descriptor   // true if successful, false otherwise
(
    GrB_Descriptor *handle,         // descriptor to return
    const mxArray *D_builtin,       // built-in struct
    const char *name                // name of the descriptor
)
{
    // a null descriptor is OK; the method will use defaults
    (*handle) = NULL ;
    if (D_builtin == NULL || mxIsEmpty (D_builtin))
    {
        return (true) ;
    }

    // the built-in desc is present and not empty, so create the GraphBLAS one
    GrB_Descriptor D = NULL ;
    GrB_Info info = GrB_Descriptor_new (&D) ;
    if (info != GrB_SUCCESS)
    {
        GrB_Matrix_free_(&D) ;
        mexWarnMsgIdAndTxt ("GB:warn", "new D failed") ;
        return (false) ;
    }

    // get each component of the descriptor struct
    if (!get_descriptor (D, D_builtin, "outp", GrB_OUTP) ||
        !get_descriptor (D, D_builtin, "inp0", GrB_INP0) ||
        !get_descriptor (D, D_builtin, "inp1", GrB_INP1) ||
        !get_descriptor (D, D_builtin, "mask", GrB_MASK) ||
        !get_descriptor (D, D_builtin, "axb",  GxB_AxB_METHOD))
    {
        GrB_Matrix_free_(&D) ;
        mexWarnMsgIdAndTxt ("GB:warn", "descriptor failed") ;
        return (false) ;
    }

    // return the non-null Descriptor to the caller
    ASSERT_DESCRIPTOR_OK (D, name, GB0) ;
    (*handle) = D ;
    return (true) ;
}

