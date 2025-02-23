//------------------------------------------------------------------------------
// GB_mx_mxArray_to_indices: get a list of indices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Get a list of indices from a MATLAB array or struct with I.begin, or
// a MATLAB struct holding a GraphBLAS GrB_Vector.

#include "GB_mex.h"

bool GB_mx_mxArray_to_indices       // true if successful, false otherwise
(
    // input:
    const mxArray *I_builtin,       // built-in mxArray to get
    // output:
    uint64_t **I_handle,            // index array
    uint64_t *ni,                   // length of I, or special
    uint64_t Icolon [3],            // for all but GB_LIST
    bool *I_is_list,                // true if GB_LIST
    GrB_Vector *I_vector            // non-NULL if found
)
{

    (*I_handle) = NULL ;
    if (I_vector != NULL)
    {
        (*I_vector) = NULL ;
    }

    mxArray *X ;
    uint64_t *I ;

    if (I_builtin == NULL || mxIsEmpty (I_builtin))
    {

        //----------------------------------------------------------------------
        // I is NULL or empty; treat as ":"
        //----------------------------------------------------------------------

        I = (uint64_t *) GrB_ALL ;       // like the ":" in C=A(:,j)
        (*ni) = 0 ;
        (*I_handle) = I ;
        (*I_is_list) = false ;
        // Icolon not used

    }
    else if (mxIsStruct (I_builtin) &&
        mxGetFieldNumber (I_builtin, "begin") >= 0)
    {

        //----------------------------------------------------------------------
        // I is a struct with I.begin
        //----------------------------------------------------------------------

        // a struct with 3 integers: I.begin, I.inc, I.end
        (*I_is_list) = false ;

        // look for I.begin (required)
        int fieldnumber = mxGetFieldNumber (I_builtin, "begin") ;
        if (fieldnumber < 0)
        {
            mexWarnMsgIdAndTxt ("GB:warn","I.begin missing") ;
            return (false) ;
        }
        X = mxGetFieldByNumber (I_builtin, 0, fieldnumber) ;
        Icolon [GxB_BEGIN] = (int64_t) mxGetScalar (X) ;

        // look for I.end (required)
        fieldnumber = mxGetFieldNumber (I_builtin, "end") ;
        if (fieldnumber < 0)
        {
            mexWarnMsgIdAndTxt ("GB:warn","I.end missing") ;
            return (false) ;
        }
        mxArray *X ;
        X = mxGetFieldByNumber (I_builtin, 0, fieldnumber) ;
        Icolon [GxB_END] = (int64_t) mxGetScalar (X) ;

        // look for I.inc (optional)
        fieldnumber = mxGetFieldNumber (I_builtin, "inc") ;
        if (fieldnumber < 0)
        {
            (*ni) = GxB_RANGE ;
            Icolon [GxB_INC] = 1 ;
        }
        else
        {
            X = mxGetFieldByNumber (I_builtin, 0, fieldnumber) ;
            int64_t iinc = (int64_t) mxGetScalar (X) ;
            if (iinc == 0)
            {
                // this can be either a stride, or backwards.  Either 
                // one works the same, but try a mixture, just for testing.
                (*ni) = (Icolon [GxB_BEGIN] % 2) ?
                    GxB_STRIDE : GxB_BACKWARDS ;
                Icolon [GxB_INC] = 0 ;
            }
            else if (iinc > 0)
            {
                (*ni) = GxB_STRIDE ;
                Icolon [GxB_INC] = iinc ;
            }
            else
            {
                // GraphBLAS must be given the magnitude of the stride
                (*ni) = GxB_BACKWARDS ;
                Icolon [GxB_INC] = -iinc ;
            }
        }
        (*I_handle) = Icolon ;

    }
    else if (mxIsClass (I_builtin, "uint64") && I_vector == NULL)
    {

        //----------------------------------------------------------------------
        // I is a built-in array of type uint64
        //----------------------------------------------------------------------

        (*I_is_list) = true ;
        I = mxGetData (I_builtin) ;
        (*ni) = (uint64_t) mxGetNumberOfElements (I_builtin) ;
        (*I_handle) = I ;

    }
    else if (I_vector != NULL)
    {

        //----------------------------------------------------------------------
        // I could be a GrB_Vector held as a struct or MATLAB matrix
        //----------------------------------------------------------------------

        (*I_vector) = GB_mx_mxArray_to_Vector (I_builtin, "I_vector",
            true, true) ;
        if ((*I_vector) == NULL)
        {
            mexWarnMsgIdAndTxt ("GB:warn", "indices invalid") ;
            return (false) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // I not recognized
        //----------------------------------------------------------------------

        mexWarnMsgIdAndTxt ("GB:warn", "indices invalid") ;
        return (false) ;
    }

    return (true) ;
}

