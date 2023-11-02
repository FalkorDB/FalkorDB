//------------------------------------------------------------------------------
// GB_mx_mxArray_to_indices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Get a list of indices from a built-in array

#include "GB_mex.h"

bool GB_mx_mxArray_to_indices       // true if successful, false otherwise
(
    GrB_Index **handle,             // index array returned
    const mxArray *I_builtin,       // built-in mxArray to get
    GrB_Index *ni,                  // length of I, or special
    GrB_Index Icolon [3],           // for all but GB_LIST
    bool *I_is_list                 // true if GB_LIST
)
{

    (*handle) = NULL ;

    mxArray *X ;
    GrB_Index *I ;

    if (I_builtin == NULL || mxIsEmpty (I_builtin))
    {
        I = (GrB_Index *) GrB_ALL ;       // like the ":" in C=A(:,j)
        (*ni) = 0 ;
        (*handle) = I ;
        (*I_is_list) = false ;
        // Icolon not used
    }
    else
    {
        if (mxIsStruct (I_builtin))
        {
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
                // 
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
            (*handle) = Icolon ;

        }
        else
        {
            if (!mxIsClass (I_builtin, "uint64"))
            {
                mexWarnMsgIdAndTxt ("GB:warn","indices must be uint64") ;
                return (false) ;
            }

            (*I_is_list) = true ;
            I = mxGetData (I_builtin) ;
            (*ni) = (uint64_t) mxGetNumberOfElements (I_builtin) ;
            (*handle) = I ;
        }
    }

    return (true) ;
}

