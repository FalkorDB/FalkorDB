//------------------------------------------------------------------------------
// gbmx_mxstring_to_string: copy a built-in string into a C string
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The string is converted to lower case, so that all input strings to the
// SuiteSparse:GraphBLAS interface are case-insensitive.

void gbmx_mxstring_to_string  // copy a built-in string into a C string
(
    // output:
    char *string,           // size at least maxlen+1
    // input:
    const size_t maxlen,    // length of string
    const mxArray *S,       // built-in mxArray containing a string
    const char *name        // name of the mxArray
)
{

    size_t len = 0 ;
    string [0] = '\0' ;
    if (S != NULL && mxGetNumberOfElements (S) > 0)
    {
        if (!mxIsChar (S))
        { 
            ERROR2 ("%s must be a string", name, GrB_DOMAIN_MISMATCH) ;
        }
        len = mxGetNumberOfElements (S) ;
        if (len > 0)
        {
            mxGetString (S, string, maxlen) ;
            string [maxlen] = '\0' ;
            // convert the string to lower case
            for (int k = 0 ; k < maxlen && string [k] != '\0' ; k++)
            { 
                string [k] = tolower (string [k]) ;
            }
        }
    }
}

