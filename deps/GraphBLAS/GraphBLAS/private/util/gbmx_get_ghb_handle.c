//------------------------------------------------------------------------------
// gbmx_get_ghb_handle: get a GhB matrix from a struct/object, as a handle
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input to this method is an mxArray G, which must either be a GhB object,
// or the G.opaque struct content of a GhB object.  The output is an mxArray
// containing the G.opaque handle to the GrB_Matrix that the GhB object holds.
// Returns NULL if the input is not a GhB handle object from GraphBLAS v10.4.0
// or later.  Since this method is used by gbmex_delete, it cannot throw an
// error.

mxArray *gbmx_get_ghb_handle    // the MATLAB GhB opaque handle
(
    // input
    const mxArray *G            // must be a GhB object
)
{

    //--------------------------------------------------------------------------
    // get the GrB_Matrix handle
    //--------------------------------------------------------------------------

    mxArray *G_opaque = NULL ;

    if (G != NULL && mxIsClass (G, "GhB"))
    { 
        // G is a GhB object; get its opaque content (which must be a struct).
        // This is very fast since the opaque property is only 8 bytes in size.
        G = mxGetProperty (G, 0, "opaque") ;
    }

    if (G != NULL && mxIsStruct (G) && mxGetNumberOfFields (G) == 1 &&
        mxGetNumberOfElements (G) == 1)
    { 
        // G is a single struct with a single field, which must come from the
        // opaque content of a GhB object: a uint8 array of size 1-by-8.
        G_opaque = mxGetFieldByNumber (G, 0, 0) ;
        if (! (mxGetM (G_opaque) == 1 &&
               mxGetN (G_opaque) == sizeof (GrB_Matrix) &&
               mxGetClassID (G_opaque) == mxUINT8_CLASS))
        {
            return (NULL) ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    return (G_opaque) ;
}

