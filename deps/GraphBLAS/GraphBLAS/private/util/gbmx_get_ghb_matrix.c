//------------------------------------------------------------------------------
// gbmx_get_ghb_matrix: get a GhB handle matrix argument
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Returns NULL if G is not a GhB handle object from GraphBLAS 10.4.0 or
// later, or its G.opaque content.

GrB_Matrix gbmx_get_ghb_matrix  // the content of a MATLAB GhB handle object
(
    // input
    const mxArray *G            // must be a GhB object
)
{

    //--------------------------------------------------------------------------
    // get the GrB_Matrix
    //--------------------------------------------------------------------------

    mxArray *G_opaque = gbmx_get_ghb_handle (G) ;
    GrB_Matrix C = NULL ;
    if (G_opaque != NULL)
    {
        C = (*((GrB_Matrix *) mxGetData (G_opaque))) ;
    }
    return (C) ;
}

