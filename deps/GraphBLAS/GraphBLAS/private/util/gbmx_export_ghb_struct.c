//------------------------------------------------------------------------------
// gbmx_export_ghb_mxstruct: construct pargout [arg] for a GhB matrix handle
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Creates an output GhB argument for a mexFunction.  This is done at the
// start of a mexFunction that needs to return a G.opaque handle, so that if it
// fails, no memory is leaked by subsequent calls to GraphBLAS in the
// mexFunction.  It is not needed for the GrB value matrix object.

/* usage:

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    // at start of a typical mexFunction with [C,kind] output arguments:

    gbmx_usage (...) ;
    GrB_Matrix *C_opaque ;
    pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;
    pargout [1] = mxCreateDoubleScalar (0) ;
    double *kind_output = (double *) mxGetData (pargout [1]) ;

    // The 3 regions of a mexFunction are delineated with a "////...//"
    // comment.  Any mex* or mx* method can be used before the first
    // "////...///" line, drawn below, because on malloc/free has yet been
    // used.  gbmx_* utility methods can be safely used here.

    // If any call to mxMalloc fails, the mexFunction immediately returns and
    // frees all mxMalloc'd space.  It does not free malloc'd space, so in
    // this region, an mxMalloc failure will not result in a memory leak.

    // Only GrB* methods that do not allocate any memory can be used in this
    // section, with few exceptions (see gbmx_usage, which calls GrB_init).

    ////////////////////////////////////////////////////////////////////////////

    // No mex* or mx* method can be used in this region of the code, because
    // if they fail, they do not return to this mexFunction.  Instead, any
    // mxMalloc'd space is freed and control is immediately returned back to
    // MATLAB m-file that called this mexFunction.  However, this region of
    // code allocates memory with malloc/free in the default arena 0, and any
    // such error handling would leave them unfree'd, resulting in a memory
    // leak.  All GrB* methods and gb_* utility methods can be used here.

    // ... This mexFunction creates the output GrB_Matrix C

    // at end of mexFunction:

    OK (gb_export (C_opaque, &C, gbdesc.kind, ghb, err)) ;
    (*kind_output) = (double) gbdesc.kind ;

    // gb_wrapup technically relies on mx* methods, but it is used in testing
    // and test coverage only, not in production.  In production, gb_wrapup()
    // is an empty macro:
    gb_wrapup ( ) ;

    ////////////////////////////////////////////////////////////////////////////

    // Only a few mex* and mx* methods can be safely used after the second
    // "////...///" line.  All malloc'd space has been freed, except for the
    // output matrix C.  If the mexFunction fails here, C is not yet a fully-
    // formed GhB object, since pargout [0] contains just the C.opaque
    // content.  If an mx* or mex* method fails here, the destructor for C in
    // GhB.m will not be called.

    // As a result, only mx* and mex* methods that can never fail can be used
    // here.  Most mexFunctions do not need this section of code.
}
*/

static const char *fields [1] = { "opaque" } ;

mxArray *gbmx_export_ghb_mxstruct   // construct an mxArray struct for GhB
(
    GrB_Matrix **C_opaque_handle
)
{ 
    mxArray *C_struct = mxCreateStructMatrix (1, 1, 1, fields) ;
    mxArray *C_opaque = mxCreateNumericMatrix (1, sizeof (GrB_Matrix),
        mxUINT8_CLASS, mxREAL) ;
    mxSetFieldByNumber (C_struct, 0, 0, C_opaque) ;
    (*C_opaque_handle) = (GrB_Matrix *) mxGetData (C_opaque) ;
    return (C_struct) ;
}

