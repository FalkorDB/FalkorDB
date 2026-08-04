//------------------------------------------------------------------------------
// gbmx_get_mxargs: get input arguments to a GraphBLAS mexFunction 
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmx_get_mxargs collects all the input arguments for the GraphBLAS
// mexFunctions.  The user-level view is described below.  For the private
// mexFunctions, the descriptor optionally appears as the last argument.  The
// matrix arguments are either built-in sparse or full matrices, GraphBLAS
// matrices.

void gbmx_get_mxargs
(
    // input:
    int nargin,                 // # inputs for mexFunction (must be > 0)
    const mxArray *pargin [ ],  // input arguments for mexFunction
    const char *usage,          // usage to print, if too many args appear
    // output:
    struct gb_matrix_struct Matrix [6], // matrix arguments
    int *nmatrices,             // # of matrix arguments
    char String [2][LEN+2],     // string arguments
    int *nstrings,              // # of string arguments
    mxArray *Cell [2],          // cell array arguments
    int *ncells,                // # of cell array arguments
    gb_descriptor gbdesc        // gb_descriptor struct
)
{

    //--------------------------------------------------------------------------
    // find the descriptor (always the last argument, if present)
    //--------------------------------------------------------------------------

    ASSERT (gbdesc != NULL) ;
    ASSERT (nargin > 1) ;
    gbmx_mxarray_to_descriptor (gbdesc, pargin [nargin-1]) ;
    if (gbdesc->is_present)
    { 
        // descriptor is present, remove it from further consideration
        nargin-- ;
    }
    ASSERT (nargin > 1) ;

    //--------------------------------------------------------------------------
    // find the remaining arguments
    //--------------------------------------------------------------------------

    (*nmatrices) = 0 ;
    (*nstrings) = 0 ;
    (*ncells) = 0 ;
    String [0][0] = '\0' ;
    String [1][0] = '\0' ;

    for (int k = 1 ; k < nargin ; k++)
    {
        if (mxIsCell (pargin [k]))
        {
            // I or J index arguments
            if ((*ncells) >= 2)
            { 
                ERROR ("only 2D indexing is supported", GrB_INVALID_VALUE) ;
            }
            Cell [(*ncells)++] = (mxArray *) pargin [k] ;
        }
        else if (mxIsChar (pargin [k]))
        {
            // accum operator, unary op, binary op, monoid, semiring, or
            // other string parameter.
            if ((*nstrings) >= 2)
            { 
                // No mexFunction requires more than 2 input strings
                ERROR (usage, GrB_INVALID_VALUE) ;
            }
            // copy the MATLAB string into the char String array
            gbmx_mxstring_to_string (&(String [*nstrings][0]), LEN, pargin [k],
                "arg") ;
            (*nstrings)++ ;
        }
        else
        {
            // a matrix argument is C, M, A, or B
            if ((*nmatrices) >= 6)
            { 
                // at most 6 matrix inputs are allowed
                ERROR (usage, GrB_INVALID_VALUE) ;
            }
            gbmx_get_matrix (&(Matrix [*nmatrices]), pargin [k]) ;
            (*nmatrices)++ ;
        }
    }
}

