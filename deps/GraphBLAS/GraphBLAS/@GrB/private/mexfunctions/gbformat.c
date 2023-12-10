//------------------------------------------------------------------------------
// gbformat: get/set the matrix format to use in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage

// fmt = gbformat ;                   get the global default format (row/col)
// fmt = gbformat (fmt) ;             set the global default format
// [f,sparsity,iso] = gbformat (G) ;  get the format, sparsity, and iso status
//                                    of a matrix (either @GrB or built-in)

#include "gb_interface.h"

#define USAGE "usage: [f,s,iso] = GrB.format(G), f = GrB.format (f), or f = GrB.format"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin <= 1 && nargout <= 3, USAGE) ;

    //--------------------------------------------------------------------------
    // get/set the format
    //--------------------------------------------------------------------------

    GxB_Format_Value fmt = GxB_BY_COL ;
    int sparsity = GxB_AUTO_SPARSITY ;
    bool iso = false ;
    bool v5_1_or_later = false ;

    if (nargin == 0)
    { 

        //----------------------------------------------------------------------
        // format = GrB.format
        //----------------------------------------------------------------------

        // get the global format
        OK (GxB_Global_Option_get (GxB_FORMAT, &fmt)) ;

    }
    else // if (nargin == 1)
    {

        if (mxIsChar (pargin [0]))
        { 


            //------------------------------------------------------------------
            // GrB.format (format)
            //------------------------------------------------------------------

            // parse the format string
            int ignore ;
            bool ok = gb_mxstring_to_format (pargin [0], &fmt, &ignore) ;
            CHECK_ERROR (!ok, "invalid format") ;
            // set the global format
            OK (GxB_Global_Option_set (GxB_FORMAT, fmt)) ;

        }
        else if (mxIsStruct (pargin [0]))
        { 

            //------------------------------------------------------------------
            // GrB.format (G) for a GraphBLAS matrix G
            //------------------------------------------------------------------

            // get the type
            mxArray *mx_type = mxGetField (pargin [0], 0, "GraphBLASv7_3") ;
            if (mx_type == NULL)
            {
                // check if it is a GraphBLASv5_1 struct
                mx_type = mxGetField (pargin [0], 0, "GraphBLASv5_1") ;
            }
            if (mx_type != NULL)
            {
                // v5_1 or v7_3
                v5_1_or_later = true ;
            }
            if (mx_type == NULL)
            {
                // check if it is a GraphBLASv5 struct
                mx_type = mxGetField (pargin [0], 0, "GraphBLASv5") ;
            }
            if (mx_type == NULL)
            {
                // check if it is a GraphBLASv4 struct
                mx_type = mxGetField (pargin [0], 0, "GraphBLASv4") ;
            }
            if (mx_type == NULL)
            {
                // check if it is a GraphBLASv3 struct
                mx_type = mxGetField (pargin [0], 0, "GraphBLAS") ;
            }
            CHECK_ERROR (mx_type == NULL, "invalid GraphBLAS struct") ;

            // get the row/column format of the input matrix G
            mxArray *opaque = mxGetField (pargin [0], 0, "s") ;
            CHECK_ERROR (opaque == NULL, "invalid GraphBLAS struct") ;
            // use mxGetData (best for Octave, fine for MATLAB)
            int64_t *s = (int64_t *) mxGetData (opaque) ;
            bool is_csc = (bool) (s [6]) ;
            fmt = (is_csc) ? GxB_BY_COL : GxB_BY_ROW ;
            iso = (v5_1_or_later) ? ((bool) s [9]) : false ;

            // get the current sparsity status of the input matrix G
            switch (mxGetNumberOfFields (pargin [0]))
            {
                case 3 : sparsity = GxB_FULL ;        break ;
                case 4 : sparsity = GxB_BITMAP ;      break ;
                case 5 : sparsity = GxB_SPARSE ;      break ;
                case 9 : // fall through to hypersparse
                case 6 : sparsity = GxB_HYPERSPARSE ; break ;
                default: ERROR ("invalid GraphBLAS struct") ;
            }

        }
        else
        { 

            //------------------------------------------------------------------
            // GrB.format (A) for a built-in matrix A
            //------------------------------------------------------------------

            // built-in matrices are always stored by column
            fmt = GxB_BY_COL ;
            // built-in matrices are sparse or full, never hypersparse or bitmap
            sparsity = mxIsSparse (pargin [0]) ? GxB_SPARSE : GxB_FULL ;

        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    pargout [0] = mxCreateString ((fmt == GxB_BY_ROW) ? "by row" : "by col") ;
    if (nargout > 1)
    { 
        char *s ;
        switch (sparsity)
        {
            case GxB_HYPERSPARSE : s = "hypersparse" ; break ;
            case GxB_SPARSE :      s = "sparse"      ; break ;
            case GxB_BITMAP :      s = "bitmap"      ; break ;
            case GxB_FULL :        s = "full"        ; break ;
            default :              s = ""            ; break ;
        }
        pargout [1] = mxCreateString (s) ;
    }
    if (nargout > 2)
    { 
        pargout [2] = mxCreateString (iso ? "iso-valued" : "non-iso-valued") ;
    }

    GB_WRAPUP ;
}

