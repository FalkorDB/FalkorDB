//------------------------------------------------------------------------------
// gbmex_format: get/set the matrix format to use in GraphBLAS
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage

// fmt = gbmex_format ;         get the global default format (row/col)
// fmt = gbmex_format (fmt) ;   set the global default format
// [f,sparsity,iso] = gbmex_format (A) ;  get the format, sparsity,
//                          and iso status of a matrix (GrB, GhB or built-in)

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_ALL
#define FREE_ALL GrB_Matrix_free (&A_to_free) ;

#define USAGE "usage: [f,s,iso] = GrB.format(A), " \
    "f = GrB.format (f), or f = GrB.format"

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

    GrB_Matrix A = NULL, A_to_free = NULL ;
    int arena = GrB_DEFAULT ;   // use default arena for temporary workspace

    GBMX_USAGE (nargin <= 1 && nargout <= 3, USAGE) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [1] ;
    if (nargin == 1 && !mxIsChar (pargin [0]))
    {
        gbmx_get_matrix (&(Matrix [0]), pargin [0]) ;
    }

    ////////////////////////////////////////////////////////////////////////////

    // Calls to mx* and GrB methods are intermingled below.  This usage is safe
    // from memory leaks because (a) the get/set methods do not allocate any
    // memory, and (b) the shallow matrix A is created and then freed before
    // any subsequent mx* methods are used.

    //--------------------------------------------------------------------------
    // get/set the format
    //--------------------------------------------------------------------------

    int fmt = GxB_BY_COL ;
    int sparsity = GxB_AUTO_SPARSITY ;
    int iso = false ;

    if (nargin == 0)
    { 

        //----------------------------------------------------------------------
        // format = GrB.format
        //----------------------------------------------------------------------

        // get the global format
        OK (GrB_Global_get_INT32 (GrB_GLOBAL, &fmt, GxB_FORMAT)) ;

    }
    else // if (nargin == 1)
    {

        if (mxIsChar (pargin [0]))
        { 

            //------------------------------------------------------------------
            // GrB.format (format): set the global format ('by row' or 'by col')
            //------------------------------------------------------------------

            // parse the format string: 'by row' or 'by col' only
            int ignore ;
            bool fmt_present = false ;
            char format_string [LEN+2] ;
            gbmx_mxstring_to_string (format_string, LEN, pargin [0], "format") ;
            gb_string_to_format (format_string, &fmt, &fmt_present,
                /* ignore any sparsity setting: */ &ignore, NULL) ;
            CHECK_ERROR (!fmt_present, "invalid format") ;
            // set the global format
            OK (GrB_Global_set_INT32 (GrB_GLOBAL, fmt, GxB_FORMAT)) ;

        }
        else
        { 

            //------------------------------------------------------------------
            // GrB.format (A)
            //------------------------------------------------------------------

            // The input matrix is freed, so that mx* methods can allocate
            // memory below.  This eliminates any potential memory leaks if A
            // is a handle GhB matrix using malloc/free.

            OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
            OK (GrB_Matrix_get_INT32 (A, &fmt, GxB_FORMAT)) ;
            OK (GrB_Matrix_get_INT32 (A, &sparsity, GxB_SPARSITY_STATUS)) ;
            OK (GrB_Matrix_get_INT32 (A, &iso, GxB_ISO)) ;
            FREE_ALL ;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

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

    gb_wrapup ( ) ;
}

