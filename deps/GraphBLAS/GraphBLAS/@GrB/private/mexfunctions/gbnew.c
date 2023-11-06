//------------------------------------------------------------------------------
// gbnew: create a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A may be a built-in sparse matrix, or a built-in struct containing a
// GraphBLAS matrix.  C is returned as a built-in struct containing a GraphBLAS
// matrix.

// Usage:

// C = gbnew (A)
// C = gbnew (A, type)
// C = gbnew (A, format)
// C = gbnew (m, n)
// C = gbnew (m, n, format)
// C = gbnew (m, n, type)
// C = gbnew (A, type, format)
// C = gbnew (A, format, type)
// C = gbnew (m, n, type, format)
// C = gbnew (m, n, format, type)

#include "gb_interface.h"

#define USAGE "usage: C = GrB (m,n,type,format) or C = GrB (A,type,format)"

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

    gb_usage (nargin >= 1 && nargin <= 4 && nargout <= 1, USAGE) ;

    //--------------------------------------------------------------------------
    // construct the GraphBLAS matrix
    //--------------------------------------------------------------------------

    GrB_Matrix C ;
    GxB_Format_Value fmt ;
    int sparsity = 0 ;

    if (nargin == 1)
    { 

        //----------------------------------------------------------------------
        // C = GrB (A)
        //----------------------------------------------------------------------

        // GraphBLAS copy of A, same type and format as A
        C = gb_get_deep (pargin [0]) ;

    }
    else if (nargin == 2)
    {

        //----------------------------------------------------------------------
        // C = GrB (A, type)
        // C = GrB (A, format)
        // C = GrB (m, n)
        //----------------------------------------------------------------------

        if (mxIsChar (pargin [1]))
        {

            //------------------------------------------------------------------
            // C = GrB (A, type)
            // C = GrB (A, format)
            //------------------------------------------------------------------

            GrB_Type type = gb_mxstring_to_type (pargin [1]) ;
            bool ok = gb_mxstring_to_format (pargin [1], &fmt, &sparsity) ;

            if (type != NULL)
            {

                //--------------------------------------------------------------
                // C = GrB (A, type)
                //--------------------------------------------------------------

                if (gb_mxarray_is_empty (pargin [0]))
                { 
                    // A is a 0-by-0 built-in matrix.  create a new 0-by-0
                    // GraphBLAS matrix C of the given type, with the default
                    // format.
                    C = gb_new (type, 0, 0, -1, 0) ;
                }
                else
                { 
                    // get a shallow copy and then typecast it to type.
                    // use the same format as A
                    GrB_Matrix A = gb_get_shallow (pargin [0]) ;
                    OK (GxB_Matrix_Option_get (A, GxB_FORMAT, &fmt)) ;
                    C = gb_typecast (A, type, fmt, 0) ;
                    OK (GrB_Matrix_free (&A)) ;
                }

            }
            else if (ok)
            { 

                //--------------------------------------------------------------
                // C = GrB (A, format)
                //--------------------------------------------------------------

                // get a shallow copy of A
                GrB_Matrix A = gb_get_shallow (pargin [0]) ;
                // C = A with the requested format and sparsity, no typecast
                C = gb_typecast (A, NULL, fmt, sparsity) ;
                OK (GrB_Matrix_free (&A)) ;

            }
            else
            { 
                ERROR ("unknown type or format") ;
            }

        }
        else if (gb_mxarray_is_scalar (pargin [0]) &&
                 gb_mxarray_is_scalar (pargin [1]))
        { 

            //------------------------------------------------------------------
            // C = GrB (m, n)
            //------------------------------------------------------------------

            // m-by-n GraphBLAS double matrix, no entries, default format
            GrB_Index nrows = gb_mxget_uint64_scalar (pargin [0], "m") ;
            GrB_Index ncols = gb_mxget_uint64_scalar (pargin [1], "n") ;
            C = gb_new (GrB_FP64, nrows, ncols, -1, 0) ;

        }
        else
        { 
            ERROR ("usage: C=GrB(m,n), C=GrB(A,type), or C=GrB(A,format)") ;
        }

    }
    else if (nargin == 3)
    {

        //----------------------------------------------------------------------
        // C = GrB (m, n, format)
        // C = GrB (m, n, type)
        // C = GrB (A, type, format)
        // C = GrB (A, format, type)
        //----------------------------------------------------------------------

        if (gb_mxarray_is_scalar (pargin [0]) &&
            gb_mxarray_is_scalar (pargin [1]) && mxIsChar (pargin [2]))
        {

            //------------------------------------------------------------------
            // C = GrB (m, n, format)
            // C = GrB (m, n, type)
            //------------------------------------------------------------------

            // create an m-by-n matrix with no entries
            GrB_Index nrows = gb_mxget_uint64_scalar (pargin [0], "m") ;
            GrB_Index ncols = gb_mxget_uint64_scalar (pargin [1], "n") ;
            GrB_Type type = gb_mxstring_to_type (pargin [2]) ;
            bool ok = gb_mxstring_to_format (pargin [2], &fmt, &sparsity) ;

            if (type != NULL)
            { 
                // create an m-by-n matrix of the desired type, no entries,
                // use the default format.
                C = gb_new (type, nrows, ncols, -1, sparsity) ;
            }
            else if (ok)
            { 
                // create an m-by-n double matrix of the desired format
                C = gb_new (GrB_FP64, nrows, ncols, fmt, sparsity) ;
            }
            else
            { 
                ERROR ("unknown type or format") ;
            }

        }
        else if (mxIsChar (pargin [1]) && mxIsChar (pargin [2]))
        {

            //------------------------------------------------------------------
            // C = GrB (A, type, format)
            // C = GrB (A, format, type)
            //------------------------------------------------------------------

            GrB_Type type = gb_mxstring_to_type (pargin [1]) ;
            bool ok = gb_mxstring_to_format (pargin [2], &fmt, &sparsity) ;

            if (ok)
            { 
                // C = GrB (A, type, format)
            }
            else
            { 
                // C = GrB (A, format, type)
                ok = gb_mxstring_to_format (pargin [1], &fmt, &sparsity) ;
                type = gb_mxstring_to_type (pargin [2]) ;
            }

            if (type == NULL || !ok)
            { 
                ERROR ("unknown type and/or format") ;
            }

            if (gb_mxarray_is_empty (pargin [0]))
            { 
                C = gb_new (type, 0, 0, fmt, sparsity) ;
            }
            else
            { 
                // get a shallow copy, typecast it, and set the format
                GrB_Matrix A = gb_get_shallow (pargin [0]) ;
                C = gb_typecast (A, type, fmt, sparsity) ;
                OK (GrB_Matrix_free (&A)) ;
            }
        }
        else
        { 
            ERROR ("unknown usage") ;
        }

    }
    else // if (nargin == 4)
    {

        //----------------------------------------------------------------------
        // C = GrB (m, n, type, format)
        // C = GrB (m, n, format, type)
        //----------------------------------------------------------------------

        if (gb_mxarray_is_scalar (pargin [0]) &&
            gb_mxarray_is_scalar (pargin [1]) &&
            mxIsChar (pargin [2]) && mxIsChar (pargin [3]))
        {

            // create an m-by-n matrix with no entries, of the requested
            // type and format
            GrB_Index nrows = gb_mxget_uint64_scalar (pargin [0], "m") ;
            GrB_Index ncols = gb_mxget_uint64_scalar (pargin [1], "n") ;

            GrB_Type type = gb_mxstring_to_type (pargin [2]) ;
            bool ok = gb_mxstring_to_format (pargin [3], &fmt, &sparsity) ;

            if (ok)
            { 
                // C = GrB (m, n, type, format)
            }
            else
            { 
                // C = GrB (m, n, format, type)
                ok = gb_mxstring_to_format (pargin [2], &fmt, &sparsity) ;
                type = gb_mxstring_to_type (pargin [3]) ;
            }

            if (type == NULL || !ok)
            { 
                ERROR ("unknown type and/or format") ;
            }

            C = gb_new (type, nrows, ncols, fmt, sparsity) ;
        }
        else
        { 
            ERROR ("unknown usage") ;
        }
    }

    //--------------------------------------------------------------------------
    // export the output matrix C as a GraphBLAS matrix
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, KIND_GRB) ;
    GB_WRAPUP ;
}

