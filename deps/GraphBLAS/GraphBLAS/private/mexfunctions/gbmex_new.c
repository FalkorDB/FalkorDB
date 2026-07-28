//------------------------------------------------------------------------------
// gbmex_new: create a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A may be a built-in sparse matrix, or a built-in struct containing a
// GraphBLAS matrix.  C is returned as a built-in struct containing a GraphBLAS
// matrix.

// Usage:

// C = gbmex_new (ghb, A)
// C = gbmex_new (ghb, A, type)
// C = gbmex_new (ghb, A, format)
// C = gbmex_new (ghb, m, n)
// C = gbmex_new (ghb, m, n, format)
// C = gbmex_new (ghb, m, n, type)
// C = gbmex_new (ghb, A, type, format)
// C = gbmex_new (ghb, A, format, type)
// C = gbmex_new (ghb, m, n, type, format)
// C = gbmex_new (ghb, m, n, format, type)

#include "gb_interface.h"
#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&A_to_free) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

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
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Matrix *C_opaque = NULL, C = NULL, A = NULL, A_to_free = NULL ;

    GBMX_USAGE (nargin >= 2 && nargin <= 5 && nargout <= 1, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    char string_1 [LEN+2] ;
    char string_2 [LEN+2] ;
    string_1 [0] = '\0' ;
    string_2 [0] = '\0' ;

    bool nargin_2_is_char = false ;
    bool nargin_2_is_mn = false ;
    bool nargin_3_first_case = false ;

    uint64_t nrows = 0, ncols = 0 ;

    struct gb_matrix_struct Matrix [1] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [1]) ;

    if (nargin == 2)
    { 

        //----------------------------------------------------------------------
        // C = GrB (A)
        //----------------------------------------------------------------------

    }
    else if (nargin == 3)
    { 

        //----------------------------------------------------------------------
        // C = GrB (A, type)
        // C = GrB (A, format)
        // C = GrB (m, n)
        //----------------------------------------------------------------------

        if (mxIsChar (pargin [2]))
        { 

            //------------------------------------------------------------------
            // C = GrB (A, type)
            // C = GrB (A, format)
            //------------------------------------------------------------------

            nargin_2_is_char = true ;
            gbmx_mxstring_to_string (string_1, LEN, pargin [2], "") ;

        }
        else if (gbmx_mxarray_is_scalar (pargin [1]) &&
                 gbmx_mxarray_is_scalar (pargin [2]))
        { 

            //------------------------------------------------------------------
            // C = GrB (m, n)
            //------------------------------------------------------------------

            nargin_2_is_mn = true ;
            nrows = gbmx_get_uint64_scalar (pargin [1], "m") ;
            ncols = gbmx_get_uint64_scalar (pargin [2], "n") ;
        }

    }
    else if (nargin == 4)
    { 

        //----------------------------------------------------------------------
        // C = GrB (m, n, format)
        // C = GrB (m, n, type)
        // C = GrB (A, type, format)
        // C = GrB (A, format, type)
        //----------------------------------------------------------------------

        if (gbmx_mxarray_is_scalar (pargin [1]) &&
            gbmx_mxarray_is_scalar (pargin [2]) && mxIsChar (pargin [3]))
        { 

            //------------------------------------------------------------------
            // C = GrB (m, n, format)
            // C = GrB (m, n, type)
            //------------------------------------------------------------------

            nargin_3_first_case = true ;
            nrows = gbmx_get_uint64_scalar (pargin [1], "m") ;
            ncols = gbmx_get_uint64_scalar (pargin [2], "n") ;
            gbmx_mxstring_to_string (string_1, LEN, pargin [3], "") ;

        }
        else if (mxIsChar (pargin [2]) && mxIsChar (pargin [3]))
        { 

            //------------------------------------------------------------------
            // C = GrB (A, type, format)
            // C = GrB (A, format, type)
            //------------------------------------------------------------------

            gbmx_mxstring_to_string (string_1, LEN, pargin [2], "") ;
            gbmx_mxstring_to_string (string_2, LEN, pargin [3], "") ;
        }
        else
        { 
            ERROR ("unknown usage", GrB_INVALID_VALUE) ;
        }

    }
    else // if (nargin == 5)
    { 

        //----------------------------------------------------------------------
        // C = GrB (m, n, type, format)
        // C = GrB (m, n, format, type)
        //----------------------------------------------------------------------

        if (gbmx_mxarray_is_scalar (pargin [1]) &&
            gbmx_mxarray_is_scalar (pargin [2]) &&
            mxIsChar (pargin [3]) && mxIsChar (pargin [4]))
        { 
            nrows = gbmx_get_uint64_scalar (pargin [1], "m") ;
            ncols = gbmx_get_uint64_scalar (pargin [2], "n") ;
            gbmx_mxstring_to_string (string_1, LEN, pargin [3], "") ;
            gbmx_mxstring_to_string (string_2, LEN, pargin [4], "") ;
        }
        else
        { 
            ERROR ("unknown usage", GrB_INVALID_VALUE) ;
        }
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // construct the GraphBLAS matrix
    //--------------------------------------------------------------------------

    int fmt = GxB_BY_COL ;
    int sparsity = 0 ;

    if (nargin == 2)
    { 

        //----------------------------------------------------------------------
        // C = GrB (A)
        //----------------------------------------------------------------------

        // GraphBLAS copy of A, same type and format as A
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
        OK (gb_dup (&C, A, arena, err)) ;

    }
    else if (nargin == 3)
    { 

        //----------------------------------------------------------------------
        // C = GrB (A, type)
        // C = GrB (A, format)
        // C = GrB (m, n)
        //----------------------------------------------------------------------

        if (nargin_2_is_char)
        { 

            //------------------------------------------------------------------
            // C = GrB (A, type)
            // C = GrB (A, format)
            //------------------------------------------------------------------

            GrB_Type type = gb_string_to_type (string_1) ;
            bool ok = gb_string_to_format (string_1, &fmt, NULL,
                &sparsity, NULL) ;

            if (type != NULL)
            { 

                //--------------------------------------------------------------
                // C = GrB (A, type)
                //--------------------------------------------------------------

                if (Matrix [0].is_empty)
                { 
                    // A is a 0-by-0 built-in matrix.  create a new 0-by-0
                    // GraphBLAS matrix C of the given type, with the default
                    // format.
                    OK (gb_new (&C, type, 0, 0, -1, 0, arena, err)) ;
                }
                else
                { 
                    // get a shallow copy and then typecast it to type.
                    // use the same format as A
                    OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena,
                        err)) ;
                    OK (GrB_Matrix_get_INT32 (A, &fmt, GxB_FORMAT)) ;
                    OK (gb_typecast (&C, A, type, fmt, 0, arena, err)) ;
                }

            }
            else if (ok)
            { 

                //--------------------------------------------------------------
                // C = GrB (A, format)
                //--------------------------------------------------------------

                // get a shallow copy of A
                OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
                // C = A with the requested format and sparsity, no typecast
                OK (gb_typecast (&C, A, NULL, fmt, sparsity, arena, err)) ;

            }
            else
            { 
                ERROR ("unknown type or format", GrB_INVALID_VALUE) ;
            }

        }
        else if (nargin_2_is_mn)
        { 

            //------------------------------------------------------------------
            // C = GrB (m, n)
            //------------------------------------------------------------------

            // m-by-n GraphBLAS double matrix, no entries, default format
            OK (gb_new (&C, GrB_FP64, nrows, ncols, -1, 0, arena, err)) ;

        }
        else
        { 
            ERROR ("usage: C=GrB(m,n), C=GrB(A,type), or C=GrB(A,format)",
                GrB_INVALID_VALUE) ;
        }

    }
    else if (nargin == 4)
    { 

        //----------------------------------------------------------------------
        // C = GrB (m, n, format)
        // C = GrB (m, n, type)
        // C = GrB (A, type, format)
        // C = GrB (A, format, type)
        //----------------------------------------------------------------------

        if (nargin_3_first_case)
        { 

            //------------------------------------------------------------------
            // C = GrB (m, n, format)
            // C = GrB (m, n, type)
            //------------------------------------------------------------------

            // create an m-by-n matrix with no entries
            GrB_Type type = gb_string_to_type (string_1) ;
            bool ok = gb_string_to_format (string_1, &fmt, NULL,
                &sparsity, NULL) ;

            if (type != NULL)
            { 
                // C = GrB (m, n, type)
                // create an m-by-n matrix of the desired type, no entries,
                // use the default format.
                OK (gb_new (&C, type, nrows, ncols, -1, sparsity, arena, err)) ;
            }
            else if (ok)
            { 
                // C = GrB (m, n, format)
                // create an m-by-n double matrix of the desired format
                OK (gb_new (&C, GrB_FP64, nrows, ncols, fmt, sparsity, arena,
                    err)) ;
            }
            else
            { 
                ERROR ("unknown type or format", GrB_INVALID_VALUE) ;
            }

        }
        else
        { 

            //------------------------------------------------------------------
            // C = GrB (A, type, format)
            // C = GrB (A, format, type)
            //------------------------------------------------------------------

            GrB_Type type = gb_string_to_type (string_1) ;
            bool ok = gb_string_to_format (string_2, &fmt, NULL,
                &sparsity, NULL) ;

            if (ok)
            { 
                // C = GrB (A, type, format)
            }
            else
            { 
                // C = GrB (A, format, type)
                ok = gb_string_to_format (string_1, &fmt, NULL,
                    &sparsity, NULL) ;
                type = gb_string_to_type (string_2) ;
            }

            if (type == NULL || !ok)
            { 
                ERROR ("unknown type and/or format", GrB_INVALID_VALUE) ;
            }

            if (Matrix [0].is_empty)
            { 
                OK (gb_new (&C, type, 0, 0, fmt, sparsity, arena, err)) ;
            }
            else
            { 
                // get a shallow copy, typecast it, and set the format
                OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
                OK (gb_typecast (&C, A, type, fmt, sparsity, arena, err)) ;
            }
        }

    }
    else // if (nargin == 5)
    { 

        //----------------------------------------------------------------------
        // C = GrB (m, n, type, format)
        // C = GrB (m, n, format, type)
        //----------------------------------------------------------------------

        // create an m-by-n matrix with no entries, of the requested
        // type and format

        GrB_Type type = gb_string_to_type (string_1) ;
        bool ok = gb_string_to_format (string_2, &fmt, NULL, &sparsity, NULL) ;

        if (ok)
        { 
            // C = GrB (m, n, type, format)
        }
        else
        { 
            // C = GrB (m, n, format, type)
            ok = gb_string_to_format (string_1, &fmt, NULL, &sparsity, NULL) ;
            type = gb_string_to_type (string_2) ;
        }

        if (type == NULL || !ok)
        { 
            ERROR ("unknown type and/or format", GrB_INVALID_VALUE) ;
        }

        OK (gb_new (&C, type, nrows, ncols, fmt, sparsity, arena, err)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    OK (gb_export (C_opaque, &C, KIND_GRB, ghb, err)) ;
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct (&C) ;
    }

    gb_wrapup ( ) ;
}

