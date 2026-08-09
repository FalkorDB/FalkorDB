//------------------------------------------------------------------------------
// gbmex_build: build a GraphBLAS matrix or a built-in sparse matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Usage:

// C = GrB.build (I, J, X)
// C = GrB.build (I, J, X, desc)
// C = GrB.build (I, J, X, m, desc)
// C = GrB.build (I, J, X, m, n, desc)
// C = GrB.build (I, J, X, m, n, dup, desc) ;
// C = GrB.build (I, J, X, m, n, dup, type, desc) ;

// X and either I or J may be a scalars, in which case they are effectively
// expanded so that they all have the same length.  X is only implicitly
// expanded if C is built as an iso matrix.

// m and n default to the largest index in I and J, respectively.

// dup is a string that defaults to 'plus.xtype' where xtype is the type of X.
// If dup is given by without a type,  type of dup defaults to the type of X.

// If dup is the empty string '' then any duplicates result in an error.
// If dup is the string 'ignore' then duplicates are ignored.

// type is a string that defines is type of C, which defaults to the type
// of X.

// If X is a scalar, and dup is '1st', '2nd', 'any', 'min', 'max', 'pair' (same
// as 'oneb'), 'or', 'and', 'bitor', or 'bitand', then GxB_Matrix_build_Scalar
// is used and C is built as an iso matrix.  X is not explicitly expanded. This
// is much faster than when using the default dup.

// The descriptor is optional; if present, it must be the last input parameter.

#include "gb_interface.h"
#include "gb_string_and_type_to_binop_or_idxunop.c"
#include "gb_string_to_binop.c"
#include "gb_string_to_binop_or_idxunop.c"

#include "gb_matrix_to_list.c"

#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Vector_free (&I_to_free) ;  \
    GrB_Vector_free (&J_to_free) ;  \
    GrB_Vector_free (&X_to_free) ;  \
    GrB_Vector_free (&I2) ;         \
    GrB_Vector_free (&J2) ;         \
    GrB_Vector_free (&X2) ;         \
    GrB_Scalar_free (&x) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#define USAGE "usage: C = GrB.build (I, J, X, m, n, dup, type, desc)"

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

    GrB_Vector
        I  = NULL, J  = NULL, X  = NULL,    // never freed; alias of [IJX][12]
        I1 = NULL, J1 = NULL, X1 = NULL,    // never freed
        I2 = NULL, J2 = NULL, X2 = NULL,    // from gb_expand_scalar_to_vector
        I_to_free = NULL, J_to_free = NULL, X_to_free = NULL ;
    GrB_Matrix *C_opaque = NULL, C = NULL ;
    uint64_t nrows = 0, ncols = 0 ;
    GrB_BinaryOp dup = GxB_IGNORE_DUP ;
    GrB_Type type = NULL ;
    GrB_Scalar x = NULL ;

    GBMX_USAGE (nargin >= 4 && nargin <= 9 && nargout <= 2, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    double *kind_output = NULL ;
    if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;
    pargout [1] = mxCreateDoubleScalar (0) ;
    kind_output = (double *) mxGetData (pargout [1]) ;

    //--------------------------------------------------------------------------
    // get I,J,X inputs
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [3] ;
    gbmx_get_matrix (&(Matrix [0]), pargin [1]) ;
    gbmx_get_matrix (&(Matrix [1]), pargin [2]) ;
    gbmx_get_matrix (&(Matrix [2]), pargin [3]) ;

    //--------------------------------------------------------------------------
    // get the descriptor
    //--------------------------------------------------------------------------

    struct gb_descriptor_struct gbdesc ;
    if (gbmx_mxarray_to_descriptor (&gbdesc, pargin [nargin-1]))
    { 
        // descriptor is present, remove it from further consideration
        nargin-- ;
    }

    if (nargin >= 5)
    { 
        // m is provided on input
        nrows = gbmx_get_uint64_scalar (pargin [4], "m") ;
    }

    if (nargin >= 6)
    { 
        // n is provided on input
        ncols = gbmx_get_uint64_scalar (pargin [5], "n") ;
    }

    bool default_dup = (nargin < 7) ;
    char op_string [LEN+2] ;
    op_string [0] = '\0' ;
    if (!default_dup)
    { 
        gbmx_mxstring_to_string (op_string, LEN, pargin [6], "dup") ;
    }

    char type_string [LEN+2] ;
    type_string [0] = '\0' ;
    if (nargin > 7)
    { 
        gbmx_mxstring_to_string (type_string, LEN, pargin [7], "type") ;
    }

    ////////////////////////////////////////////////////////////////////////////

    int base_offset = (gbdesc.base == BASE_0_INT) ? 0 : 1 ;

    //--------------------------------------------------------------------------
    // get I, J, and X and their properties
    //--------------------------------------------------------------------------

    OK (gb_matrix_to_list (&I1, &I_to_free, &(Matrix [0]), base_offset, arena,
        err)) ;
    OK (gb_matrix_to_list (&J1, &J_to_free, &(Matrix [1]), base_offset, arena,
        err)) ;
    OK (gb_matrix_to_list (&X1, &X_to_free, &(Matrix [2]), 0, arena, err)) ;

    // use the input I, J, X unless they are revised, below
    I = I1 ;
    J = J1 ;
    X = X1 ;

    uint64_t ni, nj, nx ;
    OK (GrB_Vector_nvals (&ni, I)) ;
    OK (GrB_Vector_nvals (&nj, J)) ;
    OK (GrB_Vector_nvals (&nx, X)) ;

    GrB_Type xtype ;
    OK (GxB_Vector_type (&xtype, X)) ;

    uint64_t Imax = UINT64_MAX, Jmax = UINT64_MAX ;

    //--------------------------------------------------------------------------
    // check the sizes of I, J, and X, and the type of X
    //--------------------------------------------------------------------------

    uint64_t nvals = MAX (ni, nj) ;
    nvals = MAX (nvals, nx) ;

    if (!(ni == 1 || ni == nvals) ||
        !(nj == 1 || nj == nvals) ||
        !(nx == 1 || nx == nvals))
    { 
        ERROR ("I, J, and X must have the same # of entries",
            GrB_DIMENSION_MISMATCH) ;
    }

    //--------------------------------------------------------------------------
    // expand any scalars in I and J (but not yet X)
    //--------------------------------------------------------------------------

    GrB_Monoid max = GrB_MAX_MONOID_UINT64 ;

    if (ni == 1 && ni < nvals)
    { 
        if (Imax == UINT64_MAX)
        { 
            OK (GrB_Vector_reduce_UINT64 (&Imax, NULL, max, I, NULL)) ;
        }
        GrB_Type itype = (Imax < UINT32_MAX) ? GrB_UINT32 : GrB_UINT64 ;
        OK (gb_expand_scalar_to_vector (&I2, I, itype, nvals, arena, err)) ;
        I = I2 ;
    }

    if (nj == 1 && nj < nvals)
    { 
        if (Jmax == UINT64_MAX)
        { 
            OK (GrB_Vector_reduce_UINT64 (&Jmax, NULL, max, J, NULL)) ;
        }
        GrB_Type jtype = (Jmax < UINT32_MAX) ? GrB_UINT32 : GrB_UINT64 ;
        OK (gb_expand_scalar_to_vector (&J2, J, jtype, nvals, arena, err)) ;
        J = J2 ;
    }

    //--------------------------------------------------------------------------
    // get m and n if present
    //--------------------------------------------------------------------------

    if (nargin < 5)
    { 
        // nrows = max entry in I + 1
        if (Imax == UINT64_MAX)
        { 
            OK (GrB_Vector_reduce_UINT64 (&Imax, NULL, max, I, NULL)) ;
        }
        nrows = Imax + 1 ;
    }

    if (nargin < 6)
    { 
        // ncols = max entry in J + 1
        if (Jmax == UINT64_MAX)
        { 
            OK (GrB_Vector_reduce_UINT64 (&Jmax, NULL, max, J, NULL)) ;
        }
        ncols = Jmax + 1 ;
    }

    //--------------------------------------------------------------------------
    // get the dup operator
    //--------------------------------------------------------------------------

    if (!default_dup)
    { 
        OK (gb_string_to_binop (&dup, op_string, xtype, xtype, err)) ;
    }

    bool nice_iso_dup = false ;
    if (default_dup)
    { 
        // dup defaults to plus.xtype or GrB_LOR for boolean
        if (xtype == GrB_BOOL)
        { 
            // dup is GrB_LOR which is nice for an iso build.  For all other
            // types, the dup is plus, which is not nice.
            dup = GrB_LOR ;
            nice_iso_dup = true ;
        }
        else if (xtype == GrB_INT8)
        { 
            dup = GrB_PLUS_INT8 ;
        }
        else if (xtype == GrB_INT16)
        { 
            dup = GrB_PLUS_INT16 ;
        }
        else if (xtype == GrB_INT32)
        { 
            dup = GrB_PLUS_INT32 ;
        }
        else if (xtype == GrB_INT64)
        { 
            dup = GrB_PLUS_INT64 ;
        }
        else if (xtype == GrB_UINT8)
        { 
            dup = GrB_PLUS_UINT8 ;
        }
        else if (xtype == GrB_UINT16)
        { 
            dup = GrB_PLUS_UINT16 ;
        }
        else if (xtype == GrB_UINT32)
        { 
            dup = GrB_PLUS_UINT32 ;
        }
        else if (xtype == GrB_UINT64)
        { 
            dup = GrB_PLUS_UINT64 ;
        }
        else if (xtype == GrB_FP32)
        { 
            dup = GrB_PLUS_FP32 ;
        }
        else if (xtype == GrB_FP64)
        { 
            dup = GrB_PLUS_FP64 ;
        }
        else if (xtype == GxB_FC32)
        { 
            dup = GxB_PLUS_FC32 ;
        }
        else if (xtype == GxB_FC64)
        { 
            dup = GxB_PLUS_FC64 ;
        }
        else
        {
            ERROR ("unsupported type", GrB_DOMAIN_MISMATCH) ;
        }
    }
    else if (dup == NULL || dup == GxB_IGNORE_DUP)
    { 
        // if X is a scalar and dup is '' (NULL) or 'ignore' (GxB_IGNORE_DUP),
        // then dup is a nice iso dup.
        nice_iso_dup = true ;
    }
    else
    { 
        // parse dup to see if it will build an iso matrix if X is a scalar
        int32_t position [2] ;
        gb_find_dot (position, op_string) ;
        if (position [0] >= 0) op_string [position [0]] = '\0' ;
        nice_iso_dup =
            MATCH (op_string, "1st") || MATCH (op_string, "first" ) ||
            MATCH (op_string, "2nd") || MATCH (op_string, "second") ||
            MATCH (op_string, "any") ||
            MATCH (op_string, "min") || MATCH (op_string, "max"   ) ||
            MATCH (op_string, "||" ) || MATCH (op_string, "|"     ) ||
            MATCH (op_string, "&&" ) || MATCH (op_string, "&"     ) ||
            MATCH (op_string, "or" ) || MATCH (op_string, "bitor" ) ||
            MATCH (op_string, "and") || MATCH (op_string, "bitand") ||
            MATCH (op_string, "lor") || MATCH (op_string, "land"  ) ;
    }

    //--------------------------------------------------------------------------
    // get the output matrix type
    //--------------------------------------------------------------------------

    if (nargin > 7)
    { 
        type = gb_string_to_type (type_string) ;
        CHECK_ERROR (type == NULL, "unknown type") ;
    }
    else
    { 
        type = xtype ;
    }

    //--------------------------------------------------------------------------
    // build the matrix
    //--------------------------------------------------------------------------

    OK (gb_get_format (nrows, ncols, NULL, NULL, &(gbdesc.fmt), err)) ;
    OK (gb_get_sparsity (NULL, NULL, &(gbdesc.sparsity), err)) ;
    OK (gb_new (&C, type, nrows, ncols, gbdesc.fmt, gbdesc.sparsity, arena,
        err)) ;

    if (nvals > 0)
    {
        bool X_is_scalar = (nx == 1 && nx < nvals) ;
        bool iso_build = X_is_scalar && nice_iso_dup ;
        if (iso_build)
        { 
            // build an iso matrix, with no dup operator
            OK (gb_get_first_scalar (&x, X, xtype, arena, err)) ;
            OK1 (C, GxB_Matrix_build_Scalar_Vector (C, I, J, x, NULL)) ;
        }
        else
        { 
            // build a standard matrix from the three vectors I,J,X
            if (X_is_scalar)
            { 
                // expand X from a scalar to a vector of length nvals
                OK (gb_expand_scalar_to_vector (&X2, X, xtype, nvals, arena,
                    err)) ;
                X = X2 ;
            }
            OK1 (C, GxB_Matrix_build_Vector (C, I, J, X, dup, NULL)) ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    OK (gb_export (C_opaque, &C, gbdesc.kind, ghb, err)) ;
    (*kind_output) = (double) gbdesc.kind ;
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct (&C) ;
    }

    gb_wrapup ( ) ;
}

