//------------------------------------------------------------------------------
// gbmex_vreduce: reduce a matrix to a vector
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_vreduce is an interface to GrB_Matrix_reduce, for GrB.vreduce and
// GhB.vreduce.

// Usage for GrB and GhB (omitting optional final desc argument):

// C = GrB.vreduce (op, A)                  C = op (A)
// C = GrB.vreduce (Cin, op, A)             C = Cin ; C = op (A)
// C = GrB.vreduce (Cin, accum, op, A)      C = Cin ; C += op (A)
// C = GrB.vreduce (Cin, M, op, A)          C = Cin ; C<M> = op (A)
// C = GrB.vreduce (Cin, M, accum, op, A)   C = Cin ; C<M> += op(A)

// Usage for GhB only (inplace usage):

// GhB.vreduce (C, op, A)                   C = op (A)
// GhB.vreduce (C, accum, op, A)            C += op (A)
// GhB.vreduce (C, M, op, A)                C<M> = op (A)
// GhB.vreduce (C, M, accum, op, A)         C<M> += op (A)

// where op(A) refers to reducing A to a vector using the given op.

#include "gb_interface.h"
#include "gb_string_and_type_to_binop_or_idxunop.c"
#include "gb_string_to_binop.c"
#include "gb_string_to_binop_or_idxunop.c"
#include "gb_get_descriptor_mxm.c"
#include "gb_binop_to_monoid.c"
#include "gb_string_to_monoid.c"

#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&M_to_free) ;  \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Descriptor_free (&desc) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

#define USAGE "usage: C = GrB.vreduce (Cin, M, accum, op, A, desc)"

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

    GrB_Type atype, ctype = NULL ;
    GrB_Matrix *C_opaque = NULL, C = NULL, M = NULL, A = NULL,
        M_to_free = NULL, A_to_free = NULL ;
    GrB_Descriptor desc = NULL ;

    GBMX_USAGE (nargin >= 3 && nargin <= 7 && nargout <= 2, USAGE) ;
    bool ghb = (bool) mxGetScalar (pargin [0]) ;
    int arena = ghb ? GrB_DEFAULT : MXARENA ;

    bool inplace = ghb && (nargout == 0) ;
    double *kind_output = NULL ;
    if (!inplace)
    { 
        if (ghb) pargout [0] = gbmx_export_ghb_mxstruct (&C_opaque) ;
        pargout [1] = mxCreateDoubleScalar (0) ;
        kind_output = (double *) mxGetData (pargout [1]) ;
    }
    else
    { 
        /* for tracking test coverage */ ;
    }

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [6] ;
    mxArray *Cell [2] ;
    char String [2][LEN+2] ;
    int nmatrices, nstrings, ncells ;
    struct gb_descriptor_struct gbdesc ;
    gbmx_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String,
        &nstrings, Cell, &ncells, &gbdesc) ;

    CHECK_ERROR (nmatrices < 1 || nmatrices > 3 || nstrings < 1 || ncells > 0,
        USAGE) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    OK (gb_get_descriptor_mxm (&desc, &gbdesc, arena, err)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    if (nmatrices == 1)
    { 
        CHECK_ERROR (inplace, "invalid in-place usage") ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
    }
    else if (nmatrices == 2)
    { 
        OK (gb_get_deep   (&C, inplace,    &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [1]), arena, err)) ;
    }
    else // if (nmatrices == 3)
    { 
        OK (gb_get_deep   (&C, inplace,    &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&M, &M_to_free, &(Matrix [1]), arena, err)) ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [2]), arena, err)) ;
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    if (C != NULL)
    { 
        bool is_column_vector ;
        OK (gb_is_column_vector (&is_column_vector, C, err)) ;
        CHECK_ERROR (!is_column_vector, "Cin must be a column vector") ;
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get the operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;
    GrB_Monoid monoid ;

    if (nstrings == 1)
    { 
        OK (gb_string_to_monoid (&monoid, String [0], atype, err)) ;
    }
    else 
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        OK (gb_string_to_binop (&accum, String [0], ctype, ctype, err)) ;
        OK (gb_string_to_monoid (&monoid, String [1], atype, err)) ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is not present on input.
    // Construct C of the right size and type.

    if (C == NULL)
    { 
        ASSERT (!inplace) ;

        // get the descriptor contents to determine if A is transposed
        int in0 ;
        OK (GrB_Descriptor_get_INT32 (desc, &in0, GrB_INP0)) ;
        bool A_transpose = (in0 == GrB_TRAN) ;

        // get the size of A
        uint64_t anrows, ancols ;
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;

        // determine the size of the vector C
        uint64_t cnrows = (A_transpose) ? ancols : anrows ;

        // use the ztype of the monoid as the type of C
        OK (gb_monoid_type (&ctype, monoid, err)) ;

        // create the matrix C and set its format and sparsity
        OK (gb_get_format (cnrows, 1, A, NULL, &(gbdesc.fmt), err)) ;
        OK (gb_get_sparsity (A, NULL, &(gbdesc.sparsity), err)) ;
        OK (gb_new (&C, ctype, cnrows, 1, gbdesc.fmt, gbdesc.sparsity, arena,
            err)) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += reduce(A)
    //--------------------------------------------------------------------------

    OK1 (C, GrB_Matrix_reduce_Monoid ((GrB_Vector) C, (GrB_Vector) M,
        accum, monoid, A, desc)) ;

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    FREE_WORK ;

    if (!inplace)
    { 
        OK (gb_export (C_opaque, &C, gbdesc.kind, ghb, err)) ;
        (*kind_output) = (double) gbdesc.kind ;
    }
    ////////////////////////////////////////////////////////////////////////////
    if (!ghb)
    { 
        pargout [0] = gbmx_export_grb_mxstruct (&C) ;
    }

    gb_wrapup ( ) ;
}

