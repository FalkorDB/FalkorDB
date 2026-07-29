//------------------------------------------------------------------------------
// gbmx_ewise_mexFunction: eWiseAdd and eWiseMult
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This method implements the gbmex_eadd and gbmex_emult mexFunctions
// for GrB.eadd, GrB.emult, GhB.eadd, and GhB.emult.  Below "ewise" is
// either eadd or emult.

// Usage for GrB and GhB (omitting desc argument):

// C = GrB.ewise (op, A, B)                 C = op(A,B)
// C = GrB.ewise (Cin, op, A, B)            C = op(A,B)
// C = GrB.ewise (Cin, accum, op, A, B)     C = Cin + op(A,B)
// C = GrB.ewise (Cin, M, op, A, B)         C = Cin ; C<M> = op(A,B)
// C = GrB.ewise (Cin, M, accum, op, A, B)  C = Cin ; C<M> += op(A,B)

// Usage for GhB only:

// GhB.ewise (C, op, A, B)                  C = op(A,B)
// GhB.ewise (C, accum, op, A, B)           C += op(A,B)
// GhB.ewise (C, M, op, A, B)               C<M> = op(A,B)
// GhB.ewise (C, M, accum, op, A, B)        C<M> += op(A,B)

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&M_to_free) ;  \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Matrix_free (&B_to_free) ;  \
    GrB_Descriptor_free (&desc) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

void gbmx_ewise_mexFunction
(
    int nargout,                // # output arguments for mexFunction
    mxArray *pargout [ ],       // output arguments for mexFunction
    int nargin,                 // # input arguments for mexFunction
    const mxArray *pargin [ ],  // input arguments for mexFunction
    const bool do_eadd,         // true: eadd, false: emult
    const char *usage           // usage string to print if error
)
{

    //--------------------------------------------------------------------------
    // check inputs and construct outputs
    //--------------------------------------------------------------------------

    GrB_Type atype, btype, ctype = NULL ;
    GrB_Matrix *C_opaque = NULL, C = NULL, M = NULL, A = NULL, B = NULL,
        M_to_free = NULL, A_to_free = NULL, B_to_free = NULL ;
    GrB_Descriptor desc = NULL ;

    GBMX_USAGE (nargin >= 4 && nargin <= 8 && nargout <= 2, usage) ;
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
        if (do_eadd)
        { 
            /* for tracking test coverage */ ;
        }
        else
        { 
            /* for tracking test coverage */ ;
        }
    }

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    struct gb_matrix_struct Matrix [6] ;
    mxArray *Cell [2] ;
    char String [2][LEN+2] ;
    int nmatrices, nstrings, ncells ;
    struct gb_descriptor_struct gbdesc ;
    gbmx_get_mxargs (nargin, pargin, usage, Matrix, &nmatrices, String,
        &nstrings, Cell, &ncells, &gbdesc) ;

    CHECK_ERROR (nmatrices < 2 || nmatrices > 4 || nstrings < 1 || ncells > 0,
        usage) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    OK (gb_get_descriptor (&desc, &gbdesc, arena, err)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    if (nmatrices == 2)
    { 
        CHECK_ERROR (inplace, "invalid in-place usage") ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&B, &B_to_free, &(Matrix [1]), arena, err)) ;
    }
    else if (nmatrices == 3)
    { 
        OK (gb_get_deep   (&C, inplace,    &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [1]), arena, err)) ;
        OK (gb_get_matrix (&B, &B_to_free, &(Matrix [2]), arena, err)) ;
    }
    else // if (nmatrices == 4)
    { 
        OK (gb_get_deep   (&C, inplace,    &(Matrix [0]), arena, err)) ;
        OK (gb_get_matrix (&M, &M_to_free, &(Matrix [1]), arena, err)) ;
        OK (gb_get_matrix (&A, &A_to_free, &(Matrix [2]), arena, err)) ;
        OK (gb_get_matrix (&B, &B_to_free, &(Matrix [3]), arena, err)) ;
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    OK (GxB_Matrix_type (&btype, B)) ;
    if (C != NULL)
    { 
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // get the operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL, op = NULL ;

    if (nstrings == 1)
    { 
        OK (gb_string_to_binop (&op, String [0], atype, btype, err)) ;
    }
    else 
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, usage) ;
        OK (gb_string_to_binop (&accum, String [0], ctype, ctype, err)) ;
        OK (gb_string_to_binop (&op   , String [1], atype, btype, err)) ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is not present on input.
    // Construct C of the right size and type.

    if (C == NULL)
    { 
        ASSERT (!inplace) 

        // get the descriptor contents to determine if A is transposed
        int in0 ;
        OK (GrB_Descriptor_get_INT32 (desc, &in0, GrB_INP0)) ;
        bool A_transpose = (in0 == GrB_TRAN) ;

        // get the size of A
        uint64_t anrows, ancols ;
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;

        // determine the size of C
        uint64_t cnrows = (A_transpose) ? ancols : anrows ;
        uint64_t cncols = (A_transpose) ? anrows : ancols ;

        // use the ztype of the op as the type of C
        OK (gb_binaryop_ztype (&ctype, op, err)) ;

        // create the matrix C and set its format and sparsity
        OK (gb_get_format (cnrows, cncols, A, B, &(gbdesc.fmt), err)) ;
        OK (gb_get_sparsity (A, B, &(gbdesc.sparsity), err)) ;
        OK (gb_new (&C, ctype, cnrows, cncols, gbdesc.fmt, gbdesc.sparsity,
            arena, err)) ;
    }

    //--------------------------------------------------------------------------
    // compute C<M> += (A+B) or (A.*B)
    //--------------------------------------------------------------------------

    if (do_eadd)
    { 
        OK1 (C, GrB_Matrix_eWiseAdd_BinaryOp (C, M, accum, op, A, B, desc)) ;
    }
    else
    { 
        OK1 (C, GrB_Matrix_eWiseMult_BinaryOp (C, M, accum, op, A, B, desc)) ;
    }

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

#undef  FREE_WORK
#define FREE_WORK
#undef  FREE_ALL
#define FREE_ALL

