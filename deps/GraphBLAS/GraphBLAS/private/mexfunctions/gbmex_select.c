//------------------------------------------------------------------------------
// gbmex_select: select entries from a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbmex_select is an interface to GrB_Matrix_select, for GrB.select
// and GhB.select.

// Usage for GrB and GhB (omitting desc argument):

// C = GrB.select (op, A)                       C = op(A)
// C = GrB.select (Cin, op, A)                  C = Cin ; C = op(A)
// C = GrB.select (Cin, accum, op, A)           C = Cin ; C += op(A)
// C = GrB.select (Cin, M, op, A)               C = Cin ; C<M> = op(A)
// C = GrB.select (Cin, M, accum, op, A)        C = Cin ; C<M> += op(A)

// C = GrB.select (op, A, b)                    C = op(A,b)
// C = GrB.select (Cin, op, A, b)               C = Cin ; C = op(A,b)
// C = GrB.select (Cin, accum, op, A, b)        C = Cin ; C += op(A,b)
// C = GrB.select (Cin, M, op, A, b)            C = Cin ; C<M> = op(A,b)
// C = GrB.select (Cin, M, accum, op, A, b)     C = Cin ; C<M> += op(A,b)

// Usage for GhB only:

// GhB.select (C, op, A)                        C = op(A)
// GhB.select (C, accum, op, A)                 C += op(A)
// GhB.select (C, M, op, A)                     C<M> = op(A)
// GhB.select (C, M, accum, op, A)              C<M> += op(A)

// GhB.select (C, op, A, b)                     C = op(A,b)
// GhB.select (C, accum, op, A, b)              C += op(A,b)
// GhB.select (C, M, op, A, b)                  C<M> = op(A,b)
// GhB.select (C, M, accum, op, A, b)           C<M> += op(A,b)

// where op(A) refers to select(A) using the given op, and op(A,b) uses
// an operator that requires a scalar input b.

// If op is '==' or '~=' and b is a NaN, and A has type GrB_FP32, GrB_FP64,
// GxB_FC32, or GxB_FC64, then a user-defined operator is used instead of
// G*B_VALUEEQ* or G*B_VALUENE*.

// The 'tril', 'triu', 'diag', 'offdiag', and 2-input operators all require
// the b scalar.  The b scalar must not appear for the '*0' operators.

#include "gb_interface.h"
#include "gb_string_to_idxunop.c"
#include "gb_string_and_type_to_binop_or_idxunop.c"
#include "gb_string_to_binop.c"
#include "gb_string_to_binop_or_idxunop.c"

#include "gbmx_interface.h"

#undef  FREE_WORK
#define FREE_WORK                       \
    GrB_Scalar_free (&Zero) ;           \
    GrB_Matrix_free (&M_to_free) ;      \
    GrB_Matrix_free (&A_to_free) ;      \
    GrB_Matrix_free (&b_to_free) ;      \
    GrB_Matrix_free (&b3) ;             \
    GrB_Matrix_free (&b4) ;             \
    GrB_IndexUnaryOp_free (&nan_test) ; \
    GrB_Descriptor_free (&desc) ;

#undef  FREE_ALL
#define FREE_ALL                        \
    FREE_WORK ;                         \
    GrB_Matrix_free (&C) ;

#define USAGE "usage: C = GrB.select (Cin, M, accum, op, A, b, desc)"

//------------------------------------------------------------------------------
// nan functions for GrB_IndexUnaryOp operators
//------------------------------------------------------------------------------

void gb_isnan32 (bool *z, const float *aij,
                 int64_t i, int64_t j, const void *thunk) ;
void gb_isnan64 (bool *z, const double *aij,
                 int64_t i, int64_t j, const void *thunk) ;
void gb_isnotnan32 (bool *z, const float *aij,
                    int64_t i, int64_t j, const void *thunk) ;
void gb_isnotnan64 (bool *z, const double *aij,
                    int64_t i, int64_t j, const void *thunk) ;
void gb_isnanfc32 (bool *z, const GxB_FC32_t *x,
                   int64_t i, int64_t j, const void *thunk) ;
void gb_isnanfc64 (bool *z, const GxB_FC64_t *aij,
                   int64_t i, int64_t j, const void *thunk) ;
void gb_isnotnanfc32 (bool *z, const GxB_FC32_t *aij,
                      int64_t i, int64_t j, const void *thunk) ;
void gb_isnotnanfc64 (bool *z, const GxB_FC64_t *aij,
                      int64_t i, int64_t j, const void *thunk) ;

void gb_isnan32 (bool *z, const float *aij,
                 int64_t i, int64_t j, const void *thunk)
{ 
    (*z) = isnan (*aij) ;
}

#define ISNAN32_DEFN                                                \
"void gb_isnan32 (bool *z, const float *aij,                    \n" \
"                 int64_t i, int64_t j, const void *thunk)      \n" \
"{                                                              \n" \
"    (*z) = isnan (*aij) ;                                      \n" \
"}"

void gb_isnan64 (bool *z, const double *aij,
                 int64_t i, int64_t j, const void *thunk)
{ 
    (*z) = isnan (*aij) ;
}

#define ISNAN64_DEFN                                                \
"void gb_isnan64 (bool *z, const double *aij,                   \n" \
"                 int64_t i, int64_t j, const void *thunk)      \n" \
"{                                                              \n" \
"    (*z) = isnan (*aij) ;                                      \n" \
"}"

void gb_isnotnan32 (bool *z, const float *aij,
                    int64_t i, int64_t j, const void *thunk)
{ 
    (*z) = !isnan (*aij) ;
}

#define ISNOTNAN32_DEFN                                             \
"void gb_isnotnan32 (bool *z, const float *aij,                 \n" \
"                    int64_t i, int64_t j, const void *thunk)   \n" \
"{                                                              \n" \
"    (*z) = !isnan (*aij) ;                                     \n" \
"}"

void gb_isnotnan64 (bool *z, const double *aij,
                    int64_t i, int64_t j, const void *thunk)
{ 
    (*z) = !isnan (*aij) ;
}

#define ISNOTNAN64_DEFN                                             \
"void gb_isnotnan64 (bool *z, const double *aij,                \n" \
"                    int64_t i, int64_t j, const void *thunk)   \n" \
"{                                                              \n" \
"    (*z) = !isnan (*aij) ;                                     \n" \
"}"

void gb_isnanfc32 (bool *z, const GxB_FC32_t *aij,
                   int64_t i, int64_t j, const void *thunk)
{ 
    (*z) = isnan (crealf (*aij)) || isnan (cimagf (*aij)) ;
}

#define ISNANFC32_DEFN                                              \
"void gb_isnanfc32 (bool *z, const GxB_FC32_t *aij,             \n" \
"                   int64_t i, int64_t j, const void *thunk)    \n" \
"{                                                              \n" \
"    (*z) = isnan (crealf (*aij)) || isnan (cimagf (*aij)) ;    \n" \
"}"

void gb_isnanfc64 (bool *z, const GxB_FC64_t *aij,
                   int64_t i, int64_t j, const void *thunk)
{ 
    (*z) = isnan (creal (*aij)) || isnan (cimag (*aij)) ;
}

#define ISNANFC64_DEFN                                              \
"void gb_isnanfc64 (bool *z, const GxB_FC64_t *aij,             \n" \
"                   int64_t i, int64_t j, const void *thunk)    \n" \
"{                                                              \n" \
"    (*z) = isnan (creal (*aij)) || isnan (cimag (*aij)) ;      \n" \
"}"

void gb_isnotnanfc32 (bool *z, const GxB_FC32_t *aij,
                      int64_t i, int64_t j, const void *thunk)
{ 
    (*z) = !isnan (crealf (*aij)) && !isnan (cimagf (*aij)) ;
}

#define ISNOTNANFC32_DEFN                                           \
"void gb_isnotnanfc32 (bool *z, const GxB_FC32_t *aij,          \n" \
"                      int64_t i, int64_t j, const void *thunk) \n" \
"{                                                              \n" \
"    (*z) = !isnan (crealf (*aij)) && !isnan (cimagf (*aij)) ;  \n" \
"}"

void gb_isnotnanfc64 (bool *z, const GxB_FC64_t *aij,
                      int64_t i, int64_t j, const void *thunk)
{ 
    (*z) = !isnan (creal (*aij)) && !isnan (cimag (*aij)) ;
}

#define ISNOTNANFC64_DEFN                                           \
"void gb_isnotnanfc64 (bool *z, const GxB_FC64_t *aij,          \n" \
"                      int64_t i, int64_t j, const void *thunk) \n" \
"{                                                              \n" \
"    (*z) = !isnan (creal (*aij)) && !isnan (cimag (*aij)) ;    \n" \
"}"

//------------------------------------------------------------------------------
// gbmex_select mexFunction
//------------------------------------------------------------------------------

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

    GrB_IndexUnaryOp idxunop = NULL ;
    GrB_Type atype, ctype = NULL ;
    GrB_Matrix *C_opaque = NULL, C = NULL, M = NULL, A = NULL, b = NULL,
        M_to_free = NULL, A_to_free = NULL, b_to_free = NULL,
        b3 = NULL, b4 = NULL ;
    GrB_Descriptor desc = NULL ;
    GrB_Scalar Zero = NULL ;
    GrB_IndexUnaryOp nan_test = NULL ;

    GBMX_USAGE (nargin >= 3 && nargin <= 8 && nargout <= 2, USAGE) ;
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

    CHECK_ERROR (nmatrices < 1 || nmatrices > 4 || nstrings < 1 || ncells > 0,
        USAGE) ;

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // get the GrB_Descriptor
    //--------------------------------------------------------------------------

    OK (gb_get_descriptor (&desc, &gbdesc, arena, err)) ;

    //--------------------------------------------------------------------------
    // get the select operator; determine the type and ithunk later
    //--------------------------------------------------------------------------

    int64_t ithunk = 0 ;
    bool thunk_zero = false ; 
    bool op_is_positional = false ;

    OK (gb_string_to_idxunop (&idxunop, &thunk_zero, &op_is_positional, &ithunk,
        String [nstrings-1], GrB_FP64, err)) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    if (thunk_zero)
    { 
        if (nmatrices == 1)
        { 
            // C = select (op, A)
            CHECK_ERROR (inplace, "invalid in-place usage") ;
            OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
        }
        else if (nmatrices == 2)
        { 
            OK (gb_get_deep   (&C, inplace,    &(Matrix [0]), arena, err)) ;
            OK (gb_get_matrix (&A, &A_to_free, &(Matrix [1]), arena, err)) ;
        }
        else if (nmatrices == 3)
        { 
            OK (gb_get_deep   (&C, inplace,    &(Matrix [0]), arena, err)) ;
            OK (gb_get_matrix (&M, &M_to_free, &(Matrix [1]), arena, err)) ;
            OK (gb_get_matrix (&A, &A_to_free, &(Matrix [2]), arena, err)) ;
        }
        else // if (nmatrices == 4)
        { 
            ERROR (USAGE, GrB_INVALID_VALUE) ;
        }
    }
    else
    { 
        if (nmatrices == 1)
        { 
            ERROR ("operator input is missing", GrB_INVALID_VALUE) ;
        }
        else if (nmatrices == 2)
        { 
            CHECK_ERROR (inplace, "invalid in-place usage") ;
            // C = select (op, A, b)
            OK (gb_get_matrix (&A, &A_to_free, &(Matrix [0]), arena, err)) ;
            OK (gb_get_matrix (&b, &b_to_free, &(Matrix [1]), arena, err)) ;
        }
        else if (nmatrices == 3)
        { 
            OK (gb_get_deep   (&C, inplace,    &(Matrix [0]), arena, err)) ;
            OK (gb_get_matrix (&A, &A_to_free, &(Matrix [1]), arena, err)) ;
            OK (gb_get_matrix (&b, &b_to_free, &(Matrix [2]), arena, err)) ;
        }
        else // if (nmatrices == 4)
        { 
            OK (gb_get_deep   (&C, inplace,    &(Matrix [0]), arena, err)) ;
            OK (gb_get_matrix (&M, &M_to_free, &(Matrix [1]), arena, err)) ;
            OK (gb_get_matrix (&A, &A_to_free, &(Matrix [2]), arena, err)) ;
            OK (gb_get_matrix (&b, &b_to_free, &(Matrix [3]), arena, err)) ;
        }
    }

    OK (GxB_Matrix_type (&atype, A)) ;
    if (C != NULL)
    { 
        OK (GxB_Matrix_type (&ctype, C)) ;
    }

    //--------------------------------------------------------------------------
    // finalize the select operator and ithunk
    //--------------------------------------------------------------------------

    ithunk = 0 ;
    GrB_Type btype = NULL ;
    if (b != NULL)
    { 
        OK (GxB_Matrix_type (&btype, b)) ;
        if (op_is_positional)
        { 
            // get ithunk from the b scalar
            OK0 (GrB_Matrix_extractElement_INT64 (&ithunk, b, 0, 0)) ;
        }
    }

    OK (gb_string_to_idxunop (&idxunop, &thunk_zero, &op_is_positional, &ithunk,
        String [nstrings-1], atype, err)) ;

    //--------------------------------------------------------------------------
    // get the accum operator
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;
    if (nstrings > 1)
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        OK (gb_string_to_binop (&accum, String [0], ctype, ctype, err)) ;
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

        // determine the size of C
        uint64_t cnrows = (A_transpose) ? ancols : anrows ;
        uint64_t cncols = (A_transpose) ? anrows : ancols ;

        // C has the same type as A
        OK (GxB_Matrix_type (&ctype, A)) ;

        // create the matrix C and set its format and sparsity
        OK (gb_get_format (cnrows, cncols, A, NULL, &(gbdesc.fmt), err)) ;
        OK (gb_get_sparsity (A, NULL, &(gbdesc.sparsity), err)) ;
        OK (gb_new (&C, ctype, cnrows, cncols, gbdesc.fmt, gbdesc.sparsity,
            arena, err)) ;
    }

    //--------------------------------------------------------------------------
    // construct the zero thunk scalar, if needed
    //--------------------------------------------------------------------------

    GrB_Matrix b2 = b ;

    if (thunk_zero)
    { 
        OK (GxB_Scalar_new_arena (&Zero, atype, arena, arena)) ;
        OK (GrB_Scalar_setElement_INT32 (Zero, 0)) ;
        b2 = (GrB_Matrix) Zero ;
    }

    //--------------------------------------------------------------------------
    // handle the NaN case
    //--------------------------------------------------------------------------

    if (op_is_positional)
    { 
        // construct a new int64 thunk scalar for positional ops
        OK (GxB_Matrix_new_arena (&b3, GrB_INT64, 1, 1, arena, arena)) ;
        OK (GrB_Matrix_setElement_INT64 (b3, ithunk, 0, 0)) ;
        b2 = b3 ;
    }
    else if (b != NULL && !thunk_zero)
    { 
        // check if b is NaN
        bool b_is_nan = false ;
        if (btype == GrB_FP32)
        { 
            float b_value = 0 ;
            OK0 (GrB_Matrix_extractElement_FP32 (&b_value, b, 0, 0)) ;
            b_is_nan = isnan (b_value) ;
        }
        else if (btype == GrB_FP64)
        { 
            double b_value = 0 ;
            OK0 (GrB_Matrix_extractElement_FP64 (&b_value, b, 0, 0)) ;
            b_is_nan = isnan (b_value) ;
        }
        else if (btype == GxB_FC32)
        { 
            GxB_FC32_t b_value = GxB_CMPLXF (0, 0) ;
            OK0 (GxB_Matrix_extractElement_FC32 (&b_value, b, 0, 0)) ;
            b_is_nan = isnan (crealf (b_value)) || isnan (cimagf (b_value)) ;
        }
        else if (btype == GxB_FC64)
        { 
            GxB_FC64_t b_value = GxB_CMPLX (0, 0) ;
            OK0 (GxB_Matrix_extractElement_FC64 (&b_value, b, 0, 0)) ;
            b_is_nan = isnan (creal (b_value)) || isnan (cimag (b_value)) ;
        }

        if (b_is_nan)
        { 
            // b is NaN; create a new nan_test operator if it should be used
            // instead of the built-in operators.

            if (idxunop == GrB_VALUEEQ_FP32)
            { 
                OK (GxB_IndexUnaryOp_new_arena (&nan_test,
                    (GxB_index_unary_function) gb_isnan32,
                    GrB_BOOL, GrB_FP32, GrB_FP32,
                    "gb_isnan32", ISNAN32_DEFN, arena)) ;
            }
            else if (idxunop == GrB_VALUEEQ_FP64)
            { 
                OK (GxB_IndexUnaryOp_new_arena (&nan_test,
                    (GxB_index_unary_function) gb_isnan64,
                    GrB_BOOL, GrB_FP64, GrB_FP64,
                    "gb_isnan64", ISNAN64_DEFN, arena)) ;
            }
            else if (idxunop == GxB_VALUEEQ_FC32)
            { 
                OK (GxB_IndexUnaryOp_new_arena (&nan_test,
                    (GxB_index_unary_function) gb_isnanfc32,
                    GrB_BOOL, GxB_FC32, GxB_FC32,
                    "gb_isnanfc32", ISNANFC32_DEFN, arena)) ;
            }
            else if (idxunop == GxB_VALUEEQ_FC64)
            { 
                OK (GxB_IndexUnaryOp_new_arena (&nan_test,
                    (GxB_index_unary_function) gb_isnanfc64,
                    GrB_BOOL, GxB_FC64, GxB_FC64,
                    "gb_isnanfc64", ISNANFC64_DEFN, arena)) ;
            }
            else if (idxunop == GrB_VALUENE_FP32)
            { 
                OK (GxB_IndexUnaryOp_new_arena (&nan_test,
                    (GxB_index_unary_function) gb_isnotnan32,
                    GrB_BOOL, GrB_FP32, GrB_FP32,
                    "gb_isnotnan32", ISNOTNAN32_DEFN, arena)) ;
            }
            else if (idxunop == GrB_VALUENE_FP64)
            { 
                OK (GxB_IndexUnaryOp_new_arena (&nan_test,
                    (GxB_index_unary_function) gb_isnotnan64,
                    GrB_BOOL, GrB_FP64, GrB_FP64,
                    "gb_isnotnan64", ISNOTNAN64_DEFN, arena)) ;
            }
            else if (idxunop == GxB_VALUENE_FC32)
            { 
                OK (GxB_IndexUnaryOp_new_arena (&nan_test,
                    (GxB_index_unary_function) gb_isnotnanfc32,
                    GrB_BOOL, GxB_FC32, GxB_FC32,
                    "gb_isnotnanfc32", ISNOTNANFC32_DEFN, arena)) ;
            }
            else if (idxunop == GxB_VALUENE_FC64)
            { 
                OK (GxB_IndexUnaryOp_new_arena (&nan_test,
                    (GxB_index_unary_function) gb_isnotnanfc64,
                    GrB_BOOL, GxB_FC64, GxB_FC64,
                    "gb_isnotnanfc64", ISNOTNANFC64_DEFN, arena)) ;
            }
        }

        if (nan_test != NULL)
        { 
            // use the new operator instead of the built-in one
            idxunop = nan_test ;
        }
    }

    //--------------------------------------------------------------------------
    // compute C<M> += select (A, b2)
    //--------------------------------------------------------------------------

    // typecast the b2 scalar to the idxunop->ytype
    int code ;
    OK (GrB_IndexUnaryOp_get_INT32 (idxunop, &code, GrB_INP1_TYPE_CODE)) ;
    GrB_Type ytype = gb_code_to_type (code) ;
    OK (GxB_Matrix_new_arena (&b4, ytype, 1, 1, arena, arena)) ;
    OK (GrB_Matrix_assign (b4, NULL, NULL, b2, GrB_ALL, 1, GrB_ALL, 1, NULL)) ;
    OK1 (C, GrB_Matrix_select_Scalar (C, M, accum, idxunop, A,
        (GrB_Scalar) b4, desc)) ;

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

