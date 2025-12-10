//------------------------------------------------------------------------------
// gbselect: select entries from a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// gbselect is an interface to GrB_Matrix_select.

// Usage:

// C = gbselect (op, A)
// C = gbselect (op, A, desc)
// C = gbselect (op, A, b, desc)

// C = gbselect (Cin, accum, op, A, desc)
// C = gbselect (Cin, accum, op, A, b, desc)

// C = gbselect (Cin, M, op, A, desc)
// C = gbselect (Cin, M, op, A, b, desc)

// C = gbselect (Cin, M, accum, op, A, desc)
// C = gbselect (Cin, M, accum, op, A, b, desc)

// If Cin is not present then it is implicitly a matrix with no entries, of the
// right size (which depends on A, and the descriptor).  The type of Cin, if
// not present, is determined by the ztype of the accum, if present, or
// otherwise it has the same time as A.

// If op is '==' or '~=' and b is a NaN, and A has type GrB_FP32, GrB_FP64,
// GxB_FC32, or GxB_FC64, then a user-defined operator is used instead of
// G*B_VALUEEQ* or G*B_VALUENE*.

// The 'tril', 'triu', 'diag', 'offdiag', and 2-input operators all require
// the b scalar.  The b scalar must not appear for the '*0' operators.

#include "gb_interface.h"

#define USAGE "usage: C = GrB.select (Cin, M, accum, op, A, b, desc)"

//------------------------------------------------------------------------------
// nan functions for GrB_IndexUnaryOp operators
//------------------------------------------------------------------------------

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
// gbselect mexFunction
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
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin >= 2 && nargin <= 7 && nargout <= 2, USAGE) ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    mxArray *Matrix [6], *String [2], *Cell [2] ;
    base_enum_t base ;
    kind_enum_t kind ;
    int fmt ;
    int nmatrices, nstrings, ncells, sparsity ;
    GrB_Descriptor desc ;
    gb_get_mxargs (nargin, pargin, USAGE, Matrix, &nmatrices, String, &nstrings,
        Cell, &ncells, &desc, &base, &kind, &fmt, &sparsity) ;

    CHECK_ERROR (nmatrices < 1 || nmatrices > 4 || nstrings < 1 || ncells > 0,
        USAGE) ;

    //--------------------------------------------------------------------------
    // get the select operator; determine the type and ithunk later
    //--------------------------------------------------------------------------

    int64_t ithunk = 0 ;
    GrB_IndexUnaryOp idxunop = NULL ;
    bool thunk_zero = false ; 
    bool op_is_positional = false ;

    gb_mxstring_to_idxunop (&idxunop, &thunk_zero,
        &op_is_positional, &ithunk, String [nstrings-1], GrB_FP64) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    GrB_Type atype, ctype = NULL ;
    GrB_Matrix C = NULL, M = NULL, A, b = NULL ;

    if (thunk_zero)
    {
        if (nmatrices == 1)
        { 
            A = gb_get_shallow (Matrix [0]) ;
        }
        else if (nmatrices == 2)
        { 
            C = gb_get_deep    (Matrix [0]) ;
            A = gb_get_shallow (Matrix [1]) ;
        }
        else if (nmatrices == 3)
        { 
            C = gb_get_deep    (Matrix [0]) ;
            M = gb_get_shallow (Matrix [1]) ;
            A = gb_get_shallow (Matrix [2]) ;
        }
        else // if (nmatrices == 4)
        { 
            ERROR (USAGE) ;
        }
    }
    else
    {
        if (nmatrices == 1)
        { 
            ERROR ("operator input is missing") ;
        }
        else if (nmatrices == 2)
        { 
            A = gb_get_shallow (Matrix [0]) ;
            b = gb_get_shallow (Matrix [1]) ;
        }
        else if (nmatrices == 3)
        { 
            C = gb_get_deep    (Matrix [0]) ;
            A = gb_get_shallow (Matrix [1]) ;
            b = gb_get_shallow (Matrix [2]) ;
        }
        else // if (nmatrices == 4)
        { 
            C = gb_get_deep    (Matrix [0]) ;
            M = gb_get_shallow (Matrix [1]) ;
            A = gb_get_shallow (Matrix [2]) ;
            b = gb_get_shallow (Matrix [3]) ;
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

    gb_mxstring_to_idxunop (&idxunop, &thunk_zero,
        &op_is_positional, &ithunk, String [nstrings-1], atype) ;

    //--------------------------------------------------------------------------
    // get the accum operator
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;
    if (nstrings > 1)
    { 
        // if accum appears, then Cin must also appear
        CHECK_ERROR (C == NULL, USAGE) ;
        accum = gb_mxstring_to_binop (String [0], ctype, ctype) ;
    }

    //--------------------------------------------------------------------------
    // construct the zero thunk scalar, if needed
    //--------------------------------------------------------------------------

    GrB_Scalar Zero = NULL ;
    if (thunk_zero)
    {
        OK (GrB_Scalar_new (&Zero, atype)) ;
        OK (GrB_Scalar_setElement_INT32 (Zero, 0)) ;
        b = (GrB_Matrix) Zero ;
        Zero = NULL ;
    }

    //--------------------------------------------------------------------------
    // construct C if not present on input
    //--------------------------------------------------------------------------

    // If C is NULL, then it is not present on input.
    // Construct C of the right size and type.

    if (C == NULL)
    { 
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
        fmt = gb_get_format (cnrows, cncols, A, NULL, fmt) ;
        sparsity = gb_get_sparsity (A, NULL, sparsity) ;
        C = gb_new (ctype, cnrows, cncols, fmt, sparsity) ;
    }

    //--------------------------------------------------------------------------
    // handle the NaN case
    //--------------------------------------------------------------------------

    GrB_IndexUnaryOp nan_test = NULL ;
    GrB_Matrix b2 = b ;
    GrB_Matrix b3 = NULL, b4 = NULL ;

    if (op_is_positional)
    { 
        // construct a new int64 thunk scalar for positional ops
        OK (GrB_Matrix_new (&b3, GrB_INT64, 1, 1)) ;
        OK (GrB_Matrix_setElement_INT64 (b3, ithunk, 0, 0)) ;
        b2 = b3 ;
    }
    else if (b != NULL)
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
            // instead of the built-in GxB_EQ_THUNK, GxB_NE_THUNK, GrB_VALUEEQ*
            // or GrB_VALUENE* operators.

            if (idxunop == GrB_VALUEEQ_FP32)
            { 
                OK (GxB_IndexUnaryOp_new (&nan_test,
                    (GxB_index_unary_function) gb_isnan32,
                    GrB_BOOL, GrB_FP32, GrB_FP32,
                    "gb_isnan32", ISNAN32_DEFN)) ;
            }
            else if (idxunop == GrB_VALUEEQ_FP64)
            { 
                OK (GxB_IndexUnaryOp_new (&nan_test,
                    (GxB_index_unary_function) gb_isnan64,
                    GrB_BOOL, GrB_FP64, GrB_FP64,
                    "gb_isnan64", ISNAN64_DEFN)) ;
            }
            else if (idxunop == GxB_VALUEEQ_FC32)
            { 
                OK (GxB_IndexUnaryOp_new (&nan_test,
                    (GxB_index_unary_function) gb_isnanfc32,
                    GrB_BOOL, GxB_FC32, GxB_FC32,
                    "gb_isnanfc32", ISNANFC32_DEFN)) ;
            }
            else if (idxunop == GxB_VALUEEQ_FC64)
            { 
                OK (GxB_IndexUnaryOp_new (&nan_test,
                    (GxB_index_unary_function) gb_isnanfc64,
                    GrB_BOOL, GxB_FC64, GxB_FC64,
                    "gb_isnanfc64", ISNANFC64_DEFN)) ;
            }
            else if (idxunop == GrB_VALUENE_FP32)
            { 
                OK (GxB_IndexUnaryOp_new (&nan_test,
                    (GxB_index_unary_function) gb_isnotnan32,
                    GrB_BOOL, GrB_FP32, GrB_FP32,
                    "gb_isnotnan32", ISNOTNAN32_DEFN)) ;
            }
            else if (idxunop == GrB_VALUENE_FP64)
            { 
                OK (GxB_IndexUnaryOp_new (&nan_test,
                    (GxB_index_unary_function) gb_isnotnan64,
                    GrB_BOOL, GrB_FP64, GrB_FP64,
                    "gb_isnotnan64", ISNOTNAN64_DEFN)) ;
            }
            else if (idxunop == GxB_VALUENE_FC32)
            { 
                OK (GxB_IndexUnaryOp_new (&nan_test,
                    (GxB_index_unary_function) gb_isnotnanfc32,
                    GrB_BOOL, GxB_FC32, GxB_FC32,
                    "gb_isnotnanfc32", ISNOTNANFC32_DEFN)) ;
            }
            else if (idxunop == GxB_VALUENE_FC64)
            { 
                OK (GxB_IndexUnaryOp_new (&nan_test,
                    (GxB_index_unary_function) gb_isnotnanfc64,
                    GrB_BOOL, GxB_FC64, GxB_FC64,
                    "gb_isnotnanfc64", ISNOTNANFC64_DEFN)) ;
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
    OK (GrB_Matrix_new (&b4, ytype, 1, 1)) ;
    OK (GrB_Matrix_assign (b4, NULL, NULL, b2, GrB_ALL, 1, GrB_ALL, 1, NULL)) ;
    OK1 (C, GrB_Matrix_select_Scalar (C, M, accum, idxunop, A,
        (GrB_Scalar) b4, desc)) ;

    //--------------------------------------------------------------------------
    // free shallow copies
    //--------------------------------------------------------------------------

    OK (GrB_Scalar_free (&Zero)) ;
    OK (GrB_Matrix_free (&M)) ;
    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Matrix_free (&b)) ;
    OK (GrB_Matrix_free (&b3)) ;
    OK (GrB_Matrix_free (&b4)) ;
    OK (GrB_Descriptor_free (&desc)) ;
    OK (GrB_IndexUnaryOp_free (&nan_test)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    gb_wrapup ( ) ;
}

