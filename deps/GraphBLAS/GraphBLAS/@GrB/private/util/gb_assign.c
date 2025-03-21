//------------------------------------------------------------------------------
// gb_assign: assign entries into a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// With do_subassign false, gb_assign is an interface to GrB_Matrix_assign and
// GrB_Matrix_assign_[TYPE], computing the GraphBLAS expression:

//      C<#M,replace>(I,J) = accum (C(I,J), A) or accum(C(I,J), A')

// With do_subassign true, gb_assign is an interface to GxB_Matrix_subassign
// and GxB_Matrix_subassign_[TYPE], computing the GraphBLAS expression:

//      C(I,J)<#M,replace> = accum (C(I,J), A) or accum(C(I,J), A')

// A can be a matrix or a scalar.  If it is a scalar with nnz (A) == 0,
// then it is first expanded to an empty matrix of size length(I)-by-length(J),
// and G*B_Matrix_*assign is used (not GraphBLAS scalar assignment).

// Usage:

//      C = gbassign    (Cin, M, accum, A, I, J, desc)
//      C = gbsubassign (Cin, M, accum, A, I, J, desc)

// Cin and A are required.  See GrB.m for more details.

#include "gb_interface.h"

void gb_assign                  // gbassign or gbsubassign mexFunctions
(
    int nargout,                // # output arguments for mexFunction
    mxArray *pargout [ ],       // output arguments for mexFunction
    int nargin,                 // # input arguments for mexFunction
    const mxArray *pargin [ ],  // input arguments for mexFunction
    bool do_subassign,          // true: do subassign, false: do assign
    const char *usage           // usage string to print if error
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    gb_usage (nargin >= 2 && nargin <= 7 && nargout <= 2, usage) ;

    //--------------------------------------------------------------------------
    // find the arguments
    //--------------------------------------------------------------------------

    mxArray *Matrix [4], *String [2], *Cell [2] ;
    base_enum_t base ;
    kind_enum_t kind ;
    int fmt ;
    int nmatrices, nstrings, ncells, sparsity ;
    GrB_Descriptor desc ;
    gb_get_mxargs (nargin, pargin, usage, Matrix, &nmatrices, String, &nstrings,
        Cell, &ncells, &desc, &base, &kind, &fmt, &sparsity) ;

    CHECK_ERROR (nmatrices < 2 || nmatrices > 3 || nstrings > 1, usage) ;

    //--------------------------------------------------------------------------
    // create the descriptor, if not present
    //--------------------------------------------------------------------------

    if (desc == NULL)
    { 
        OK (GrB_Descriptor_new (&desc)) ;
    }

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    GrB_Type ctype ;
    GrB_Matrix C, M = NULL, A ;

    if (nmatrices == 2)
    { 
        C = gb_get_deep    (Matrix [0]) ;
        A = gb_get_shallow (Matrix [1]) ;
    }
    else // if (nmatrices == 3)
    { 
        C = gb_get_deep    (Matrix [0]) ;
        M = gb_get_shallow (Matrix [1]) ;
        A = gb_get_shallow (Matrix [2]) ;
    }

    OK (GxB_Matrix_type (&ctype, C)) ;

    //--------------------------------------------------------------------------
    // get the operator
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;

    if (nstrings == 1)
    { 
        accum = gb_mxstring_to_binop (String [0], ctype, ctype) ;
    }

    //--------------------------------------------------------------------------
    // get the size of Cin
    //--------------------------------------------------------------------------

    uint64_t cnrows, cncols ;
    OK (GrB_Matrix_nrows (&cnrows, C)) ;
    OK (GrB_Matrix_ncols (&cncols, C)) ;

    //--------------------------------------------------------------------------
    // get I and J
    //--------------------------------------------------------------------------

    GrB_Vector I = NULL, J = NULL ;
    int icells = 0, jcells = 0 ;
    int base_offset = (base == BASE_0_INT) ? 0 : 1 ;
    int64_t I_max = -1, J_max = -1 ;
    uint64_t nI, nJ ;

    if (cnrows > 1 && cncols > 1 && ncells == 1)
    {
        ERROR ("Linear indexing not supported") ;
    }

    if (cnrows == 1 && ncells == 1)
    { 
        // only J is present
        J = gb_mxcell_to_list (Cell [0], base_offset, cncols, &nJ, &J_max) ;
        jcells = mxGetNumberOfElements (Cell [0]) ;
    }
    else if (ncells == 1)
    { 
        // only I is present
        I = gb_mxcell_to_list (Cell [0], base_offset, cnrows, &nI, &I_max) ;
        icells = mxGetNumberOfElements (Cell [0]) ;
    }
    else if (ncells == 2)
    { 
        // both I and J are present
        I = gb_mxcell_to_list (Cell [0], base_offset, cnrows, &nI, &I_max) ;
        J = gb_mxcell_to_list (Cell [1], base_offset, cncols, &nJ, &J_max) ;
        icells = mxGetNumberOfElements (Cell [0]) ;
        jcells = mxGetNumberOfElements (Cell [1]) ;
    }

    if (icells > 1)
    { 
        // I is a 3-element vector containing a stride
        OK (GrB_Descriptor_set_INT32 (desc, GxB_IS_STRIDE, GxB_ROWINDEX_LIST)) ;
    }

    if (jcells > 1)
    { 
        // J is a 3-element vector containing a stride
        OK (GrB_Descriptor_set_INT32 (desc, GxB_IS_STRIDE, GxB_COLINDEX_LIST)) ;
    }

    //--------------------------------------------------------------------------
    // expand C if needed
    //--------------------------------------------------------------------------

    uint64_t cnrows_required = I_max + 1 ;
    uint64_t cncols_required = J_max + 1 ;
    if (cnrows_required > cnrows || cncols_required > cncols)
    {
        uint64_t cnrows_new = MAX (cnrows, cnrows_required) ;
        uint64_t cncols_new = MAX (cncols, cncols_required) ;
        OK (GrB_Matrix_resize (C, cnrows_new, cncols_new)) ;
    }

    //--------------------------------------------------------------------------
    // determine if A is a scalar (ignore the transpose descriptor)
    //--------------------------------------------------------------------------

    uint64_t anrows, ancols ;
    OK (GrB_Matrix_nrows (&anrows, A)) ;
    OK (GrB_Matrix_ncols (&ancols, A)) ;
    bool scalar_assignment = (anrows == 1) && (ancols == 1) ;

    //--------------------------------------------------------------------------
    // compute C(I,J)<M> += A or C<M>(I,J) += A
    //--------------------------------------------------------------------------

    if (scalar_assignment)
    { 
        if (do_subassign)
        {
            // C(I,J)<M> += scalar
            OK1 (C, GxB_Matrix_subassign_Scalar_Vector (C, M, accum,
                (GrB_Scalar) A, I, J, desc)) ;
        }
        else
        {
            // C<M>(I,J) += scalar
            OK1 (C, GxB_Matrix_assign_Scalar_Vector (C, M, accum,
                (GrB_Scalar) A, I, J, desc)) ;
        }
    }
    else
    {
        if (do_subassign)
        { 
            // C(I,J)<M> += A
            OK1 (C, GxB_Matrix_subassign_Vector (C, M, accum, A, I, J, desc)) ;
        }
        else
        { 
            // C<M>(I,J) += A
            OK1 (C, GxB_Matrix_assign_Vector (C, M, accum, A, I, J, desc)) ;
        }
    }

    //--------------------------------------------------------------------------
    // free shallow copies and workspace
    //--------------------------------------------------------------------------

    OK (GrB_Matrix_free (&M)) ;
    OK (GrB_Matrix_free (&A)) ;
    OK (GrB_Vector_free (&I)) ;
    OK (GrB_Vector_free (&J)) ;
    OK (GrB_Descriptor_free (&desc)) ;

    //--------------------------------------------------------------------------
    // export the output matrix C
    //--------------------------------------------------------------------------

    pargout [0] = gb_export (&C, kind) ;
    pargout [1] = mxCreateDoubleScalar (kind) ;
    gb_wrapup ( ) ;
}

