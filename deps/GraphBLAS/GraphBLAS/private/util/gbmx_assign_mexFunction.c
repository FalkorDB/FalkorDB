//------------------------------------------------------------------------------
// gbmx_assign_mexFunction: assign entries into a GraphBLAS matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// With do_subassign false, gbmx_assign_mexFunction is an interface to
// GrB_Matrix_assign and GrB_Matrix_assign_[TYPE], computing the GraphBLAS
// expression:

//      C<#M,replace>(I,J) = accum (C(I,J), A) or accum(C(I,J), A')

// With do_subassign true, gbmx_assign_mexFunction is an interface to
// GxB_Matrix_subassign and GxB_Matrix_subassign_[TYPE], computing the
// GraphBLAS expression:

//      C(I,J)<#M,replace> = accum (C(I,J), A) or accum(C(I,J), A')

// A can be a matrix or a scalar.  If it is a scalar with nnz (A) == 0, then it
// is first expanded to an empty matrix of size length(I)-by-length(J), and
// G*B_Matrix_*assign is used (not GraphBLAS scalar assignment).

// This method is in the util folder, but it is an entire mexFunction, not a
// utility.

#undef  FREE_WORK
#define FREE_WORK                   \
    GrB_Matrix_free (&M_to_free) ;  \
    GrB_Matrix_free (&A_to_free) ;  \
    GrB_Vector_free (&I_to_free) ;  \
    GrB_Vector_free (&J_to_free) ;  \
    GrB_Descriptor_free (&desc) ;

#undef  FREE_ALL
#define FREE_ALL                    \
    FREE_WORK ;                     \
    GrB_Matrix_free (&C) ;

void gbmx_assign_mexFunction    // gbmex_assign or gbmex_subassign mexFunctions
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
    // check inputs and construct inputs
    //--------------------------------------------------------------------------

    GrB_Type ctype ;
    GrB_Matrix *C_opaque = NULL, C = NULL, M = NULL, A = NULL,
        M_to_free = NULL, A_to_free = NULL ;
    GrB_Vector I = NULL, J = NULL, I_to_free = NULL, J_to_free = NULL ;
    GrB_Descriptor desc = NULL ;

    GBMX_USAGE (nargin >= 3 && nargin <= 8 && nargout <= 2, usage) ;
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
        if (do_subassign)
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
    struct gb_matrix_struct Cell0_Matrix [3] ;
    struct gb_matrix_struct Cell1_Matrix [3] ;
    mxArray *Cell [2] ;
    char String [2][LEN+2] ;
    int nmatrices, nstrings, ncells ;
    struct gb_descriptor_struct gbdesc ;
    gbmx_get_mxargs (nargin, pargin, usage, Matrix, &nmatrices, String,
        &nstrings, Cell, &ncells, &gbdesc) ;

    CHECK_ERROR (nmatrices < 2 || nmatrices > 3 || nstrings > 1, usage) ;

    int Cell0_len = 0, Cell1_len = 0 ;
    if (ncells > 0)
    { 
        gbmx_mxcell_to_matrices (Cell0_Matrix, &Cell0_len, Cell [0]) ;
    }
    if (ncells > 1)
    { 
        gbmx_mxcell_to_matrices (Cell1_Matrix, &Cell1_len, Cell [1]) ;
    }

    ////////////////////////////////////////////////////////////////////////////

    //--------------------------------------------------------------------------
    // create the descriptor
    //--------------------------------------------------------------------------

    gbdesc.nondefault = true ;      // ensure the GrB_Descriptor is allocated
    OK (gb_get_descriptor (&desc, &gbdesc, arena, err)) ;
    ASSERT (desc != NULL) ;

    //--------------------------------------------------------------------------
    // get the matrices
    //--------------------------------------------------------------------------

    if (nmatrices == 2)
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

    OK (GxB_Matrix_type (&ctype, C)) ;

    //--------------------------------------------------------------------------
    // get the operator
    //--------------------------------------------------------------------------

    GrB_BinaryOp accum = NULL ;

    if (nstrings == 1)
    { 
        OK (gb_string_to_binop (&accum, String [0], ctype, ctype, err)) ;
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

    int icells = 0, jcells = 0 ;
    int base_offset = (gbdesc.base == BASE_0_INT) ? 0 : 1 ;
    int64_t I_max = -1, J_max = -1 ;
    uint64_t nI, nJ ;

    if (cnrows > 1 && cncols > 1 && ncells == 1)
    {
        ERROR ("Linear indexing not supported", GrB_NOT_IMPLEMENTED) ;
    }

    if (cnrows == 1 && ncells == 1)
    { 
        // only J is present
        OK (gb_cell_to_list (&J, &J_to_free, &nJ, &J_max,
            Cell0_Matrix, Cell0_len, base_offset, cncols, arena, err)) ;
        jcells = Cell0_len ;
    }
    else if (ncells == 1)
    { 
        // only I is present
        OK (gb_cell_to_list (&I, &I_to_free, &nI, &I_max,
            Cell0_Matrix, Cell0_len, base_offset, cnrows, arena, err)) ;
        icells = Cell0_len ;
    }
    else if (ncells == 2)
    { 
        // both I and J are present
        OK (gb_cell_to_list (&I, &I_to_free, &nI, &I_max,
            Cell0_Matrix, Cell0_len, base_offset, cnrows, arena, err)) ;
        OK (gb_cell_to_list (&J, &J_to_free, &nJ, &J_max,
            Cell1_Matrix, Cell1_len, base_offset, cncols, arena, err)) ;
        icells = Cell0_len ;
        jcells = Cell1_len ;
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

