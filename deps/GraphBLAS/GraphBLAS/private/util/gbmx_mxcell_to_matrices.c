//------------------------------------------------------------------------------
// gbmx_mxcell_to_matrices: convert cell array to a set of matrices
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

void gbmx_mxcell_to_matrices
(
    // output
    struct gb_matrix_struct Cell_Matrix [3], // matrix contents of the Cell
    int *len,                   // # of items in the Cell
    // input
    const mxArray *Cell         // built-in MATLAB cell array (at most 3 items)
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (Cell == NULL || !mxIsCell (Cell), "internal error 17") ;
    (*len) = mxGetNumberOfElements (Cell) ;
    CHECK_ERROR ((*len) > 3, "index must be a cell array of length 0 to 3") ;

    //--------------------------------------------------------------------------
    // get the contents of Cell
    //--------------------------------------------------------------------------

    for (int k = 0 ; k < (*len) ; k++)
    { 
        gbmx_get_matrix (&(Cell_Matrix [k]), mxGetCell (Cell, k)) ;
    }
}

