//------------------------------------------------------------------------------
// GB_extract: C<M> = accum(C,A(I,J))
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Not user-callable.  Implements the user-callable GrB_*_extract functions.
//
//      C<M> = accum (C, A (Rows,Cols)) or
//      C<M> = accum (C, AT(Rows,Cols)) where AT = A'
//
// equivalently:
//
//      C<M> = accum (C, A(Rows,Cols) )
//      C<M> = accum (C, A(Cols,Rows)')

#define GB_FREE_ALL GrB_Matrix_free (&T) ;

#include "GB_extract.h"
#include "GB_subref.h"
#include "GB_accum_mask.h"

GrB_Info GB_extract                 // C<M> = accum (C, A(I,J))
(
    GrB_Matrix C,                   // input/output matrix for results
    const bool C_replace,           // C matrix descriptor
    const GrB_Matrix M,             // optional mask for C, unused if NULL
    const bool Mask_comp,           // mask descriptor
    const bool Mask_struct,         // if true, use the only structure of M
    const GrB_BinaryOp accum,       // optional accum for Z=accum(C,T)
    const GrB_Matrix A,             // input matrix
    const bool A_transpose,         // A matrix descriptor
    const GrB_Index *Rows,          // row indices
    const GrB_Index nRows_in,       // number of row indices
    const GrB_Index *Cols,          // column indices
    const GrB_Index nCols_in,       // number of column indices
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    // C may be aliased with M and/or A

    GrB_Info info ;
    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = NULL ;
    GB_RETURN_IF_NULL (Rows) ;
    GB_RETURN_IF_NULL (Cols) ;
    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;

    ASSERT_MATRIX_OK (C, "C input for GB_Matrix_extract", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for GB_Matrix_extract", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GB_Matrix_extract", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for GB_Matrix_extract", GB0) ;

    // check domains and dimensions for C<M> = accum (C,T)
    GB_OK (GB_compatible (C->type, C, M, Mask_struct, accum, A->type,
        Werk)) ;

    // check the dimensions of C
    int64_t cnrows = GB_NROWS (C) ;
    int64_t cncols = GB_NCOLS (C) ;

    int64_t nRows, nCols, RowColon [3], ColColon [3] ;
    int rkind, ckind ;

    if (!A_transpose)
    { 
        // T = A(Rows,Cols)
        GB_ijlength (Rows, nRows_in, GB_NROWS (A), &nRows, &rkind, RowColon) ;
        GB_ijlength (Cols, nCols_in, GB_NCOLS (A), &nCols, &ckind, ColColon) ;
    }
    else
    { 
        // T = A(Cols,Rows)
        GB_ijlength (Rows, nRows_in, GB_NCOLS (A), &nRows, &rkind, RowColon) ;
        GB_ijlength (Cols, nCols_in, GB_NROWS (A), &nCols, &ckind, ColColon) ;
    }

    if (cnrows != nRows || cncols != nCols)
    { 
        GB_ERROR (GrB_DIMENSION_MISMATCH,
            "Dimensions not compatible:\n"
            "required size of output is " GBd "-by-" GBd "\n"
            "but actual size output is  " GBd "-by-" GBd "\n",
            nRows, nCols, cnrows, cncols) ;
    }

    // quick return if an empty mask is complemented
    GB_RETURN_IF_QUICK_MASK (C, C_replace, M, Mask_comp, Mask_struct) ;

    // delete any lingering zombies and assemble any pending tuples
    GB_MATRIX_WAIT (M) ;        // cannot be jumbled
    GB_MATRIX_WAIT (A) ;        // cannot be jumbled

    GB_BURBLE_DENSE (C, "(C %s) ") ;
    GB_BURBLE_DENSE (M, "(M %s) ") ;
    GB_BURBLE_DENSE (A, "(A %s) ") ;

    //--------------------------------------------------------------------------
    // handle the CSR/CSC format and transpose; T = A (I,J) or T = A (J,I)
    //--------------------------------------------------------------------------

    const GrB_Index *I, *J ;
    int64_t ni, nj ;
    bool T_is_csc ;

    if (A->is_csc)
    {
        if (!A_transpose)
        { 
            // T = A(Rows,Cols) where both A and T are in CSC format
            I = Rows ; ni = nRows_in ;  // indices into the vectors
            J = Cols ; nj = nCols_in ;  // vectors
            T_is_csc = true ;           // return T in CSC format
        }
        else
        { 
            // T = A(Cols,Rows) where A is CSC and T is returned as CSR format
            I = Cols ; ni = nCols_in ;  // indices into the vectors
            J = Rows ; nj = nRows_in ;  // vectors
            T_is_csc = false ;          // return T in CSR format
        }
    }
    else
    {
        if (!A_transpose)
        { 
            // T = A(Rows,Cols) where both A and T are in CSR format
            I = Cols ; ni = nCols_in ;  // indices into the vectors
            J = Rows ; nj = nRows_in ;  // vectors
            T_is_csc = false ;          // return T in CSR format
        }
        else
        { 
            // T = A(Cols,Rows) where A is CSR but T is returned as CSC format
            I = Rows ; ni = nRows_in ;  // indices into the vectors
            J = Cols ; nj = nCols_in ;  // vectors
            T_is_csc = true ;           // return T in CSC format
        }
    }

    // T has T->vdim = |J|, each vector of length T->vlen = |J|, regardless of
    // its CSR/CSC format.

    // J is a list of length |J| of vectors in the range 0:A->vdim-1
    // I is a list of length |I| of indices in the range 0:A->vlen-1

    // |I| and |J| are either nRows or nCols, depending on the 4 cases above.

    // T has the same hypersparsity as A.

    //--------------------------------------------------------------------------
    // T = A (I,J)
    //--------------------------------------------------------------------------

    // TODO::: iso:  if accum is PAIR, extract T as iso

    GB_CLEAR_STATIC_HEADER (T, &T_header) ;
    GB_OK (GB_subref (T, false, T_is_csc, A, I, ni, J, nj, false, Werk)) ;
    ASSERT_MATRIX_OK (T, "T extracted", GB0) ;
    ASSERT (GB_JUMBLED_OK (T)) ;

    //--------------------------------------------------------------------------
    // C<M> = accum (C,T): accumulate the results into C via the mask M
    //--------------------------------------------------------------------------

    return (GB_accum_mask (C, M, NULL, accum, &T, C_replace, Mask_comp,
        Mask_struct, Werk)) ;
}

