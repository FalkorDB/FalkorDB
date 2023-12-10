//------------------------------------------------------------------------------
// GB_subassign: C(Rows,Cols)<M> = accum (C(Rows,Cols),A) or A'
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// submatrix assignment: C(Rows,Cols)<M> = accum (C(Rows,Cols),A)

// All GxB_*_subassign operations rely on this function.

// With scalar_expansion = false, this method does the work for the standard
// GxB_*_subassign operations (GxB_Matrix_subassign, GxB_Vector_subassign,
// GxB_Row_subassign, and GxB_Col_subassign).  If scalar_expansion is true, it
// performs scalar assignment (the GxB_*_subassign_TYPE functions) in which
// case the input matrix A is ignored (it is NULL), and the scalar is used
// instead.

// Compare with GB_assign, which uses M and C_replace differently

#include "GB_subassign.h"
#include "GB_bitmap_assign.h"

#define GB_FREE_ALL                 \
{                                   \
    GB_Matrix_free (&Cwork) ;       \
    GB_Matrix_free (&Mwork) ;       \
    GB_Matrix_free (&Awork) ;       \
    GB_FREE_WORK (&I2, I2_size) ;   \
    GB_FREE_WORK (&J2, J2_size) ;   \
}

GrB_Info GB_subassign               // C(Rows,Cols)<M> += A or A'
(
    GrB_Matrix C_in,                // input/output matrix for results
    bool C_replace,                 // descriptor for C
    const GrB_Matrix M_in,          // optional mask for C(Rows,Cols)
    const bool Mask_comp,           // true if mask is complemented
    const bool Mask_struct,         // if true, use the only structure of M
    const bool M_transpose,         // true if the mask should be transposed
    const GrB_BinaryOp accum,       // optional accum for accum(C,T)
    const GrB_Matrix A_in,          // input matrix
    const bool A_transpose,         // true if A is transposed
    const GrB_Index *Rows,          // row indices
    const GrB_Index nRows_in,       // number of row indices
    const GrB_Index *Cols,          // column indices
    const GrB_Index nCols_in,       // number of column indices
    const bool scalar_expansion,    // if true, expand scalar to A
    const void *scalar,             // scalar to be expanded
    const GB_Type_code scalar_code, // type code of scalar to expand
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check and prep inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GrB_Matrix C = NULL ;           // C_in or Cwork
    GrB_Matrix M = NULL ;           // M_in or Mwork
    GrB_Matrix A = NULL ;           // A_in or Awork
    GrB_Index *I = NULL ;           // Rows, Cols, or I2
    GrB_Index *J = NULL ;           // Rows, Cols, or J2

    // temporary matrices and arrays
    GrB_Matrix Cwork = NULL ;
    GrB_Matrix Mwork = NULL ;
    GrB_Matrix Awork = NULL ;
    struct GB_Matrix_opaque
        Cwork_header, Mwork_header, Awork_header, MT_header, AT_header ;
    GrB_Index *I2 = NULL ; size_t I2_size = 0 ;
    GrB_Index *J2 = NULL ; size_t J2_size = 0 ;

    GrB_Type scalar_type = NULL ;
    int64_t ni, nj, nI, nJ, Icolon [3], Jcolon [3] ;
    int Ikind, Jkind ;
    int assign_kind = GB_SUBASSIGN ;
    int subassign_method ;

    GB_OK (GB_assign_prep (&C, &M, &A, &subassign_method,
        &Cwork, &Mwork, &Awork,
        &Cwork_header, &Mwork_header, &Awork_header, &MT_header, &AT_header,
        &I, &I2, &I2_size, &ni, &nI, &Ikind, Icolon,
        &J, &J2, &J2_size, &nj, &nJ, &Jkind, Jcolon,
        &scalar_type, C_in, &C_replace, &assign_kind,
        M_in, Mask_comp, Mask_struct, M_transpose, accum,
        A_in, A_transpose, Rows, nRows_in, Cols, nCols_in,
        scalar_expansion, scalar, scalar_code, Werk)) ;

    // GxB_Row_subassign, GxB_Col_subassign, GxB_Matrix_subassign and
    // GxB_Vector_subassign all use GB_SUBASSIGN.
    ASSERT (assign_kind == GB_SUBASSIGN) ;

    if (subassign_method == 0)
    { 
        // GB_assign_prep has handled the entire assignment itself
        ASSERT (C == C_in) ;
        ASSERT_MATRIX_OK (C_in, "Final C for subassign", GB0) ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // C(I,J)<M> = A or accum (C(I,J),A) via GB_subassigner
    //--------------------------------------------------------------------------

    GB_OK (GB_subassigner (C, subassign_method, C_replace,
        M, Mask_comp, Mask_struct, accum, A,
        I, ni, nI, Ikind, Icolon, J, nj, nJ, Jkind, Jcolon,
        scalar_expansion, scalar, scalar_type, Werk)) ;

    //--------------------------------------------------------------------------
    // transplant C back into C_in
    //--------------------------------------------------------------------------

    if (C == Cwork)
    { 
        // Transplant the content of Cwork into C_in and free Cwork.  Zombies
        // and pending tuples can be transplanted from Cwork into C_in, and if
        // Cwork is jumbled, C_in becomes jumbled too.
        ASSERT (Cwork->static_header || GBNSTATIC) ;
        GB_OK (GB_transplant (C_in, C_in->type, &Cwork, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // free workspace, finalize C, and return result
    //--------------------------------------------------------------------------

    GB_OK (GB_conform (C_in, Werk)) ;
    ASSERT_MATRIX_OK (C_in, "Final C for subassign", GB0) ;
    GB_FREE_ALL ;
    return (GB_block (C_in, Werk)) ;
}

