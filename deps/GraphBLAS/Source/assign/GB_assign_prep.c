//------------------------------------------------------------------------------
// GB_assign_prep: check and prepare inputs for GB_assign and GB_subassign
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_assign_prep checks the inputs for GB_assign and GB_subassign.

#include "assign/GB_subassign.h"
#include "assign/GB_bitmap_assign.h"
#include "assign/GB_assign_zombie.h"
#include "assign/GB_subassign_methods.h"
#include "transpose/GB_transpose.h"
#include "extract/GB_subref.h"
#include "include/GB_unused.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL                 \
{                                   \
    GB_Matrix_free (&Cwork) ;       \
    GB_Matrix_free (&Awork) ;       \
    GB_Matrix_free (&AT) ;          \
    GB_Matrix_free (&Mwork) ;       \
    GB_Matrix_free (&MT) ;          \
    GB_FREE_MEMORY (&I2, I2_size) ;   \
    GB_FREE_MEMORY (&J2, J2_size) ;   \
    GB_FREE_MEMORY (&I2k, I2k_size) ; \
    GB_FREE_MEMORY (&J2k, J2k_size) ; \
}

GrB_Info GB_assign_prep
(
    // output:
    GrB_Matrix *Chandle,            // C_in, or Cwork if C is aliased to M or A
    GrB_Matrix *Mhandle,            // M_in, or a modified version Mwork
    GrB_Matrix *Ahandle,            // A_in, or a modified version Awork
    int *subassign_method,          // subassign method to use

    // modified versions of the matrices C, M, and A:
    GrB_Matrix *Cwork_handle,          // NULL, or a copy of C
    GrB_Matrix *Mwork_handle,          // NULL, or a temporary matrix
    GrB_Matrix *Awork_handle,          // NULL, or a temporary matrix

    // static headers for Cwork, Mwork, Awork, MT and AT
    GrB_Matrix Cwork_header_handle,
    GrB_Matrix Mwork_header_handle,
    GrB_Matrix Awork_header_handle,
    GrB_Matrix MT_header_handle,
    GrB_Matrix AT_header_handle,

    // modified versions of the Rows/Cols lists, and their analysis:
    void **I_handle,            // Rows, Cols, or a modified copy I2
    bool *I_is_32_handle,
    void **I2_handle,           // NULL, or sorted/pruned Rows or Cols
    size_t *I2_size_handle,
    int64_t *ni_handle,
    int64_t *nI_handle,
    int *Ikind_handle,
    int64_t Icolon [3],

    void **J_handle,            // Rows, Cols, or a modified copy J2
    bool *J_is_32_handle,
    void **J2_handle,           // NULL, or sorted/pruned Rows or Cols
    size_t *J2_size_handle,
    int64_t *nj_handle,
    int64_t *nJ_handle,
    int *Jkind_handle,
    int64_t Jcolon [3],

    GrB_Type *scalar_type_handle,   // type of the scalar, or NULL if no scalar

    // input/output
    GrB_Matrix C_in,                // input/output matrix for results
    bool *C_replace,                // descriptor for C
    int *assign_kind,               // row/col assign, assign, or subassign

    // input
    const GrB_Matrix M_in,          // optional mask for C
    const bool Mask_comp,           // true if mask is complemented
    const bool Mask_struct,         // if true, use the only structure of M
    bool M_transpose,               // true if the mask should be transposed
    const GrB_BinaryOp accum,       // optional accum for accum(C,T)
    const GrB_Matrix A_in,          // input matrix
    bool A_transpose,               // true if A is transposed
    const void *Rows,               // row indices
    const bool Rows_is_32,          // if true, Rows is 32-bit; else 64-bit
    const uint64_t nRows_in,        // number of row indices
    const void *Cols,               // column indices
    const bool Cols_is_32,          // if true, Rows is 32-bit; else 64-bit
    const uint64_t nCols_in,        // number of column indices
    const bool scalar_expansion,    // if true, expand scalar to A
    const void *scalar,             // scalar to be expanded
    const GB_Type_code scalar_code, // type code of scalar to expand
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    GB_RETURN_IF_FAULTY_OR_POSITIONAL (accum) ;
    GB_RETURN_IF_NULL (Rows) ;
    GB_RETURN_IF_NULL (Cols) ;

    GrB_Matrix C = C_in ;
    GrB_Matrix M = M_in ;
    GrB_Matrix A = A_in ;

    ASSERT_MATRIX_OK (C, "C input for GB_assign_prep", GB0) ;
    ASSERT (!GB_is_shallow (C)) ;
    ASSERT_MATRIX_OK_OR_NULL (M, "M for GB_assign_prep", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GB_assign_prep", GB0) ;
    ASSERT (scalar_code <= GB_UDT_code) ;

    GrB_Matrix Cwork = NULL ;
    GrB_Matrix Mwork = NULL ;
    GrB_Matrix Awork = NULL ;
    GrB_Matrix MT = NULL ;
    GrB_Matrix AT = NULL ;

    void *I2  = NULL ; size_t I2_size = 0 ;
    void *J2  = NULL ; size_t J2_size = 0 ;
    void *I2k = NULL ; size_t I2k_size = 0 ;
    void *J2k = NULL ; size_t J2k_size = 0 ;

    (*scalar_type_handle) = NULL ;

    (*Chandle) = NULL ;
    (*Mhandle) = NULL ;
    (*Ahandle) = NULL ;

    (*Cwork_handle) = NULL ;
    (*Awork_handle) = NULL ;
    (*Mwork_handle) = NULL ;

    (*I_handle) = NULL ; 
    (*I2_handle) = NULL ;
    (*I2_size_handle) = 0 ;
    (*ni_handle) = 0 ;
    (*nI_handle) = 0 ;
    (*Ikind_handle) = 0 ;

    (*J_handle) = NULL ;
    (*J2_handle) = NULL ;
    (*J2_size_handle) = 0 ;
    (*nj_handle) = 0 ;
    (*nJ_handle) = 0 ;
    (*Jkind_handle) = 0 ;

    //--------------------------------------------------------------------------
    // determine the type of A or the scalar
    //--------------------------------------------------------------------------

    GrB_Type atype, scalar_type = NULL ;
    GrB_Type ctype = C->type ;
    if (scalar_expansion)
    { 
        // for scalar expansion, the NULL pointer case has been already checked
        // for user-defined types, and can't be NULL for built-in types.
        ASSERT (scalar != NULL) ;
        ASSERT (A == NULL) ;
        ASSERT ((*assign_kind) == GB_ASSIGN || (*assign_kind) == GB_SUBASSIGN) ;
        scalar_type = GB_code_type (scalar_code, ctype) ;
        atype = scalar_type ;
    }
    else
    { 
        // GrB_*assign, not scalar:  The user's input matrix has been checked.
        // The pointer to the scalar is NULL.
        ASSERT (scalar == NULL) ;
        ASSERT (A != NULL) ;
        ASSERT_MATRIX_OK (A, "A for GB_assign_prep", GB0) ;
        atype = A->type ;
    }

    //--------------------------------------------------------------------------
    // delete any lingering zombies and assemble any pending tuples
    //--------------------------------------------------------------------------

    // zombies and pending tuples in C or OK, but not M or A
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (M) ;
    GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (A) ;

    // some kernels allow for M and A to be jumbled
    ASSERT (GB_JUMBLED_OK (M)) ;
    ASSERT (GB_JUMBLED_OK (A)) ;

    // C can have any kind of pending work
    ASSERT (GB_ZOMBIES_OK (C)) ;
    ASSERT (GB_JUMBLED_OK (C)) ;
    ASSERT (GB_PENDING_OK (C)) ;

    //--------------------------------------------------------------------------
    // check domains of C, M, A, and accum
    //--------------------------------------------------------------------------

    // GB_compatible is not used since most of it is slightly different here
    if (accum != NULL)
    { 
        // C<M>(Rows,Cols) = accum (C(Rows,Cols),A), or
        // C(Rows,Cols)<M> = accum (C(Rows,Cols),A)
        GB_OK (GB_BinaryOp_compatible (accum, ctype, ctype,
            (scalar_expansion) ? NULL : atype,
            (scalar_expansion) ? scalar_code : GB_ignore_code, Werk)) ;
    }

    // C<M>(Rows,Cols) = T, so C and T must be compatible.
    // also C<M>(Rows,Cols) = accum(C,T) for entries in T but not C
    if (scalar_expansion)
    {
        if (!GB_code_compatible (ctype->code, scalar_code))
        { 
            GB_ERROR (GrB_DOMAIN_MISMATCH, "Input scalar of type [%s]\n"
                "cannot be typecast to output of type [%s]",
                GB_code_string (scalar_code), ctype->name) ;
        }
    }
    else
    {
        if (!GB_Type_compatible (ctype, atype))
        { 
            GB_ERROR (GrB_DOMAIN_MISMATCH, "Input of type [%s]\n"
                "cannot be typecast to output of type [%s]",
                atype->name, ctype->name) ;
        }
    }

    if (M != NULL && !Mask_struct)
    {
        // M is typecast to boolean
        if (!GB_Type_compatible (M->type, GrB_BOOL))
        { 
            GB_ERROR (GrB_DOMAIN_MISMATCH,
                "M of type [%s] cannot be typecast to boolean", M->type->name) ;
        }
    }

    //--------------------------------------------------------------------------
    // determine the properites of the Rows and Cols index lists
    //--------------------------------------------------------------------------

    int64_t nRows, nCols, RowColon [3], ColColon [3] ;
    int RowsKind, ColsKind ;
    GB_ijlength (Rows, Rows_is_32, nRows_in, GB_NROWS (C), &nRows, &RowsKind,
        RowColon) ;
    GB_ijlength (Cols, Cols_is_32, nCols_in, GB_NCOLS (C), &nCols, &ColsKind,
        ColColon) ;

    //--------------------------------------------------------------------------
    // check the dimensions of M
    //--------------------------------------------------------------------------

    if (M != NULL)
    {
        // check the mask: size depends on the method

        switch (*assign_kind)
        {
            case GB_ROW_ASSIGN : 
            {
                // GrB_Row_assign:
                // M is a column vector the same size as one row of C
                ASSERT (nRows == 1) ;
                ASSERT (!scalar_expansion) ;
                ASSERT (GB_VECTOR_OK (M)) ;
                if (GB_NROWS (M) != GB_NCOLS (C))
                { 
                    GB_ERROR (GrB_DIMENSION_MISMATCH, "Mask vector m length"
                        " is " GBd "; must match the number of columns of C ("
                        GBd ")", GB_NROWS (M), GB_NCOLS (C)) ;
                }
            }
            break ;

            case GB_COL_ASSIGN : 
            {
                // GrB_Col_assign:
                // M is a column vector the same size as one column of C
                ASSERT (nCols == 1) ;
                ASSERT (!scalar_expansion) ;
                ASSERT (GB_VECTOR_OK (M)) ;
                if (GB_NROWS (M) != GB_NROWS (C))
                { 
                    GB_ERROR (GrB_DIMENSION_MISMATCH, "Mask vector m length"
                        " is " GBd "; must match the number of rows of C ("
                        GBd ")", GB_NROWS (M), GB_NROWS (C)) ;
                }
            }
            break ;

            case GB_ASSIGN : 
            {
                // GrB_Matrix_assign, GrB_Vector_assign, and scalar variants: M
                // is a matrix the same size as C for entire matrix (or vector)
                // assignment, where A is either a matrix or a scalar
                if (GB_NROWS (M) != GB_NROWS (C) ||
                    GB_NCOLS (M) != GB_NCOLS (C))
                { 
                    GB_ERROR (GrB_DIMENSION_MISMATCH, "Mask M is " GBd "-by-"
                        GBd "; " "must match result C (" GBd "-by-" GBd ")",
                        GB_NROWS (M), GB_NCOLS (M),
                        GB_NROWS (C), GB_NCOLS (C)) ;
                }
            }
            break ;

            case GB_SUBASSIGN : 
            {
                // GxB_subassign: M is a matrix the same size as C(Rows,Cols)
                int64_t mnrows = M_transpose ? GB_NCOLS (M) : GB_NROWS (M) ;
                int64_t mncols = M_transpose ? GB_NROWS (M) : GB_NCOLS (M) ;
                if (mnrows != nRows || mncols != nCols)
                { 
                    GB_ERROR (GrB_DIMENSION_MISMATCH,
                        "M is " GBd "-by-" GBd "%s, "
                        "must match size of result C(I,J): " GBd "-by-" GBd "",
                        mnrows, mncols, M_transpose ? " (transposed)" : "",
                        nRows, nCols) ;
                }
            }
            break ;

            default:
                ASSERT (GB_DEAD_CODE) ;
        }
    }

    //--------------------------------------------------------------------------
    // check the dimensions of A
    //--------------------------------------------------------------------------

    if (!scalar_expansion)
    {
        int64_t anrows = (A_transpose) ? GB_NCOLS (A) : GB_NROWS (A) ;
        int64_t ancols = (A_transpose) ? GB_NROWS (A) : GB_NCOLS (A) ;
        if (nRows != anrows || nCols != ancols)
        { 
            GB_ERROR (GrB_DIMENSION_MISMATCH,
                "Dimensions not compatible:\n"
                "C(Rows,Cols) is " GBd "-by-" GBd "\n"
                "input is " GBd "-by-" GBd "%s",
                nRows, nCols, anrows, ancols,
                A_transpose ? " (transposed)" : "") ;
        }
    }

    //--------------------------------------------------------------------------
    // handle the CSR/CSC format of C:
    //--------------------------------------------------------------------------

    // GrB_Row_assign, GxB_Row_subassign: A is always a vector in CSC format,
    // and A_transpose is always true.  If C is in CSC format then A_transpose
    // remains true, and the n-by-1 vector A is transposed below into a 1-by-n
    // hypersparse CSC matrix.  If C is in CSR format then A_transpose becomes
    // false, and the assignment does not need to transpose A.  It remains in
    // CSC format but has the correct vector length and dimension for the
    // CSR/CSC-agnostic assignment.

    // GrB_Col_assign, GxB_Col_subassign: A is always a vector in CSC format,
    // and A_transpose is always false.  If C is in CSC format then A_transpose
    // remains false, and the assignment does not need to transpose A.  If C is
    // in CSR format then A_transpose becomes true, and the the n-by-1 vector A
    // is transposed below into a 1-by-n hypersparse CSC matrix.  The CSC
    // format is ignored by the CSR/CSC-agnostic assignment.

    // GrB_Vector_assign, GxB_Vector_subassign:  both A and C are always in CSC
    // format, and A_transpose is always false, and doesn't change below.

    // GrB_Matrix_assign, GxB_Matrix_subassign:  A and C can be in any format,
    // and A_transpose can be true or false, depending on the descriptor.  If
    // the CSR/CSC formats of A and C are the same, then A_transpose remains
    // as-is.  If they differ, then A_transpose is negated.  Then the CSR-CSC
    // agnostic assignment proceeds.

    bool C_is_csc = C->is_csc ;
    if (!scalar_expansion && C_is_csc != A->is_csc)
    { 
        // negate the sense of A_transpose
        A_transpose = !A_transpose ;
    }

    // get the I and J index lists
    int Ikind, Jkind ;
    GB_MDECL (I, const, u) ;
    GB_MDECL (J, const, u) ;
    int64_t ni, nj, nI, nJ ;

    if (C_is_csc)
    { 
        // C is in CSC format
        I      = Rows     ;     J      = Cols     ;
        ni     = nRows_in ;     nj     = nCols_in ;
        Ikind  = RowsKind ;     Jkind  = ColsKind ;
        nI     = nRows    ;     nJ     = nCols    ;
        memcpy (Icolon, RowColon, 3 * sizeof (int64_t)) ;
        memcpy (Jcolon, ColColon, 3 * sizeof (int64_t)) ;
    }
    else
    { 
        // C is in CSR format
        I       = Cols     ;    J       = Rows     ;
        ni      = nCols_in ;    nj      = nRows_in ;
        Ikind   = ColsKind ;    Jkind   = RowsKind ;
        nI      = nCols    ;    nJ      = nRows    ;
        memcpy (Icolon, ColColon, 3 * sizeof (int64_t)) ;
        memcpy (Jcolon, RowColon, 3 * sizeof (int64_t)) ;
        // flip the sense of row/col assign
        if ((*assign_kind) == GB_ROW_ASSIGN)
        { 
            // assignment to vector j = J [0], which is Rows [0]
            (*assign_kind) = GB_COL_ASSIGN ;
        }
        else if ((*assign_kind) == GB_COL_ASSIGN)
        { 
            // assignment to index i = I [0], which is Cols [0]
            (*assign_kind) = GB_ROW_ASSIGN ;
        }
    }

    bool I_is_32 = (C_is_csc) ? Rows_is_32 : Cols_is_32 ;
    bool J_is_32 = (C_is_csc) ? Cols_is_32 : Rows_is_32 ;
    GB_IPTR (I, I_is_32) ;
    GB_IPTR (J, J_is_32) ;

    // J is now a list of vectors in the range 0:C->vdim-1
    // I is now a list of indices in the range 0:C->vlen-1

    bool whole_C_matrix = (Ikind == GB_ALL && Jkind == GB_ALL) ;

    //--------------------------------------------------------------------------
    // quick return if an empty mask is complemented
    //--------------------------------------------------------------------------

    bool C_is_bitmap = GB_IS_BITMAP (C) ;
    int C_sparsity_control = GB_sparsity_control (C->sparsity_control, C->vdim);
    bool C_may_be_bitmap = (C_sparsity_control & GxB_BITMAP) ;
    bool use_bitmap_assign = (C_is_bitmap ||
        ((*C_replace) && GB_IS_FULL (C) && C_may_be_bitmap)) ;

    // an empty mask occurs when M is not present, but complemented

    if (M == NULL && Mask_comp)
    {

        //----------------------------------------------------------------------
        // C<!,replace or !replace>(I,J) = anything
        //----------------------------------------------------------------------

        // The mask M is empty, and complemented, and thus M(i,j)=0 for all i
        // and j.  The result does not depend on A or accum.  The output C is
        // either untouched (if C_replace is false) or cleared (if C_replace is
        // true).  However, GrB_Row_assign and GrB_Col_assign only clear their
        // specific row or column of C, respectively.  GB_subassign only clears
        // C(I,J).  GrB_assign clears all of C.

        // M is NULL so C and M cannot be the same, and A is ignored so it
        // doesn't matter whether or not C == A.  Thus C is not aliased to the
        // inputs.

        // This condition is like GB_RETURN_IF_QUICK_MASK(...), except that the
        // action taken by C_replace is different for row/col assign and
        // subassign.

        if (*C_replace)
        {

            //------------------------------------------------------------------
            // C<!,replace>(I,J) = anything
            //------------------------------------------------------------------

            ASSERT_MATRIX_OK (C, "C for quick mask", GB0) ;

            // to clear the whole C matrix: assign and subassign are the same

            switch (whole_C_matrix ? GB_ASSIGN : (*assign_kind))
            {

                //--------------------------------------------------------------
                // row assign: delete all entries in C(i,:)
                //--------------------------------------------------------------

                case GB_ROW_ASSIGN : 
                {
                    // delete all entries in each vector with index i
                    GB_MATRIX_WAIT_IF_PENDING (C) ;
                    if (use_bitmap_assign)
                    { 
                        // neither A nor the scalar are used, so convert this
                        // to a scalar assignment (the scalar is not used)
                        GBURBLE ("bitmap C(i,:)=zombie ") ;
                        int scalar_unused = 0 ;
                        GB_OK (GB_bitmap_assign (C, /* C_replace: */ true,
                            I,    I_is_32, 1, 1, GB_LIST, NULL, // I = [i]
                            NULL, false  , 0, 0, GB_ALL,  NULL, // J = [:]
                            /* no M: */ NULL,
                            /* Mask_comp: */ true,
                            /* Mask_struct: ignored */ false,
                            /* no accum: */ NULL,
                            /* no A: */ NULL,
                            /* scalar: */ &scalar_unused, GrB_INT32,
                            GB_ROW_ASSIGN, Werk)) ;
                    }
                    else
                    { 
                        GB_UNJUMBLE (C) ;
                        // ensure C is sparse; wait(C) may change it
                        if (GB_IS_BITMAP (C) || GB_IS_FULL (C))
                        {
                            GB_OK (GB_convert_any_to_sparse (C, Werk)) ;
                        }
                        GBURBLE ("C(i,:)=zombie ") ;
                        GB_OK (GB_assign_zombie2 (C, GB_IGET (I, 0))) ;
                    }
                }
                break ;

                //--------------------------------------------------------------
                // col assign: delete all entries in C(:,j)
                //--------------------------------------------------------------

                case GB_COL_ASSIGN : 
                {
                    GB_MATRIX_WAIT_IF_PENDING (C) ;
                    if (use_bitmap_assign)
                    { 
                        // neither A nor the scalar are used, so convert this
                        // to a scalar assignment (the scalar is not used)
                        GBURBLE ("bitmap C(:,j)=zombie ") ;
                        int scalar_unused = 0 ;
                        GB_OK (GB_bitmap_assign (C, /* C_replace: */ true,
                            NULL, false  , 0, 0, GB_ALL,  NULL, // I = [:]
                            J,    J_is_32, 1, 1, GB_LIST, NULL, // J = [j]
                            /* no M: */ NULL,
                            /* Mask_comp: */ true,
                            /* Mask_struct: ignored */ false,
                            /* no accum: */ NULL,
                            /* no A: */ NULL,
                            /* scalar: */ &scalar_unused, GrB_INT32,
                            GB_COL_ASSIGN, Werk)) ;
                    }
                    else
                    { 
                        // ensure C is sparse; wait(C) may change it
                        if (GB_IS_BITMAP (C) || GB_IS_FULL (C))
                        {
                            GB_OK (GB_convert_any_to_sparse (C, Werk)) ;
                        }
                        GBURBLE ("C(:,j)=zombie ") ;
                        GB_OK (GB_hyper_hash_build (C, Werk)) ;
                        GB_OK (GB_assign_zombie1 (C, GB_IGET (J, 0))) ;
                    }
                }
                break ;

                //--------------------------------------------------------------
                // assign: delete all entries in C
                //--------------------------------------------------------------

                case GB_ASSIGN : 
                {
                    // C<!>=anything since result does not depend on computing
                    // Z.  Since C_replace is true, all of C is cleared.  This
                    // is the same as the GB_RETURN_IF_QUICK_MASK macro.
                    // GB_clear either converts C to an empty sparse/hyper
                    // matrix, or to a bitmap matrix with no entries, depending
                    // on its sparsity control setting.
                    GBURBLE ("(clear C) ") ;
                    GB_OK (GB_clear (C, Werk)) ;
                }
                break ;

                //--------------------------------------------------------------
                // subassign: delete all entries in C(I,J)
                //--------------------------------------------------------------

                case GB_SUBASSIGN : 
                {
                    GB_MATRIX_WAIT_IF_PENDING (C) ;
                    if (use_bitmap_assign)
                    { 
                        // neither A nor the scalar are used, so convert this
                        // to a scalar assignment (the scalar is not used)
                        GBURBLE ("bitmap C(I,J)=zombie ") ;
                        int scalar_unused = 0 ;
                        GB_OK (GB_bitmap_assign (C, /* C_replace: */ true,
                            I, I_is_32, ni, nI, Ikind, Icolon,
                            J, J_is_32, nj, nJ, Jkind, Jcolon,
                            /* no M: */ NULL,
                            /* Mask_comp: */ true,
                            /* Mask_struct: ignored */ false,
                            /* no accum: */ NULL,
                            /* no A: */ NULL,
                            /* scalar: */ &scalar_unused, GrB_INT32,
                            GB_SUBASSIGN, Werk)) ;
                    }
                    else
                    { 
                        // Method 00: C(I,J) = empty, using S
                        GBURBLE ("C(I,J)=zombie ") ;
                        // ensure C is sparse; wait(C) may change it
                        if (GB_IS_BITMAP (C) || GB_IS_FULL (C))
                        {
                            GB_OK (GB_convert_any_to_sparse (C, Werk)) ;
                        }
                        GB_OK (GB_subassign_zombie (C,
                            I, I_is_32, ni, nI, Ikind, Icolon,
                            J, J_is_32, nj, nJ, Jkind, Jcolon, Werk)) ;
                    }
                }
                break ;

                default: ;
            }
        }

        //----------------------------------------------------------------------
        // finalize C if blocking mode is enabled, and return result
        //----------------------------------------------------------------------

        ASSERT_MATRIX_OK (C, "Final C for assign, quick mask", GB0) ;
        (*subassign_method) = 0 ;
        GB_FREE_ALL ;
        ASSERT (C == C_in) ;
        (*Chandle) = C ;
        return (GB_block (C, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // disable C_replace if no mask present
    //--------------------------------------------------------------------------

    bool no_mask = (M == NULL && !Mask_comp) ;
    if (no_mask)
    {
        // no_mask:  mask is not present, and not complemented
        if (*C_replace)
        { 
            // The mask is not present and not complemented.  In this case,
            // C_replace is effectively false for subassign.  Disable it, since
            // it can force pending tuples to be assembled.
            GBURBLE ("(no mask: C_replace effectively false) ") ;
            (*C_replace) = false ;
        }
    }

    //--------------------------------------------------------------------------
    // delete pending tuples for C(:,:) = x and C(:,:) = A
    //--------------------------------------------------------------------------

    if (whole_C_matrix)
    { 
        // If the assignment is C<M>(:,:) = ... then convert the assignment
        // into a subassign.
        (*assign_kind) = GB_SUBASSIGN ;
    }

    if (whole_C_matrix && no_mask && accum == NULL)
    { 

        //----------------------------------------------------------------------
        // C(:,:) = x or A:  whole matrix assignment with no mask
        //----------------------------------------------------------------------

        // C_replace is already effectively false (see no_mask condition above)
        ASSERT ((*C_replace) == false) ;

        if (GB_any_aliased (C, A) && !A_transpose && !scalar_expansion)
        { 
            // C = C, with C and A aliased, no transpose, no mask, no accum
            // operator, both I and J are ":", Mask_comp false.  C is not
            // modified at all, and there's no work to do except to check for
            // blocking mode.  The iso property of C is unchanged.
            GBURBLE ("(no-op) ") ;
            (*subassign_method) = 0 ;
            GB_FREE_ALL ;
            ASSERT (C == C_in) ;
            (*Chandle) = C ;
            return (GB_block (C, Werk)) ;
        }

        // free pending tuples early but do not clear C.  If it is
        // already dense then its pattern can be reused.
        GB_Pending_free (&(C->Pending)) ;
    }

    //--------------------------------------------------------------------------
    // transpose A if requested
    //--------------------------------------------------------------------------

    // GrB_Row_assign and GrB_Col_assign pass A as a typecasted vector,
    // which is then quickly transposed to a hypersparse matrix.

    ASSERT_MATRIX_OK (C, "C here in GB_assign_prep", GB0) ;

    if (!scalar_expansion && A_transpose)
    { 
        // AT = A', with no typecasting
        // TODO: if accum is present and it does not depend on the values of
        // A,  construct AT as iso.
        GBURBLE ("(A transpose) ") ;
        GB_CLEAR_MATRIX_HEADER (AT, AT_header_handle) ;
        GB_OK (GB_transpose_cast (AT, A->type, C_is_csc, A, false, Werk)) ;
        GB_MATRIX_WAIT (AT) ;       // A cannot be jumbled
        A = AT ;
    }

    //--------------------------------------------------------------------------
    // transpose the mask if requested
    //--------------------------------------------------------------------------

    // the mask for G*B_Col_*assign and G*B_Row_*assign is a GrB_Vector in CSC
    // form, which is quickly transposed to a hypersparse matrix, if needed.
    // G*B_Vector_*assign always has a CSC mask and CSC C matrix, since both
    // are GrB_Vectors.

    if (M != NULL)
    {
        if (M->is_csc != C_is_csc)
        { 
            // either G*B_Row_*assign and G*B_Col_*assign when matrix C is in
            // CSR format, and or G*B_Matrix_assign when the format of the
            // matrices C and M differ.
            M_transpose = !M_transpose ;
        }
        if (M_transpose)
        { 
            // MT = M' to conform M to the same CSR/CSC format as C,
            // and typecast to boolean.
            GBURBLE ("(M transpose) ") ;
            GB_CLEAR_MATRIX_HEADER (MT, MT_header_handle) ;
            GB_OK (GB_transpose_cast (MT, GrB_BOOL, C_is_csc, M, Mask_struct,
                Werk)) ;
            GB_MATRIX_WAIT (MT) ;       // M cannot be jumbled
            M = MT ;
        }
    }

    //--------------------------------------------------------------------------
    // determine the properties of I and J
    //--------------------------------------------------------------------------

    // If the descriptor says that A must be transposed, it has already been
    // transposed in the caller.  Thus C(I,J), A, and M (if present) all
    // have the same size: length(I)-by-length(J)

    bool I_unsorted, I_has_dupl, I_contig, J_unsorted, J_has_dupl, J_contig ;
    int64_t imin, imax, jmin, jmax ;
    GB_OK (GB_ijproperties (I, I_is_32, ni, nI, C->vlen, &Ikind, Icolon,
                &I_unsorted, &I_has_dupl, &I_contig, &imin, &imax, Werk)) ;
    GB_OK (GB_ijproperties (J, J_is_32, nj, nJ, C->vdim, &Jkind, Jcolon,
                &J_unsorted, &J_has_dupl, &J_contig, &jmin, &jmax, Werk)) ;

    //--------------------------------------------------------------------------
    // sort I and J and remove duplicates, if needed
    //--------------------------------------------------------------------------

    // If I or J are explicit lists, and either of are unsorted or are sorted
    // but have duplicate entries, then both I and J are sorted and their
    // duplicates are removed.  A and M are adjusted accordingly.  Removing
    // duplicates decreases the length of I and J.

    bool I_unsorted_or_has_dupl = (I_unsorted || I_has_dupl) ;
    bool J_unsorted_or_has_dupl = (J_unsorted || J_has_dupl) ;
    bool presort = I_unsorted_or_has_dupl || J_unsorted_or_has_dupl ;

    // This pre-sort of I and J is required for the parallel assignment.
    // Otherwise, multiple threads may attempt to modify the same part of C.
    // This could cause a race condition, if one thread flags a zombie at the
    // same time another thread is using that index in a binary search.  If the
    // 2nd thread finds either zombie/not-zombie, this is fine, but the
    // modification would have to be atomic.  Atomic read/write is slow, so to
    // avoid the use of atomics, the index lists I and J are sorted and all
    // duplicates are removed.

    // A side benefit of this pre-sort is that it ensures that the results of
    // GrB_assign and GxB_subassign are completely defined if I and J have
    // duplicates.  The definition of this pre-sort is given below.

    /*
        function C = subassign (C, I, J, A)
        % submatrix assignment with pre-sort of I and J; and remove duplicates

        % delete duplicates from I, keeping the last one seen
        [I2 I2k] = sort (I) ;
        Idupl = [(I2 (1:end-1) == I2 (2:end)), false] ;
        I2  = I2  (~Idupl) ;
        I2k = I2k (~Idupl) ;
        assert (isequal (I2, unique (I)))

        % delete duplicates from J, keeping the last one seen
        [J2 J2k] = sort (J) ;
        Jdupl = [(J2 (1:end-1) == J2 (2:end)), false] ;
        J2  = J2  (~Jdupl) ;
        J2k = J2k (~Jdupl) ;
        assert (isequal (J2, unique (J)))

        % do the submatrix assignment, with no duplicates in I2 or J2
        C (I2,J2) = A (I2k,J2k) ;
    */

    // With this subassign script, the result returned by GB_subassigner
    // matches the following behavior:

    /*
        C4 = C ;
        C4 (I,J) = A ;
        C3 = subassign (C, I, J, A) ;
        assert (isequal (C4, C3)) ;
    */

    // That is, the pre-sort of I, J, and A has no effect on the final C.

    // The pre-sort itself takes additional work and memory space, but it may
    // actually improve the performance since it makes the data access of C
    // more regular, even in the sequential case.

    if (presort)
    {

        ASSERT (Ikind == GB_LIST || Jkind == GB_LIST) ;
        ASSERT (!whole_C_matrix) ;
        bool I2k_is_32 = false ;
        bool J2k_is_32 = false ;

        if (I_unsorted_or_has_dupl)
        { 
            // I2 = sort I and remove duplicates
            bool I2_is_32 ;
            int64_t ni2 ;
            ASSERT (Ikind == GB_LIST) ;
            GB_OK (GB_ijsort (I, I_is_32, ni, imax,
                &ni2, &I2 , &I2_is_32 , &I2_size,
                      &I2k, &I2k_is_32, &I2k_size, Werk)) ;
            // Recheck the length and properties of the new I2.  This may
            // convert I2 to GB_ALL or GB_RANGE, after I2 has been sorted.
            GB_ijlength (I2, I2_is_32, ni2, C->vlen, &nI, &Ikind, Icolon) ;
            GB_OK (GB_ijproperties (I2, I2_is_32, ni2, nI, C->vlen, &Ikind,
                Icolon, &I_unsorted, &I_has_dupl, &I_contig, &imin, &imax,
                Werk)) ;
            ASSERT (! (I_unsorted || I_has_dupl)) ;
            // replace I with I2
            I = I2 ;
            ni = ni2 ;
            I_is_32 = I2_is_32 ;
        }

        if (J_unsorted_or_has_dupl)
        { 
            // J2 = sort J and remove duplicates
            bool J2_is_32 ;
            int64_t nj2 ;
            ASSERT (Jkind == GB_LIST) ;
            GB_OK (GB_ijsort (J, J_is_32, nj, jmax,
                &nj2, &J2 , &J2_is_32 , &J2_size,
                      &J2k, &J2k_is_32, &J2k_size, Werk)) ;
            // Recheck the length and properties of the new J2.  This may
            // convert J2 to GB_ALL or GB_RANGE, after J2 has been sorted.
            GB_ijlength (J2, J2_is_32, nj2, C->vdim, &nJ, &Jkind, Jcolon) ;
            GB_OK (GB_ijproperties (J2, J2_is_32, nj2, nJ, C->vdim, &Jkind,
                Jcolon, &J_unsorted, &J_has_dupl, &J_contig, &jmin, &jmax,
                Werk)) ;
            ASSERT (! (J_unsorted || J_has_dupl)) ;
            // replace J with J2
            J = J2 ;
            nj = nj2 ;
            J_is_32 = J2_is_32 ;
        }

        // inverse index lists to create the Awork and Mwork submatrices:
        const void *Iinv = I_unsorted_or_has_dupl ? I2k : GrB_ALL ;
        const void *Jinv = J_unsorted_or_has_dupl ? J2k : GrB_ALL ;

        if (!scalar_expansion)
        { 
            // Awork = A (Iinv, Jinv)
            GB_CLEAR_MATRIX_HEADER (Awork, Awork_header_handle) ;
            GB_OK (GB_subref (Awork, false, A->is_csc, A,
                Iinv, I2k_is_32, ni,
                Jinv, J2k_is_32, nj,
                false, Werk)) ;
            // GB_subref can return a jumbled result
            ASSERT (GB_JUMBLED_OK (Awork)) ;
            if (A == AT)
            { 
                GB_Matrix_free (&AT) ;
                AT = NULL ;
            }
            A = Awork ;
        }

        if (M != NULL && (*assign_kind) == GB_SUBASSIGN)
        { 
            // Mwork = M (Iinv, Jinv)
            // if Mask_struct then Mwork is extracted as iso
            GB_CLEAR_MATRIX_HEADER (Mwork, Mwork_header_handle) ;
            GB_OK (GB_subref (Mwork, Mask_struct, M->is_csc, M,
                Iinv, I2k_is_32, ni,
                Jinv, J2k_is_32, nj,
                false, Werk)) ;
            // GB_subref can return a jumbled result
            ASSERT (GB_JUMBLED_OK (Mwork)) ;
            if (M == MT)
            {
                GB_Matrix_free (&MT) ;
                MT = NULL ;
            }
            M = Mwork ;
        }

        GB_FREE_MEMORY (&I2k, I2k_size) ;
        GB_FREE_MEMORY (&J2k, J2k_size) ;
    }

    // I and J are now sorted, with no duplicate entries.  They are either
    // GB_ALL, GB_RANGE, or GB_STRIDE, which are intrinsically sorted with no
    // duplicates, or they are explicit GB_LISTs with sorted entries and no
    // duplicates.

    ASSERT (!I_unsorted) ; ASSERT (!I_has_dupl) ;
    ASSERT (!J_unsorted) ; ASSERT (!J_has_dupl) ;

    //--------------------------------------------------------------------------
    // check for early C_replace
    //--------------------------------------------------------------------------

    // C(:,:)<any mask, replace> = A or x

    // C_replace_may_be_done_early is true if the C_replace action can take
    // place now.  If true, the final C does not depend on the contents of
    // C on input.  If bitmap assigment might be done, delay the clearing of
    // C since it would be faster to set its bitmap to all zero later on,
    // instead of freeing it and reallocating it.

    bool C_replace_may_be_done_early = (whole_C_matrix && (*C_replace)
        && accum == NULL && !use_bitmap_assign) ;

    // If the entire C(:,:) is being assigned to, and if no accum operator is
    // present, then the matrix can be cleared of all entries now, and then
    // C_replace can be set false.  Clearing C now speeds up the assignment
    // since the wait on C can be skipped, below.  It also simplifies the
    // kernels.  If S is constructed, it is just an empty matrix.

    // By clearing C now and setting C_replace to false, the following methods
    // are used: 09 becomes 05, 10 becomes 06n or 06s, 17 becomes 13, and 18
    // becomes 14.  The S matrix for methods 06s, 13, and 14 is still created,
    // but it is very fast to construct and traverse since C is empty.  Method
    // 00 can be skipped since C is already empty (see "quick" case below).

        // prior time             new  time           action
        // ----- ----             ---  ----           ------

        // 00:  O(S)              nothing, O(1)       C already cleared

        // 09:  O(M+S)            05:  O(M)           C<M> = x, no S

        // 10:  O((A+S)*log(m))   06n: O(M*(log(a))   C<M> = A, no S
        //                        06s: O(A*(log(m))   C<M> = A, with S

        // 17:  O(m*n)            13:  O(m*n)         C<!M> = x, with S

        // 18:  O(A*log(m))       14:  O(A*log(m))    C<!M> = A, with S

        //  =====================       ==============
        //  M   cmp rpl acc A   S       method: action
        //  =====================       ==============

        //  M   -   -   -   -   -       05:  C(I,J)<M> = x, no S
        //  M   -   -   -   A   -       06n: C(I,J)<M> = A, no S
        //  M   -   -   -   A   S       06s: C(I,J)<M> = A, with S

        //  M   -   r   -   -   S       09:  C(I,J)<M,repl> = x, with S
        //  M   -   r   -   A   S       10:  C(I,J)<M,repl> = A, with S

        //  M   c   -   -   -   S       13:  C(I,J)<!M> = x, with S
        //  M   c   -   -   A   S       14:  C(I,J)<!M> = A, with S

        //  M   c   r   -   -   S       17:  C(I,J)<!M,repl> = x, with S
        //  M   c   r   -   A   S       18:  C(I,J)<!M,repl> = A, with S

        // Methods 09, 10, 17, and 18 are now used only if C(I,J) is a
        // submatrix of C, and not for the whole_C_matrix case.

    //--------------------------------------------------------------------------
    // make a copy Z = C if C is aliased to A or M
    //--------------------------------------------------------------------------

    // TODO: bitmap assign can handle C==M and C==A aliasing in some cases

    // If C is aliased to A and/or M, a copy of C typically must be made.
    bool C_any_alias_with_A = GB_any_aliased (C, A) ;
    bool C_aliased = C_any_alias_with_A || GB_any_aliased (C, M) ;

    // However, if C == M is aliased, M is structural and not complemented, I
    // and J are both ":", and scalar assignment is being done, then the alias
    // of C and M can be exploited.  The assignment is C<C,s>=scalar.
    // C<C,s>+=scalar might be exploited in the future.
    // C_replace is effectively false.
    bool C_exploit_alias_with_M =
        ((C == M)               // C is exactly aliased with M
        && Mask_struct          // mask is structural
        && !Mask_comp           // and not complemented
        && whole_C_matrix       // C(:,:) is being assigned to
        &&
        (
            // C<C,s> = scalar assignment (method 05f)
            // no accum (accum can be handled in the future)
            ((accum == NULL) && scalar_expansion)
        ||
            // C<C,s> += A, matrix update (method 27)
            // C and A cannot be aliased in any way and cannot be bitmap
            ((accum != NULL) && !scalar_expansion && !C_any_alias_with_A
            && !GB_IS_BITMAP (A) && !C_is_bitmap && !use_bitmap_assign)
        )) ;

    // GB_assign cannot tolerate any alias with the input mask,
    // if the C_replace phase will be performed.
    if ((*C_replace) && ((*assign_kind) != GB_SUBASSIGN))
    { 
        // the C_replace phase requires C and M_in not to be aliased
        C_aliased = C_aliased || GB_any_aliased (C, M_in) ;
    }

    if (C_exploit_alias_with_M)
    {
        // C<C,s>=scalar (method 05f) or C<C,s>+=A (method 27):
        // C_replace can be ignored.
        ASSERT (C_aliased) ;            // C is aliased with M, but this is OK
        ASSERT (!C_any_alias_with_A) ;  // no A matrix, or C not aliased with A
        if (*C_replace)
        { 
            GBURBLE ("(C_replace ignored) ") ;
            (*C_replace) = false ;
        }
    }
    else if (C_aliased)
    {
        // C is aliased with M or A: make a copy of C to assign into
        GB_CLEAR_MATRIX_HEADER (Cwork, Cwork_header_handle) ;
        if (C_replace_may_be_done_early)
        { 
            // Instead of duplicating C, create a new empty matrix Cwork.
            int sparsity = (C->h != NULL) ? GxB_HYPERSPARSE : GxB_SPARSE ;
            GB_OK (GB_new (&Cwork, // sparse or hyper, existing header
                ctype, C->vlen, C->vdim, GB_ph_calloc, C_is_csc,
                sparsity, C->hyper_switch, 1,
                C->p_is_32, C->j_is_32, C->i_is_32)) ;
            GBURBLE ("(C alias cleared; C_replace early) ") ;
            (*C_replace) = false ;
        }
        else
        { 
            // finish any computations in C, but leave it jumbled
            // FUTURE:: keep zombies in C, and avoid making a copy of C
            GBURBLE ("(%sC alias: duplicate) ", C->iso ? "iso " : "") ;

#if 0
if (C==M && !GB_any_aliased (C, A) && Mask_struct && !Mask_comp && whole_C_matrix)
{
    // C<C,struct> = A or scalar, with or without accum
    printf ("\nhack1 accum: %d, scalar: %d\n", accum != NULL,
        scalar_expansion) ;
    // abort ( ) ;
}
if (C==A && !GB_any_aliased (C, M) && Mask_struct && !Mask_comp && whole_C_matrix)
{
    // C<A,struct> = A
    printf ("\nhack2 accum: %d, scalar: %d\n", accum != NULL, scalar_expansion) ;
    // abort ( ) ;
}
if (C==A && !GB_any_aliased (C, M) && Mask_struct && Mask_comp && whole_C_matrix)
{
    // C<not A,struct> = A
    // only in LAGr_MaximumMatching.c:848
    printf ("\nhack3 accum: %d, scalar: %d\n", accum != NULL, scalar_expansion) ;
    // abort ( ) ;
}
#endif

            GB_MATRIX_WAIT_IF_PENDING_OR_ZOMBIES (C) ;
            ASSERT (!GB_ZOMBIES (C)) ;
            ASSERT (GB_JUMBLED_OK (C)) ;
            ASSERT (!GB_PENDING (C)) ;
            // Cwork = duplicate of C, which must be freed when done
            GB_OK (GB_dup_worker (&Cwork, C->iso, C, true, NULL)) ;
        }
        // Cwork must be transplanted back into C when done
        C = Cwork ;
        ASSERT (C->header_size == 0 || GBNSTATIC) ;
    }
    else
    {
        // C is not aliased, but check if it can be cleared early
        if (C_replace_may_be_done_early)
        { 
            // Clear C early.
            GB_OK (GB_clear (C, Werk)) ;
            GBURBLE ("(C(:,:)<any mask>: C_replace early) ") ;
            (*C_replace) = false ;
        }
        // the assignment operates on C in-place
    }

    //--------------------------------------------------------------------------
    // disable C_replace if C is empty
    //--------------------------------------------------------------------------

    bool C_is_empty = (GB_nnz (C) == 0 && !GB_PENDING (C) && !GB_ZOMBIES (C)) ;
    if (C_is_empty)
    { 
        // C is completely empty.  C_replace is irrelevant so set it to false.
        (*C_replace) = false ;
    }

    //--------------------------------------------------------------------------
    // determine the initial subassign method to use (prior to wait)
    //--------------------------------------------------------------------------

    // This decision can change if wait(C) is done.

    bool C_iso_out = false ;
    size_t csize = ctype->size ;
    GB_void cout [GB_VLA(csize)] ;
    (*subassign_method) = GB_subassigner_method (&C_iso_out, cout, C,
        (*C_replace), M, Mask_comp, Mask_struct, accum, A,
        Ikind, Jkind, nJ, Jcolon, scalar_expansion, scalar, scalar_type) ;

    //--------------------------------------------------------------------------
    // check compatibilty of prior pending tuples
    //--------------------------------------------------------------------------

    // The action: ( delete ) can only delete a live entry in the pattern.  It
    // cannot delete a pending tuple; pending tuples cannot become zombies.
    // Thus, if this assignment has the potential for creating zombies, all
    // prior pending tuples must be assembled now.  They thus become live
    // entries in the pattern of C, so that this GB_subassigner can
    // (potentially) turn them into zombies via action: ( delete ).

    // If accum is NULL, the operation is C(I,J) = A, or C(I,J)<M> = A.  If A
    // has any implicit zeros at all, or if M is present, then the
    // action: ( delete ) is possible.  This action is taken when an entry is
    // found in C but not A.  It is thus not possible to check A in advance if
    // an entry in C must be deleted.  If an entry does not appear in C but
    // appears as a pending tuple, deleting it would require a scan of all the
    // pending tuples in C.  This is costly, and simply assembling all pending
    // tuples first is faster.

    // The action: ( insert ) adds additional pending tuples.  All pending
    // tuples will be assembled sometime later on, using a single pending
    // operator, and thus the current accum operator must match the prior
    // pending operator.  If the operators do not match, then all prior pending
    // tuples must be assembled now, so that this GB_subassigner can
    // (potentially) insert new pending tuples whose pending operator is accum.

    // These tests are conservative because it is possible that this
    // GxB_subassign will not need to use action: ( insert ).

    // In the discussion below, let SECOND_Ctype denote the SECOND operator
    // z=f(x,y) whose ztype, xtype, and ytype matches the type of C.

    bool wait = false ;

    if (C->Pending == NULL)
    { 

        //----------------------------------------------------------------------
        // no pending tuples currently exist
        //----------------------------------------------------------------------

        // If any new pending tuples are added, their pending operator is
        // accum, or the implicit SECOND_Ctype operator if accum is NULL.
        // The type of any pending tuples will become ctype.
        // Prior zombies have no effect on this decision.

        wait = false ;

    }
    else
    {

        //----------------------------------------------------------------------
        // prior pending tuples exist: check if action: ( delete ) can occur
        //----------------------------------------------------------------------

        // action: ( delete ) can only operate on entries in the pattern by
        // turning them into zombies.  It cannot delete prior pending tuples.
        // Thus all prior pending tuples must be assembled first if
        // action: ( delete ) can occur.

        if (*C_replace)
        { 
            // C_replace must use the action: ( delete )
            wait = true ;
        }
        else if (accum == NULL)
        {
            // This GxB_subassign can potentially use action: ( delete ), and
            // thus prior pending tuples must be assembled first.  However, if
            // A is full, then C(I,J)=A cannot delete any entries from C.
            if (scalar_expansion || GB_IS_FULL (A))
            { 
                // A is a scalar or dense matrix, so entries cannot be deleted
                wait = false ;
            }
            else
            { 
                // A is sparse.  action: ( delete ) might occur.
                wait = true ;
            }
        }

        //----------------------------------------------------------------------
        // check if pending operator is compatible
        //----------------------------------------------------------------------

        if (!wait)
        { 
            // ( delete ) will not occur, but new pending tuples may be added
            // via the action: ( insert ).  Check if the accum operator is the
            // same as the prior pending operator and ensure the types are
            // the same.

            ASSERT (C->Pending != NULL) ;
            ASSERT (C->Pending->type != NULL) ;

            wait =
                // entries in A are copied directly into the list of pending
                // tuples for C, with no typecasting.  The type of the prior
                // pending tuples must match the type of A.  If the types do
                // not match, prior updates must be assembled first.
                (atype != C->Pending->type)
                // also wait if the pending operator has changed.
                || !((accum == C->Pending->op)
                    || (GB_op_is_second (accum, ctype) &&
                        GB_op_is_second (C->Pending->op, ctype)))
                // also wait if the iso property of C changes.
                || (C->iso != C_iso_out) ;
        }
    }

    //--------------------------------------------------------------------------
    // wait on the matrix, if required
    //--------------------------------------------------------------------------

    if (wait)
    { 
        // Prior computations are not compatible with this assignment, so all
        // prior work must be finished.  This potentially costly.
        // delete any lingering zombies and assemble any pending tuples
        ASSERT_MATRIX_OK (C, "C before wait", GB0) ;
        GB_MATRIX_WAIT (C) ;

        // GB_wait may have deleted all the zombies in C, so check again if C
        // is empty.  If so, C_replace is irrelevant so set it false
        C_is_empty = (GB_nnz (C) == 0 && !GB_PENDING (C) && !GB_ZOMBIES (C)) ;
        if (C_is_empty) (*C_replace) = false ;

        // C has changed so recompute the subassigner method
        (*subassign_method) = GB_subassigner_method (&C_iso_out, cout, C,
            (*C_replace), M, Mask_comp, Mask_struct, accum, A, Ikind, Jkind,
            nJ, Jcolon, scalar_expansion, scalar, scalar_type) ;
    }

    ASSERT_MATRIX_OK (C, "C before subassign", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for assign", GB0) ;

    if (C_iso_out)
    { 
        GBURBLE ("(C iso assign) ") ;
    }

    //--------------------------------------------------------------------------
    // keep track of the current accum operator
    //--------------------------------------------------------------------------

    // If accum is NULL and pending tuples are added, they will be assembled
    // sometime later (not here) using the implied SECOND_Ctype operator.  This
    // GB_subassigner operation corresponds to C(I,J)=A or C(I,J)<M>=A.
    // Subsequent calls to GrB_setElement, and subsequent calls to GrB_assign
    // or GxB_subassign with an explict SECOND_Ctype operator, may create
    // additional pending tuples and add them to the list without requiring
    // that they be assembled first.

    // If accum is non-NULL, then all prior pending tuples have the same
    // pending operator as this accum.  If that prior operator was the implicit
    // SECOND_Ctype and those pending tuples still exist, then this accum
    // operator is the explicit SECOND_ctype operator.  The implicit
    // SECOND_Ctype operator is replaced with the current accum, which is the
    // explicit SECOND_Ctype operator.

    // If C is iso, the pending op is effectively the implicit SECOND_Ctype op.

    if (C->Pending != NULL)
    { 
        C->Pending->op = (C_iso_out) ? NULL : accum ;
    }

    //--------------------------------------------------------------------------
    // convert C to its final iso property
    //--------------------------------------------------------------------------

    if (C->iso && !C_iso_out)
    { 
        // C is iso on input, but non-iso on output; expand the iso value
        // into all of C->x
        // set C->iso = false
        GB_OK (GB_convert_any_to_non_iso (C, true)) ;
    }
    else if (!C->iso && C_iso_out)
    { 
        // C is non-iso on input, but iso on output
        // copy the cout scalar into C->x
        // set C->iso = true
        // C<C,struct>=scalar (method 05f) uses this, if C is not iso on input;
        // it always becomes iso on output.
        GB_OK (GB_convert_any_to_iso (C, cout)) ;
    }
    else if (C->iso && C_iso_out)
    { 
        // The iso status of C is unchanged; set its new iso value.
        // C<C,struct>=scalar (method 05f) uses this, if C is iso on input; it
        // always stays iso on output.
        memcpy (C->x, cout, csize) ;
    }
    ASSERT_MATRIX_OK (C, "C output from GB_assign_prep", GB0) ;

    //--------------------------------------------------------------------------
    // return results
    //--------------------------------------------------------------------------

    (*Chandle) = C ;            // C is C_in or Cwork
    (*Mhandle) = M ;            // M is M_in or Mwork
    (*Ahandle) = A ;            // A is A_in or Awork

    (*Cwork_handle) = Cwork ;
    (*Mwork_handle) = (MT != NULL) ? MT : Mwork ;
    (*Awork_handle) = (AT != NULL) ? AT : Awork ;

    (*scalar_type_handle) = scalar_type ;   // may be NULL

    // modified versions of the Rows/Cols lists, and their analysis:
    (*I_handle      ) = (void *) I ;    // either Rows, Cols, or I2
    (*I_is_32_handle) = I_is_32 ;
    (*I2_handle     ) = I2 ;        // temporary sorted copy of Rows/Cols list
    (*I2_size_handle) = I2_size ;
    (*ni_handle     ) = ni ;
    (*nI_handle     ) = nI ;
    (*Ikind_handle  ) = Ikind ;

    (*J_handle      ) = (void *) J ;    // either Rows, Cols, or J2
    (*J_is_32_handle) = J_is_32 ;
    (*J2_handle     ) = J2 ;        // temporary sorted copy of Rows/Cols list
    (*J2_size_handle) = J2_size ;
    (*nj_handle     ) = nj ;
    (*nJ_handle     ) = nJ ;
    (*Jkind_handle  ) = Jkind ;

    return (GrB_SUCCESS) ;
}

