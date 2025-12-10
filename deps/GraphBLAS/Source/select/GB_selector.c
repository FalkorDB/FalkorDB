//------------------------------------------------------------------------------
// GB_selector:  select entries from a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_selector does the work for GB_select.  It also deletes zombies for
// GB_wait using the GxB_NONZOMBIE operator, deletes entries outside a smaller
// matrix for GxB_*resize using GrB_ROWLE, and extracts the diagonal entries
// for GB_Vector_diag.

// For GB_resize (using GrB_ROWLE) and GB_wait (using GxB_NONZOMBIE), C may be
// NULL.  In this case, A is always sparse or hypersparse.  If C is NULL on
// input, A is modified in-place.  Otherwise, C is an uninitialized static
// header.

// TODO: GB_selector does not exploit the mask.

#include "select/GB_select.h"

#define GB_FREE_ALL                         \
    GB_FREE_MEMORY (&ythunk, ythunk_size) ;   \
    GB_FREE_MEMORY (&athunk, athunk_size) ;

GrB_Info GB_selector
(
    GrB_Matrix C,               // output matrix, NULL or existing header
    const GrB_IndexUnaryOp op,
    const bool flipij,          // if true, flip i and j for user operator
    GrB_Matrix A,               // input matrix
    const GrB_Scalar Thunk,
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_INDEXUNARYOP_OK (op, "idxunop for GB_selector", GB0) ;
    ASSERT_SCALAR_OK (Thunk, "Thunk for GB_selector", GB0) ;
    ASSERT_MATRIX_OK (A, "A input for GB_selector", GB0_Z) ;
    // positional op (tril, triu, diag, offdiag, resize, rowindex, ...):
    // can't be jumbled.  nonzombie, entry-valued op, user op: jumbled OK
    GB_Opcode opcode = op->opcode ;
    ASSERT (GB_IMPLIES (GB_IS_INDEXUNARYOP_CODE_POSITIONAL (opcode),
        !GB_JUMBLED (A))) ;

    ASSERT (C != NULL && (C->header_size == 0 || GBNSTATIC)) ;
    const bool A_iso = A->iso ;
    void *ythunk = NULL ; size_t ythunk_size = 0 ;
    void *athunk = NULL ; size_t athunk_size = 0 ;

    //--------------------------------------------------------------------------
    // get Thunk
    //--------------------------------------------------------------------------

    // get the type of the thunk input of the operator
    ASSERT (GB_nnz ((GrB_Matrix) Thunk) > 0) ;
    const GB_Type_code tcode = Thunk->type->code ;

    // allocate the ythunk and athunk scalars.  Use calloc instead of putting
    // them on the CPU stack, so the CUDA kernels can access them.
    const size_t ysize = op->ytype->size ;
    const size_t asize = A->type->size ;
    ythunk = GB_CALLOC_MEMORY (1, ysize, &ythunk_size) ;
    athunk = GB_CALLOC_MEMORY (1, asize, &athunk_size) ;
    if (ythunk == NULL || athunk == NULL)
    {
        // out of memory
        GB_FREE_ALL ;
        return (GrB_OUT_OF_MEMORY) ;
    }

    // ythunk = (op->ytype) Thunk
    GB_cast_scalar (ythunk, op->ytype->code, Thunk->x, tcode, ysize) ;

    // ithunk = (int64) Thunk, if compatible
    int64_t ithunk = 0 ;
    if (GB_Type_compatible (GrB_INT64, Thunk->type))
    {
        GB_cast_scalar (&ithunk, GB_INT64_code, Thunk->x, tcode,
            sizeof (int64_t)) ;
    }

    // athunk = (A->type) Thunk, for VALUEEQ operator only
    if (opcode == GB_VALUEEQ_idxunop_code)
    {
        ASSERT (GB_Type_compatible (A->type, Thunk->type)) ;
        GB_cast_scalar (athunk, A->type->code, Thunk->x, tcode, asize) ;
    }

    //--------------------------------------------------------------------------
    // determine if C is iso for a non-iso A
    //--------------------------------------------------------------------------

    bool C_iso = A_iso ||                       // C iso value is Ax [0]
        (opcode == GB_VALUEEQ_idxunop_code) ;   // C iso value is thunk
    if (C_iso)
    { 
        GB_BURBLE_MATRIX (A, "(iso select) ") ;
    }

    //--------------------------------------------------------------------------
    // handle iso case for built-in ops that depend only on the value
    //--------------------------------------------------------------------------

    if (A_iso && opcode >= GB_VALUENE_idxunop_code
              && opcode <= GB_VALUELE_idxunop_code)
    { 
        // C is either entirely empty, or a completely shallow copy of A.
        // This method takes O(1) time and space.
        GB_OK (GB_select_value_iso (C, op, A, ithunk, athunk, ythunk, Werk)) ;
        GB_FREE_ALL ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // bitmap/as-if-full case
    //--------------------------------------------------------------------------

    bool use_select_bitmap ;
    if (opcode == GB_NONZOMBIE_idxunop_code)
    { 
        // GB_select_bitmap does not support the nonzombie opcode.  For the
        // NONZOMBIE operator, A will never be full or bitmap.
        use_select_bitmap = false ;
    }
    else if (opcode == GB_DIAG_idxunop_code)
    { 
        // GB_select_bitmap supports the DIAG operator, but it is currently
        // not efficient (GB_select_bitmap should return a sparse diagonal
        // matrix, not bitmap).  So use the sparse case if A is not bitmap,
        // since the sparse case below does not support the bitmap case.  For
        // this case, GB_select_sparse is used when A is a sparse, hypersparse,
        // or full matrix.  The full case is not handled in the CUDA kernel
        // below, however.
        use_select_bitmap = GB_IS_BITMAP (A) ;
    }
    else
    { 
        // For bitmap and full matrices, all other opcodes use GB_select_bitmap
        use_select_bitmap = GB_IS_BITMAP (A) || GB_IS_FULL (A) ;
    }

    if (use_select_bitmap)
    { 
        // A is bitmap/full.  C is always computed as bitmap.
        GB_BURBLE_MATRIX (A, "(bitmap select) ") ;
        GB_OK (GB_select_bitmap (C, C_iso, op, flipij, A, ithunk, athunk,
            ythunk, Werk)) ;
        GB_FREE_ALL ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // column selector
    //--------------------------------------------------------------------------

    if (opcode == GB_COLINDEX_idxunop_code ||
        opcode == GB_COLLE_idxunop_code ||
        opcode == GB_COLGT_idxunop_code)
    { 
        // A is sparse or hypersparse, never bitmap or full.
        // COLINDEX: C = A(:,j)
        // COLLE:    C = A(:,0:j)
        // COLGT:    C = A(:,j+1:n)
        // where j = ithunk.
        GB_OK (GB_select_column (C, op, A, ithunk, Werk)) ;
        GB_FREE_ALL ;
        return (GrB_SUCCESS) ;
    }

    //--------------------------------------------------------------------------
    // general case: usually sparse/hypersparse, with one exception
    //--------------------------------------------------------------------------

    // C is computed as sparse/hypersparse.  A is sparse/hypersparse, except
    // for a single case: for the DIAG operator, A may be full.  See
    // use_select_bitmap above.

    info = GrB_NO_VALUE ;

    #if defined ( GRAPHBLAS_HAS_CUDA )
    if ((GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A))
        && GB_cuda_select_branch (A, op))
    {
        // It is possible for non-sparse matrices to use the sparse kernel; see
        // the use_select_bitmap test above (the DIAG operator). The CUDA
        // select_sparse kernel will not work in this case, so make this go to
        // the CPU.
        // FIXME CUDA: put the test of sparse(A) or hypersparse(A) in
        // GB_cuda_select_branch.
        info = GB_cuda_select_sparse (C, C_iso, op, flipij, A, athunk, ythunk,
            Werk) ;
    }
    #endif

    if (info == GrB_NO_VALUE)
    {
        info = GB_select_sparse (C, C_iso, op, flipij, A, ithunk, athunk,
            ythunk, Werk) ;
    }

    GB_OK (info) ;  // check for out-of-memory or other failures

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "C output of GB_selector", GB0) ;
    return (GrB_SUCCESS) ;
}

