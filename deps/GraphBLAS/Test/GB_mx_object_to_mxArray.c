//------------------------------------------------------------------------------
// GB_mx_object_to_mxArray
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Convert a GraphBLAS sparse or full matrix to a built-in struct C containing
// C.matrix and a string C.class.  The GraphBLAS matrix is destroyed.

// This could be done using only user-callable GraphBLAS functions, by
// extracting the tuples and converting them into a built-in sparse matrix.  But
// that would be much slower and take more memory.  Instead, most of the work
// can be done by pointers, and directly accessing the internal contents of C.
// If C has type GB_BOOL_code or GB_FP64_code, then C can be converted to a
// built-in matrix in constant time with essentially no extra memory allocated.
// This is faster, but it means that this Test interface will only work with
// this specific implementation of GraphBLAS.

// Note that the GraphBLAS matrix may contain explicit zeros.

// If the GraphBLAS matrix is iso, it is converted to non-iso, but if it is
// returned as a struct, the C.iso is set true.  Then when the struct is read
// back into GraphBLAS, the flag can be used to restore the iso property of the
// GraphBLAS matrix .

#include "GB_mex.h"

#define GB_AS_IF_FREE(p)                \
{                                       \
    GB_Global_memtable_remove (p) ;     \
    (p) = NULL ;                        \
}

static const char *MatrixFields [ ] = { "matrix", "class", "iso", "values" } ;

mxArray *GB_mx_object_to_mxArray   // returns the built-in mxArray
(
    GrB_Matrix *handle,             // handle of GraphBLAS matrix to convert
    const char *name,
    const bool create_struct        // if true, then return a struct
)
{

    // get the inputs
    mxArray *A, *Astruct, *X = NULL ;
    GrB_Matrix C = *handle ;
    GrB_Type ctype = C->type ;

    // may have pending tuples
    ASSERT_MATRIX_OK (C, name, GB0) ;

    // C must not be shallow
    ASSERT (!C->p_shallow) ;
    ASSERT (!C->h_shallow) ;
    ASSERT (!C->b_shallow) ;
    ASSERT (!C->i_shallow) ;
    ASSERT (!C->x_shallow) ;

    // make sure there are no pending computations
    if (GB_IS_FULL (C) || GB_IS_BITMAP (C))
    {
        ASSERT (!GB_JUMBLED (C)) ;
        ASSERT (!GB_ZOMBIES (C)) ;
        ASSERT (!GB_PENDING (C)) ;
    }
    else
    {
        // this may convert C to full
        GrB_Matrix_wait (C, GrB_MATERIALIZE) ;
        C = (*handle) ;
    }

    // must be done after GrB_Matrix_wait:
    int64_t cnz = GB_nnz (C) ;

    ASSERT_MATRIX_OK (C, "TO mxArray after assembling pending tuples", GB0) ;

    // ensure C is sparse or full, not hypersparse or bitmap
    GxB_Matrix_Option_set_(C, GxB_SPARSITY_CONTROL, GxB_FULL + GxB_SPARSE) ;
    ASSERT_MATRIX_OK (C, "TO mxArray, sparse or full", GB0) ;
    ASSERT (!GB_IS_HYPERSPARSE (C)) ;
    ASSERT (!GB_IS_BITMAP (C)) ;

    // get the current sparsity
    int sparsity ;
    GxB_Matrix_Option_get_(C, GxB_SPARSITY_STATUS, &sparsity) ;
    ASSERT (sparsity == GxB_FULL || sparsity == GxB_SPARSE) ;

    // make sure it's CSC
    if (!C->is_csc)
    {
        GxB_Matrix_Option_set_(C, GxB_FORMAT, GxB_BY_COL) ;
    }

    // setting to CSC may have transposed the matrix
    ASSERT (GB_JUMBLED_OK (C)) ;
    GrB_Matrix_wait (C, GrB_MATERIALIZE) ;
    ASSERT (!GB_JUMBLED (C)) ;
    cnz = GB_nnz (C) ;

    ASSERT_MATRIX_OK (C, "TO mxArray, non-hyper CSC", GB0) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_IS_HYPERSPARSE (C)) ;
    ASSERT (!GB_IS_BITMAP (C)) ;
    ASSERT (GB_IS_SPARSE (C) || GB_IS_FULL (C)) ;
    ASSERT (C->is_csc) ;

    // convert C to non-iso
    bool C_iso = C->iso ;
    if (C_iso)
    {
        GB_convert_any_to_non_iso (C, true) ;
        ASSERT_MATRIX_OK (C, "TO mxArray, non-iso non-hyper CSC", GB0) ;
    }

    // empty built-in matrices don't want NULL pointers
    if (C->x == NULL)
    {
        ASSERT (cnz == 0) ;
        C->x = (GB_void *) GB_malloc_memory (2 * sizeof (double),
            sizeof (GB_void), &(C->x_size)) ;
        memset (C->x, 0, 2 * sizeof (double)) ;
        C->x_shallow = false ;
    }

    bool C_is_full = (sparsity == GxB_FULL) ;
    if (!C_is_full)
    {
        // empty built-in sparse matrices don't want NULL pointers
        if (C->i == NULL)
        {
            ASSERT (cnz == 0) ;
            C->i = (int64_t *) GB_malloc_memory (1, sizeof (int64_t),
                &(C->i_size)) ;
            C->i [0] = 0 ;
            C->i_shallow = false ;
        }
        if (C->p == NULL)
        {
            ASSERT (cnz == 0) ;
            C->p = (int64_t *) GB_malloc_memory (C->vdim + 1, 
                sizeof (int64_t), &(C->p_size)) ;
            memset (C->p, 0, (C->vdim + 1) * sizeof (int64_t)) ;
            C->p_shallow = false ;
        }
    }

    //--------------------------------------------------------------------------
    // create the built-in matrix A and link in the numerical values of C
    //--------------------------------------------------------------------------

    if (C_is_full)
    {
        // C is full.
        // allocate an empty dense matrix of the right type, then set content

        void *Cx = (void *) C->x ;

        if (ctype == GrB_BOOL)
        { 
            A = mxCreateLogicalMatrix (0, 0) ;
            mxSetData (A, Cx) ;
        }
        else if (ctype == GrB_FP32)
        { 
            A = mxCreateNumericMatrix (0, 0, mxSINGLE_CLASS, mxREAL) ;
            mxSetSingles (A, Cx) ;
        }
        else if (ctype == GrB_FP64)
        { 
            A = mxCreateNumericMatrix (0, 0, mxDOUBLE_CLASS, mxREAL) ;
            mxSetDoubles (A, Cx) ;
        }
        else if (ctype == GrB_INT8)
        { 
            A = mxCreateNumericMatrix (0, 0, mxINT8_CLASS, mxREAL) ;
            mxSetInt8s (A, Cx) ;
        }
        else if (ctype == GrB_INT16)
        { 
            A = mxCreateNumericMatrix (0, 0, mxINT16_CLASS, mxREAL) ;
            mxSetInt16s (A, Cx) ;
        }
        else if (ctype == GrB_INT32)
        { 
            A = mxCreateNumericMatrix (0, 0, mxINT32_CLASS, mxREAL) ;
            mxSetInt32s (A, Cx) ;
        }
        else if (ctype == GrB_INT64)
        { 
            A = mxCreateNumericMatrix (0, 0, mxINT64_CLASS, mxREAL) ;
            mxSetInt64s (A, Cx) ;
        }
        else if (ctype == GrB_UINT8)
        { 
            A = mxCreateNumericMatrix (0, 0, mxUINT8_CLASS, mxREAL) ;
            mxSetUint8s (A, Cx) ;
        }
        else if (ctype == GrB_UINT16)
        { 
            A = mxCreateNumericMatrix (0, 0, mxUINT16_CLASS, mxREAL) ;
            mxSetUint16s (A, Cx) ;
        }
        else if (ctype == GrB_UINT32)
        { 
            A = mxCreateNumericMatrix (0, 0, mxUINT32_CLASS, mxREAL) ;
            mxSetUint32s (A, Cx) ;
        }
        else if (ctype == GrB_UINT64)
        { 
            A = mxCreateNumericMatrix (0, 0, mxUINT64_CLASS, mxREAL) ;
            mxSetUint64s (A, Cx) ;
        }
        else if (ctype == GxB_FC32)
        {
            A = mxCreateNumericMatrix (0, 0, mxSINGLE_CLASS, mxCOMPLEX) ;
            mxSetComplexSingles (A, Cx) ;
        }
        else if (ctype == Complex || ctype == GxB_FC64)
        {
            A = mxCreateNumericMatrix (0, 0, mxDOUBLE_CLASS, mxCOMPLEX) ;
            mxSetComplexDoubles (A, Cx) ;
        }
        else
        {
            mexErrMsgTxt ("... unsupported type") ;
        }

        mexMakeMemoryPersistent (C->x) ;
        C->x_shallow = false ;
        GB_AS_IF_FREE (C->x) ;   // unlink C->x from C; now in built-in C

    }
    else if (C->type == GrB_BOOL)
    {
        // C is boolean, which is the same as a built-in logical sparse matrix
        A = mxCreateSparseLogicalMatrix (0, 0, 0) ;
        mexMakeMemoryPersistent (C->x) ;
        mxSetData (A, (bool *) C->x) ;
        C->x_shallow = false ;

        // C->x is treated as if it was freed
        GB_AS_IF_FREE (C->x) ;   // unlink C->x from C; now in built-in C

    }
    else if (C->type == GrB_FP64)
    {
        // C is double, which is the same as a built-in double sparse matrix
        A = mxCreateSparse (0, 0, 0, mxREAL) ;
        mexMakeMemoryPersistent (C->x) ;
        mxSetData (A, C->x) ;
        C->x_shallow = false ;

        // C->x is treated as if it was freed
        GB_AS_IF_FREE (C->x) ;   // unlink C->x from C; in built-in C

    }
    else if (C->type == Complex || C->type == GxB_FC64)
    {

        // user-defined Complex type, or GraphBLAS GxB_FC64
        A = mxCreateSparse (C->vlen, C->vdim, cnz, mxCOMPLEX) ;
        memcpy (mxGetComplexDoubles (A), C->x, cnz * sizeof (GxB_FC64_t)) ;

    }
    else if (C->type == GxB_FC32)
    {

        // C is single complex, typecast to sparse double complex
        A = mxCreateSparse (C->vlen, C->vdim, cnz, mxCOMPLEX) ;
        GB_void *Ax = (GB_void *) mxGetComplexDoubles (A) ;
        if (Ax == NULL && cnz > 0) mexErrMsgTxt ("Ax is NULL!\n") ;
        GB_cast_array (Ax, GB_FC64_code, C, 1) ;

    }
    else
    {

        // otherwise C is cast into a built-in double sparse matrix
        A = mxCreateSparse (0, 0, 0, mxREAL) ;
        size_t Sx_size ;
        double *Sx = (double *) GB_malloc_memory (cnz+1, sizeof (double),
            &Sx_size) ;
        if (Sx == NULL && cnz > 0) mexErrMsgTxt ("Sx is NULL!\n") ;
        GB_cast_array ((GB_void *) Sx, GB_FP64_code, C, 1) ;
        mexMakeMemoryPersistent (Sx) ;
        mxSetPr (A, Sx) ;

        // Sx was just malloc'd.  Treat it as if GraphBLAS has freed it
        GB_AS_IF_FREE (Sx) ;

        if (create_struct)
        {
            // If C is int64 or uint64, then typecasting can lose information,
            // so keep an uncasted copy of C->x as well.
            X = GB_mx_create_full (0, 0, C->type) ;
            mxSetM (X, cnz) ;
            mxSetN (X, 1) ;
            mxSetData (X, C->x) ;
            mexMakeMemoryPersistent (C->x) ;
            C->x_shallow = false ;
            // treat C->x as if it were freed
            GB_AS_IF_FREE (C->x) ;
        }
    }

    // set nrows, ncols, nzmax, and the pattern of A
    mxSetM (A, C->vlen) ;
    mxSetN (A, C->vdim) ;
    mxSetNzmax (A, cnz) ;

    if (!C_is_full)
    {
        mxFree (mxGetJc (A)) ;
        mxFree (mxGetIr (A)) ;
        mexMakeMemoryPersistent (C->p) ;
        mexMakeMemoryPersistent (C->i) ;
        mxSetJc (A, (size_t *) C->p) ;
        mxSetIr (A, (size_t *) C->i) ;

        // treat C->p as if freed
        GB_AS_IF_FREE (C->p) ;

        // treat C->i as if freed
        C->i_shallow = false ;
        GB_AS_IF_FREE (C->i) ;
    }

    // free C, but leave any shallow components untouched
    // since these have been transplanted into the built-in matrix.
    GrB_Matrix_free_(handle) ;

    if (create_struct)
    {
        // create the type
        mxArray *atype = GB_mx_Type_to_mxstring (ctype) ;
        // create the iso flag
        mxArray *c_iso = mxCreateLogicalScalar (C_iso) ;
        // create the output struct
        Astruct = mxCreateStructMatrix (1, 1,
           (X == NULL) ? 3 : 4, MatrixFields) ;
        mxSetFieldByNumber (Astruct, 0, 0, A) ;
        mxSetFieldByNumber (Astruct, 0, 1, atype) ;
        mxSetFieldByNumber (Astruct, 0, 2, c_iso) ;
        if (X != NULL)
        {
            mxSetFieldByNumber (Astruct, 0, 3, X) ;
        }
        return (Astruct) ;
    }
    else
    {
        return (A) ;
    }
}

