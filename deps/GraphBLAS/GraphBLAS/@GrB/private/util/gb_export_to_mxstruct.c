//------------------------------------------------------------------------------
// gb_export_to_mxstruct: export a GrB_Matrix to a built-in struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input GrB_Matrix A is exported to a GraphBLAS matrix struct G, and freed.

// The input GrB_Matrix A must be deep.  The output is a built-in struct
// holding the content of the GrB_Matrix.

// The GraphBLASv4 and v5 structs are identical, except that s has size 9
// in v4 and size 10 in v5.  The added s [9] entry is true if the matrix is
// uniform valued.  If the matrix is uniform-valued, the x array is only
// large enough to hold a single entry.

// GraphBLASv5 and GraphBLASv5_1 are identical, except that s [9] is present
// but always false for GraphBLASv5.

// GraphBLASv7_3 is identical to GraphBLASv5_1, except that it adds the Y
// hyper_hash with 3 components: Yp, Yi, and Yx.

// mxGetData and mxSetData are used instead of the MATLAB-recommended
// mxGetDoubles, etc, because mxGetData and mxSetData work best for Octave, and
// they work fine for MATLAB since GraphBLAS requires R2018a with the
// interleaved complex data type.

#include "gb_interface.h"

// for hypersparse, sparse, or full matrices
static const char *MatrixFields [9] =
{
    // these fields are identical to GraphBLASv5_1, except for the name
    // of the first field
    "GraphBLASv7_3",    // 0: "logical", "int8", ... "double",
                        //    "single complex", or "double complex"
    "s",                // 1: all scalar info goes here
    "x",                // 2: array of uint8, size (sizeof(type)*nzmax), or
                        //    just sizeof(type) if the matrix is uniform-valued
    "p",                // 3: array of int64_t, size plen+1
    "i",                // 4: array of int64_t, size nzmax
    "h",                // 5: array of int64_t, size plen if hypersparse
    // added for v7.2: for hypersparse matrices only:
    "Yp",               // 6: Y->p, a uint64_t array of size Y->vdim+1
    "Yi",               // 7: Y->i, a uint64_t array of size nvec (s[3])
    "Yx"                // 8: Y->x, an int64_t array of size nvec
} ;

// for bitmap matrices only
static const char *Bitmap_MatrixFields [4] =
{
    "GraphBLASv7_3",    // 0: "logical", "int8", ... "double",
                        //    "single complex", or "double complex"
    "s",                // 1: all scalar info goes here
    "x",                // 2: array of uint8, size (sizeof(type)*nzmax), or
                        //    just sizeof(type) if the matrix is uniform-valued
    "b"                 // 3: array of int8_t, size nzmax, for bitmap only
} ;

//------------------------------------------------------------------------------

mxArray *gb_export_to_mxstruct  // return exported built-in struct G
(
    GrB_Matrix *A_handle        // matrix to export; freed on output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (A_handle == NULL, "matrix missing") ;

    GrB_Matrix T = NULL ;
    if (GB_is_shallow (*A_handle))
    {
        // A is shallow so make a deep copy
        OK (GrB_Matrix_dup (&T, *A_handle)) ;
        OK (GrB_Matrix_free (A_handle)) ;
        (*A_handle) = T ;
    }

    GrB_Matrix A = (*A_handle) ;
    GrB_Matrix Y = NULL ;

    //--------------------------------------------------------------------------
    // make sure the matrix is finished, including the creation of A->Y
    //--------------------------------------------------------------------------

    OK1 (A, GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;

    //--------------------------------------------------------------------------
    // get the sparsity_status and CSR/CSC format
    //--------------------------------------------------------------------------

    GxB_Format_Value fmt ;
    int sparsity_status, sparsity_control ;
    OK (GxB_Matrix_Option_get (A, GxB_SPARSITY_STATUS,  &sparsity_status)) ;
    OK (GxB_Matrix_Option_get (A, GxB_SPARSITY_CONTROL, &sparsity_control)) ;
    OK (GxB_Matrix_Option_get (A, GxB_FORMAT, &fmt)) ;

    //--------------------------------------------------------------------------
    // extract the opaque content not provided by GxB*export
    //--------------------------------------------------------------------------

    int64_t nzmax = GB_nnz_max (A) ;
    int64_t plen = A->plen ;
    int64_t nvec_nonempty = A->nvec_nonempty ;

    //--------------------------------------------------------------------------
    // extract the content of the GrB_Matrix and free it
    //--------------------------------------------------------------------------

    size_t type_size = 0 ;
    GrB_Type type = NULL ;
    GrB_Index nrows = 0, ncols = 0 ;
    int8_t *Ab = NULL ;
    uint64_t *Ap = NULL, *Ah = NULL, *Ai = NULL ;
    void *Ax = NULL ;
    GrB_Index Ap_size = 0, Ah_size = 0, Ab_size = 0, Ai_size = 0, Ax_size = 0 ;
    int64_t nvals = 0, nvec = 0 ;
    bool by_col = (fmt == GxB_BY_COL) ;
    bool iso = false ;

    GrB_Type ytype = NULL ;
    uint64_t *Yp = NULL ; GrB_Index Yp_size = 0 ;
    uint64_t *Yi = NULL ; GrB_Index Yi_size = 0 ;
    void     *Yx = NULL ; GrB_Index Yx_size = 0 ;
    uint64_t yvdim, ynrows ;

    switch (sparsity_status)
    {
        case GxB_FULL :
            if (by_col)
            {
                OK (GxB_Matrix_export_FullC (&A, &type, &nrows, &ncols,
                    &Ax, &Ax_size, &iso, NULL)) ;
            }
            else
            {
                OK (GxB_Matrix_export_FullR (&A, &type, &nrows, &ncols,
                    &Ax, &Ax_size, &iso, NULL)) ;
            }
            break ;

        case GxB_SPARSE :
            if (by_col)
            {
                OK (GxB_Matrix_export_CSC (&A, &type, &nrows, &ncols,
                    &Ap, &Ai, &Ax,
                    &Ap_size, &Ai_size, &Ax_size, &iso, NULL, NULL)) ;
            }
            else
            {
                OK (GxB_Matrix_export_CSR (&A, &type, &nrows, &ncols,
                    &Ap, &Ai, &Ax,
                    &Ap_size, &Ai_size, &Ax_size, &iso, NULL, NULL)) ;
            }
            break ;

        case GxB_HYPERSPARSE :

            // export and free the A->Y hyper_hash.  It is always sparse,
            // GrB_UINT64, held by column, and non-iso
            OK (GxB_unpack_HyperHash (A, &Y, NULL)) ;
            if (Y != NULL)
            {
                OK (GxB_Matrix_export_CSC (&Y, &ytype, &ynrows, &yvdim,
                    &Yp, &Yi, &Yx, &Yp_size, &Yi_size, &Yx_size,
                    NULL, NULL, NULL)) ;
            }

            // export and free the rest of the hypersparse matrix
            if (by_col)
            {
                OK (GxB_Matrix_export_HyperCSC (&A, &type, &nrows, &ncols,
                    &Ap, &Ah, &Ai, &Ax,
                    &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso,
                    &nvec, NULL, NULL)) ;
            }
            else
            {
                OK (GxB_Matrix_export_HyperCSR (&A, &type, &nrows, &ncols,
                    &Ap, &Ah, &Ai, &Ax,
                    &Ap_size, &Ah_size, &Ai_size, &Ax_size, &iso,
                    &nvec, NULL, NULL)) ;
            }
            break ;

        case GxB_BITMAP :
            if (by_col)
            {
                OK (GxB_Matrix_export_BitmapC (&A, &type, &nrows, &ncols,
                    &Ab, &Ax, &Ab_size, &Ax_size, &iso, &nvals, NULL)) ;
            }
            else
            {
                OK (GxB_Matrix_export_BitmapR (&A, &type, &nrows, &ncols,
                    &Ab, &Ax, &Ab_size, &Ax_size, &iso, &nvals, NULL)) ;
            }
            break ;

        default: ;
    }

    OK (GxB_Type_size (&type_size, type)) ;

    //--------------------------------------------------------------------------
    // construct the output struct
    //--------------------------------------------------------------------------

    mxArray *G ;
    switch (sparsity_status)
    {
        case GxB_FULL :
            // A is full, with 3 fields: GraphBLAS*, s, x
            G = mxCreateStructMatrix (1, 1, 3, MatrixFields) ;
            break ;

        case GxB_SPARSE :
            // A is sparse, with 5 fields: GraphBLAS*, s, x, p, i
            G = mxCreateStructMatrix (1, 1, 5, MatrixFields) ;
            break ;

        case GxB_HYPERSPARSE :
            // A is hypersparse, with 6 or 9 fields: GraphBLAS*, s, x, p, i, h,
            // Yp, Yi, Yx
            G = mxCreateStructMatrix (1, 1, (Yp == NULL) ? 6 : 9,
                MatrixFields) ;
            break ;

        case GxB_BITMAP :
            // A is bitmap, with 4 fields: GraphBLAS*, s, x, b
            G = mxCreateStructMatrix (1, 1, 4, Bitmap_MatrixFields) ;
            break ;

        default : ERROR ("invalid GraphBLAS struct") ;
    }

    //--------------------------------------------------------------------------
    // export content into the output struct
    //--------------------------------------------------------------------------

    // export the GraphBLAS type as a string
    mxSetFieldByNumber (G, 0, 0, gb_type_to_mxstring (type)) ;

    // export the scalar content
    mxArray *opaque = mxCreateNumericMatrix (1, 10, mxINT64_CLASS, mxREAL) ;
    int64_t *s = (int64_t *) mxGetData (opaque) ;
    s [0] = plen ;
    s [1] = (by_col) ? nrows : ncols ;  // was A->vlen ;
    s [2] = (by_col) ? ncols : nrows ;  // was A->vdim ;
    s [3] = (sparsity_status == GxB_HYPERSPARSE) ? nvec : (s [2]) ;
    s [4] = nvec_nonempty ;
    s [5] = sparsity_control ;
    s [6] = (int64_t) by_col ;
    s [7] = nzmax ;
    s [8] = nvals ;
    s [9] = (int64_t) iso ;             // new in GraphBLASv5
    mxSetFieldByNumber (G, 0, 1, opaque) ;

    // These components do not need to be exported: Pending, nzombies,
    // queue_next, queue_head, enqueued, *_shallow, jumbled, logger,
    // hyper_switch, bitmap_switch.

    if (sparsity_status == GxB_SPARSE || sparsity_status == GxB_HYPERSPARSE)
    {
        // export the pointers
        mxArray *Ap_mx = mxCreateNumericMatrix (1, 0, mxINT64_CLASS, mxREAL) ;
        mxSetN (Ap_mx, Ap_size / sizeof (int64_t)) ;
        void *p = (void *) mxGetData (Ap_mx) ; gb_mxfree (&p) ;
        mxSetData (Ap_mx, Ap) ;
        mxSetFieldByNumber (G, 0, 3, Ap_mx) ;

        // export the indices
        mxArray *Ai_mx = mxCreateNumericMatrix (1, 0, mxINT64_CLASS, mxREAL) ;
        if (Ai_size > 0)
        { 
            mxSetN (Ai_mx, Ai_size / sizeof (int64_t)) ;
            p = (void *) mxGetData (Ai_mx) ; gb_mxfree (&p) ;
            mxSetData (Ai_mx, Ai) ;
        }
        mxSetFieldByNumber (G, 0, 4, Ai_mx) ;
    }

    // export the values as uint8
    mxArray *Ax_mx = mxCreateNumericMatrix (1, 0, mxUINT8_CLASS, mxREAL) ;
    if (Ax_size > 0)
    { 
        mxSetN (Ax_mx, Ax_size) ;
        void *p = mxGetData (Ax_mx) ; gb_mxfree (&p) ;
        mxSetData (Ax_mx, Ax) ;
    }
    mxSetFieldByNumber (G, 0, 2, Ax_mx) ;

    if (sparsity_status == GxB_HYPERSPARSE)
    {
        // export the hyperlist
        mxArray *Ah_mx = mxCreateNumericMatrix (1, 0, mxINT64_CLASS, mxREAL) ;
        if (Ah_size > nvec * sizeof (int64_t))
        {
            memset (Ah + nvec, 0, Ah_size - nvec * sizeof (int64_t)) ;
        }
        if (Ah_size > 0)
        { 
            mxSetN (Ah_mx, Ah_size / sizeof (int64_t)) ;
            void *p = (void *) mxGetData (Ah_mx) ; gb_mxfree (&p) ;
            mxSetData (Ah_mx, Ah) ;
        }
        mxSetFieldByNumber (G, 0, 5, Ah_mx) ;

        if (Yp != NULL)
        {

            // export Yp, of size yvdim+1
            mxArray *Yp_mx = mxCreateNumericMatrix (1, 0, mxUINT64_CLASS,
                mxREAL) ;
            mxSetN (Yp_mx, yvdim+1) ;
            void *p = (void *) mxGetData (Yp_mx) ; gb_mxfree (&p) ;
            mxSetData (Yp_mx, Yp) ;
            mxSetFieldByNumber (G, 0, 6, Yp_mx) ;

            // export Yi, of size nvec
            mxArray *Yi_mx = mxCreateNumericMatrix (1, 0, mxUINT64_CLASS,
                mxREAL) ;
            mxSetN (Yi_mx, nvec) ;
            p = (void *) mxGetData (Yi_mx) ; gb_mxfree (&p) ;
            mxSetData (Yi_mx, Yi) ;
            mxSetFieldByNumber (G, 0, 7, Yi_mx) ;

            // export Yx, of size nvec
            mxArray *Yx_mx = mxCreateNumericMatrix (1, 0, mxUINT64_CLASS,
                mxREAL) ;
            mxSetN (Yx_mx, nvec) ;
            p = (void *) mxGetData (Yx_mx) ; gb_mxfree (&p) ;
            mxSetData (Yx_mx, Yx) ;
            mxSetFieldByNumber (G, 0, 8, Yx_mx) ;
        }
    }

    if (sparsity_status == GxB_BITMAP)
    { 
        // export the bitmap
        mxArray *Ab_mx = mxCreateNumericMatrix (1, 0, mxINT8_CLASS, mxREAL) ;
        if (Ab_size > 0)
        { 
            mxSetN (Ab_mx, Ab_size) ;
            void *p = (void *) mxGetData (Ab_mx) ; gb_mxfree (&p) ;
            mxSetData (Ab_mx, Ab) ;
        }
        mxSetFieldByNumber (G, 0, 3, Ab_mx) ;
    }

    //--------------------------------------------------------------------------
    // return the built-in struct
    //--------------------------------------------------------------------------

    return (G) ;
}

