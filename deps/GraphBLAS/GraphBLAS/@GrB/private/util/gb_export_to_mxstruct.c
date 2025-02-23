//------------------------------------------------------------------------------
// gb_export_to_mxstruct: export a GrB_Matrix to a MATLAB struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// The input GrB_Matrix A is exported to a GraphBLAS matrix struct G, and freed.

// The input GrB_Matrix A must be deep.  The output is a MATLAB struct
// holding the content of the GrB_Matrix.

// The GraphBLASv4 and v5 structs are identical, except that s has size 9
// in v4 and size 10 in v5.  The added s [9] entry is true if the matrix is
// uniform valued.  If the matrix is uniform-valued, the x array is only
// large enough to hold a single entry.

// GraphBLASv5 and GraphBLASv5_1 are identical, except that s [9] is present
// but always false for GraphBLASv5.

// GraphBLASv7_3 is identical to GraphBLASv5_1, except that it adds the Y
// hyper_hash with 3 components: Yp, Yi, and Yx.

// GraphBLASv10 is identical to GraphBLASv7_3, except that Ap, Ah, Ai, Yp, Yi,
// and Yx can be uint32_t.  The fields are the same as GraphBLASv7_3.

// mxGetData and mxSetData are used instead of the MATLAB-recommended
// mxGetDoubles, etc, because mxGetData and mxSetData work best for Octave, and
// they work fine for MATLAB since GraphBLAS requires R2018a with the
// interleaved complex data type.

// This function accesses GB_methods inside GraphBLAS.

#include "gb_interface.h"

// for hypersparse, sparse, or full matrices
static const char *MatrixFields [9] =
{
    // these fields are identical to GraphBLASv5_1, except for the name
    // of the first field
    "GraphBLASv10",     // 0: "logical", "int8", ... "double",
                        //    "single complex", or "double complex"
    "s",                // 1: all scalar info goes here
    "x",                // 2: array of uint8, size (sizeof(type)*nzmax), or
                        //    just sizeof(type) if the matrix is uniform-valued
    "p",                // 3: array of uint32_t or uint64_t, size plen+1
    "i",                // 4: array of uint32_t or uint64_t, size nzmax
    "h",                // 5: array of uint32_t or uint64_t, size plen if hyper
    // added for v7.2: for hypersparse matrices only:
    "Yp",               // 6: Yp, uint32_t or uint64_t array, size yvdim+1
    "Yi",               // 7: Yi, uint32_t or uint64_t array, size nvec (s[3])
    "Yx"                // 8: Yx, uint32_t or uint64_t array, size nvec
} ;

// for bitmap matrices only
static const char *Bitmap_MatrixFields [4] =
{
    "GraphBLASv10",     // 0: "logical", "int8", ... "double",
                        //    "single complex", or "double complex"
    "s",                // 1: all scalar info goes here
    "x",                // 2: array of uint8, size (sizeof(type)*nzmax), or
                        //    just sizeof(type) if the matrix is uniform-valued
    "b"                 // 3: array of int8_t, size nzmax, for bitmap only
} ;

//------------------------------------------------------------------------------

mxArray *gb_export_to_mxstruct  // return exported MATLAB struct G
(
    GrB_Matrix *A_handle        // matrix to export; freed on output
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (A_handle == NULL, "matrix missing") ;

    GrB_Matrix T = NULL ;
    if (gb_is_readonly (*A_handle))
    {
        // A is shallow so make a deep copy
        OK (GrB_Matrix_dup (&T, *A_handle)) ;
        OK (GrB_Matrix_free (A_handle)) ;
        (*A_handle) = T ;
    }

    GrB_Matrix A = (*A_handle) ;
    GrB_Matrix Y = NULL ;

    //--------------------------------------------------------------------------
    // make sure the matrix is finished, including the creation of Y
    //--------------------------------------------------------------------------

    OK1 (A, GrB_Matrix_wait (A, GrB_MATERIALIZE)) ;
    // OK (GxB_Matrix_fprint (A, "A to export", 0, NULL)) ;

    //--------------------------------------------------------------------------
    // extract the content of the GrB_Matrix and free it
    //--------------------------------------------------------------------------

    int sparsity_control ;
    OK (GrB_Matrix_get_INT32 (A, &sparsity_control, GxB_SPARSITY_CONTROL)) ;

    GxB_Container Container = GB_helper_container ( ) ;
    OK (GxB_unload_Matrix_into_Container (A, Container, NULL)) ;
    GrB_Matrix_free (&A) ;

    // get the scalars from the Container
    int sparsity_status = Container->format ;
    bool by_col = (Container->orientation == GrB_COLMAJOR) ;
    uint64_t nrows = Container->nrows ;
    uint64_t ncols = Container->ncols ;
    uint64_t nvals = Container->nvals ;
    uint64_t nrows_nonempty = Container->nrows_nonempty ;
    uint64_t ncols_nonempty = Container->ncols_nonempty ;
    bool iso = Container->iso ;
    int ro = 0 ;                        // ignored; no content is readonly
//  bool jumbled = Container->jumbled ; // not needed; matrix is not jumbled

    // get the vectors from the Container
    void *Ap = NULL ; uint64_t Ap_size, Ap_len ; GrB_Type Ap_type = NULL ;
    void *Ah = NULL ; uint64_t Ah_size, Ah_len ; GrB_Type Ah_type = NULL ;
    void *Ab = NULL ; uint64_t Ab_size, Ab_len ; GrB_Type Ab_type = NULL ;
    void *Ai = NULL ; uint64_t Ai_size, Ai_len ; GrB_Type Ai_type = NULL ;
    void *Ax = NULL ; uint64_t Ax_size, Ax_len ; GrB_Type Ax_type = NULL ;

    OK (GxB_Vector_unload (Container->p, &Ap, &Ap_type, &Ap_len, &Ap_size, &ro,
        NULL)) ;
    OK (GxB_Vector_unload (Container->h, &Ah, &Ah_type, &Ah_len, &Ah_size, &ro,
        NULL)) ;
    OK (GxB_Vector_unload (Container->b, &Ab, &Ab_type, &Ab_len, &Ab_size, &ro,
        NULL)) ;
    OK (GxB_Vector_unload (Container->i, &Ai, &Ai_type, &Ai_len, &Ai_size, &ro,
        NULL)) ;
    OK (GxB_Vector_unload (Container->x, &Ax, &Ax_type, &Ax_len, &Ax_size, &ro,
        NULL)) ;

    // get the Y matrix from the Container
    void *Yp = NULL ; uint64_t Yp_size, Yp_len ; GrB_Type Yp_type = NULL ;
    void *Yi = NULL ; uint64_t Yi_size, Yi_len ; GrB_Type Yi_type = NULL ;
    void *Yx = NULL ; uint64_t Yx_size, Yx_len ; GrB_Type Yx_type = NULL ;
    uint64_t yncols = 0 ;
    if (Container->Y != NULL)
    {
        // remove Y from the Container and unload it; reusing the Container
        Y = Container->Y ;
        Container->Y = NULL ;
        OK (GxB_unload_Matrix_into_Container (Y, Container, NULL)) ;
        GrB_Matrix_free (&Y) ;
        OK (GxB_Vector_unload (Container->p, &Yp, &Yp_type, &Yp_len, &Yp_size,
            &ro, NULL)) ;
        OK (GxB_Vector_unload (Container->i, &Yi, &Yi_type, &Yi_len, &Yi_size,
            &ro, NULL)) ;
        OK (GxB_Vector_unload (Container->x, &Yx, &Yx_type, &Yx_len, &Yx_size,
            &ro, NULL)) ;
        yncols = Container->ncols ;
    }

    //--------------------------------------------------------------------------
    // construct the output struct for MATLAB
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

    // export the GraphBLAS Ax_type as a string
    mxSetFieldByNumber (G, 0, 0, gb_type_to_mxstring (Ax_type)) ;

    // export the scalar content
    mxArray *opaque = mxCreateNumericMatrix (1, 10, mxINT64_CLASS, mxREAL) ;
    int64_t *s = (int64_t *) mxGetData (opaque) ;
    s [0] = Ap_len - 1 ;                    // plen
    s [1] = (by_col) ? nrows : ncols ;      // vlen
    s [2] = (by_col) ? ncols : nrows ;      // vdim
    s [3] = (sparsity_status == GxB_HYPERSPARSE) ? Ah_len : (s [2]) ;
    s [4] = (by_col) ? ncols_nonempty : nrows_nonempty ;    // nvec_nonempty
    s [5] = sparsity_control ;
    s [6] = (int64_t) by_col ;
    s [7] = Ax_len ;
    s [8] = nvals ;
    s [9] = (int64_t) iso ;             // new in GraphBLASv5
    mxSetFieldByNumber (G, 0, 1, opaque) ;

    // These components do not need to be exported: Pending, nzombies,
    // queue_next, queue_head, enqueued, *_shallow, jumbled, logger,
    // hyper_switch, bitmap_switch.

    if (sparsity_status == GxB_SPARSE || sparsity_status == GxB_HYPERSPARSE)
    {
        // export the pointers
        mxClassID Ap_class = (Ap_type == GrB_UINT32) ?
            mxUINT32_CLASS : mxUINT64_CLASS ;
        mxArray *Ap_mx = mxCreateNumericMatrix (1, 0, Ap_class, mxREAL) ;
        mxSetN (Ap_mx, Ap_len) ;
        void *p = (void *) mxGetData (Ap_mx) ; gb_mxfree (&p) ;
        mxSetData (Ap_mx, Ap) ;
        mxSetFieldByNumber (G, 0, 3, Ap_mx) ;

        // export the indices
        mxClassID Ai_class = (Ai_type == GrB_UINT32) ?
            mxUINT32_CLASS : mxUINT64_CLASS ;
        mxArray *Ai_mx = mxCreateNumericMatrix (1, 0, Ai_class, mxREAL) ;
        if (Ai_size > 0)
        { 
            mxSetN (Ai_mx, Ai_len) ;
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

    mxClassID Ah_class = (Ah_type == GrB_UINT32) ?
        mxUINT32_CLASS : mxUINT64_CLASS ;

    if (sparsity_status == GxB_HYPERSPARSE)
    {
        // export the hyperlist
        mxArray *Ah_mx = mxCreateNumericMatrix (1, 0, Ah_class, mxREAL) ;
        if (Ah_size > 0)
        { 
            mxSetN (Ah_mx, Ah_len) ;
            void *p = (void *) mxGetData (Ah_mx) ; gb_mxfree (&p) ;
            mxSetData (Ah_mx, Ah) ;
        }
        mxSetFieldByNumber (G, 0, 5, Ah_mx) ;

        if (Yp != NULL)
        {

            // export Yp, of size yncols+1
            mxArray *Yp_mx = mxCreateNumericMatrix (1, 0, Ah_class, mxREAL) ;
            mxSetN (Yp_mx, yncols+1) ;
            void *p = (void *) mxGetData (Yp_mx) ; gb_mxfree (&p) ;
            mxSetData (Yp_mx, Yp) ;
            mxSetFieldByNumber (G, 0, 6, Yp_mx) ;

            // export Yi, of size Ah_len
            mxArray *Yi_mx = mxCreateNumericMatrix (1, 0, Ah_class, mxREAL) ;
            mxSetN (Yi_mx, Ah_len) ;
            p = (void *) mxGetData (Yi_mx) ; gb_mxfree (&p) ;
            mxSetData (Yi_mx, Yi) ;
            mxSetFieldByNumber (G, 0, 7, Yi_mx) ;

            // export Yx, of size Ah_len
            mxArray *Yx_mx = mxCreateNumericMatrix (1, 0, Ah_class, mxREAL) ;
            mxSetN (Yx_mx, Ah_len) ;
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
            mxSetN (Ab_mx, Ab_len) ;
            void *p = (void *) mxGetData (Ab_mx) ; gb_mxfree (&p) ;
            mxSetData (Ab_mx, Ab) ;
        }
        mxSetFieldByNumber (G, 0, 3, Ab_mx) ;
    }

    //--------------------------------------------------------------------------
    // return the built-in MATLAB struct containing the GrB_Matrix components
    //--------------------------------------------------------------------------

    return (G) ;
}

