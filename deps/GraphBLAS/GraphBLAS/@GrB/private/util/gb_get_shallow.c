//------------------------------------------------------------------------------
// gb_get_shallow: create a shallow copy of a MATLAB sparse matrix or struct
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// A = gb_get_shallow (X) constructs a shallow GrB_Matrix from a MATLAB
// mxArray, which can either be a MATLAB sparse matrix (double, complex, or
// logical) or a MATLAB struct that contains a GraphBLAS matrix.

// X must not be NULL, but it can be an empty matrix, as X = [ ] or even X = ''
// (the empty string).  In this case, A is returned as NULL.  This is not an
// error here, since the caller might be getting an optional input matrix, such
// as Cin or the Mask.

// For v4, iso is false, and the s component has length 9.
// For v5, iso is present but false, and the s component has length 10.
// For v5_1, iso is true/false, and the s component has length 10.
// For v7_3: the same content as v5_1, except that Yp, Yi, and Yx are added.
// For v10: Ap, Ah, Ai, Yp, Yi, and Yx can be 32-bit or 64-bit

// mxGetData is used instead of the MATLAB-recommended mxGetDoubles, etc,
// because mxGetData works best for Octave, and it works fine for MATLAB
// since GraphBLAS requires R2018a with the interleaved complex data type.

// This function accesses GB_methods inside GraphBLAS.

#include "gb_interface.h"

#define IF(error,message) \
    CHECK_ERROR (error, "invalid GraphBLAS struct (" message ")" ) ;

GrB_Matrix gb_get_shallow   // shallow copy of MATLAB sparse matrix or struct
(
    const mxArray *X
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    CHECK_ERROR (X == NULL, "matrix missing") ;

    //--------------------------------------------------------------------------
    // turn off the burble
    //--------------------------------------------------------------------------

    int burble ;
    OK (GrB_Global_get_INT32 (GrB_GLOBAL, &burble, GxB_BURBLE)) ;
    OK (GrB_Global_set_INT32 (GrB_GLOBAL, false, GxB_BURBLE)) ;

    //--------------------------------------------------------------------------
    // construct the shallow GrB_Matrix
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL, Y = NULL ;

    if (gb_mxarray_is_empty (X))
    { 

        //----------------------------------------------------------------------
        // matrix is empty
        //----------------------------------------------------------------------

        // X is a 0-by-0 MATLAB-in matrix.  Create a new 0-by-0 matrix of the
        // same type as X, with the default format.
        OK (GrB_Matrix_new (&A, gb_mxarray_type (X), 0, 0)) ;

    }
    else if (mxIsStruct (X))
    { 

        //----------------------------------------------------------------------
        // construct a shallow GrB_Matrix copy from a MATLAB struct
        //----------------------------------------------------------------------

        bool GraphBLASv4 = false ;
        bool GraphBLASv3 = false ;

        // get the type
        mxArray *mx_type = mxGetField (X, 0, "GraphBLASv10") ;
        if (mx_type == NULL)
        {
            // check if it is a GraphBLASv7_3 struct
            mx_type = mxGetField (X, 0, "GraphBLASv7_3") ;
        }
        if (mx_type == NULL)
        {
            // check if it is a GraphBLASv5_1 struct
            mx_type = mxGetField (X, 0, "GraphBLASv5_1") ;
        }

        if (mx_type == NULL)
        {
            // check if it is a GraphBLASv5 struct
            mx_type = mxGetField (X, 0, "GraphBLASv5") ;
        }
        if (mx_type == NULL)
        {
            // check if it is a GraphBLASv4 struct
            mx_type = mxGetField (X, 0, "GraphBLASv4") ;
            GraphBLASv4 = true ;
        }
        if (mx_type == NULL)
        {
            // check if it is a GraphBLASv3 struct
            mx_type = mxGetField (X, 0, "GraphBLAS") ;
            GraphBLASv3 = true ;
        }
        CHECK_ERROR (mx_type == NULL, "not a GraphBLAS struct") ;

        GrB_Type Ax_type = gb_mxstring_to_type (mx_type) ;
        size_t type_size ;
        OK (GrB_Type_get_SIZE (Ax_type, &type_size, GrB_SIZE)) ;

        // get the scalar info
        mxArray *opaque = mxGetField (X, 0, "s") ;
        IF (opaque == NULL, ".s missing") ;
        IF (mxGetM (opaque) != 1, ".s wrong size") ;
        size_t s_size = mxGetN (opaque) ;
        if (GraphBLASv3)
        {
            IF (s_size != 8, ".s wrong size") ;
        }
        else if (GraphBLASv4)
        {
            IF (s_size != 9, ".s wrong size") ;
        }
        else
        {
            IF (s_size != 10, ".s wrong size") ;
        }
        int64_t *s = (int64_t *) mxGetData (opaque) ;
        int64_t plen          = s [0] ;
        int64_t vlen          = s [1] ;
        int64_t vdim          = s [2] ;
        int64_t nvec          = s [3] ;
        int64_t nvec_nonempty = s [4] ;
        bool    by_col        = (bool) (s [6]) ;
        int64_t nzmax         = s [7] ;

        int sparsity_status, sparsity_control ;
        int64_t nvals ;
        bool iso ;

        if (GraphBLASv3)
        {
            // GraphBLASv3 struct: sparse or hypersparse only
            sparsity_control = GxB_AUTO_SPARSITY ;
            nvals            = 0 ;
            iso              = false ;
        }
        else
        {
            // GraphBLASv4 or later struct: sparse, hypersparse, bitmap, or full
            sparsity_control = (int) (s [5]) ;
            nvals            = s [8] ;
            if (GraphBLASv4)
            {
                // GraphBLASv4: iso is always false
                iso = false ;
            }
            else
            {
                // GraphBLASv5 and GraphBLASv5_1: iso is present as s [9]
                // GraphBLASv5: iso is present as s [9] but always false
                iso = (bool) s [9] ;
            }
        }

        int nfields = mxGetNumberOfFields (X) ;
        switch (nfields)
        {
            case 3 :
                // A is full, with 3 fields: GraphBLAS*, s, x
                sparsity_status = GxB_FULL ;
                break ;

            case 5 :
                // A is sparse, with 5 fields: GraphBLAS*, s, x, p, i
                sparsity_status = GxB_SPARSE ;
                break ;

            case 6 :
            case 9 :
                // A is hypersparse, with 6 fields: GraphBLAS*, s, x, p, i, h
                // or with 9 fields: Yp, Yi, and Yx added.
                sparsity_status = GxB_HYPERSPARSE ;
                // GraphBLAS v9 and earlier can export a matrix to the MATLAB
                // struct with plen of 1 but nvec of 0.  Fix it here for v9 and
                // earlier structs, and also in gb_export_to_mxstruct for v10:
                plen = nvec ;
                break ;

            case 4 :
                // A is bitmap, with 4 fields: GraphBLAS*, s, x, b
                sparsity_status = GxB_BITMAP ;
                break ;

            default : ERROR ("invalid GraphBLAS struct") ;
        }

        // each component
        void   *Ap = NULL ; uint64_t Ap_size = 0 ;
        void   *Ah = NULL ; uint64_t Ah_size = 0 ;
        void   *Ai = NULL ; uint64_t Ai_size = 0 ;
        int8_t *Ab = NULL ; uint64_t Ab_size = 0, Ab_len = 0 ;
        void   *Ax = NULL ; uint64_t Ax_size = 0, Ax_len = 0 ;
        void   *Yp = NULL ; uint64_t Yp_size = 0, Yp_len = 0 ;
        void   *Yi = NULL ; uint64_t Yi_size = 0, Yi_len = 0 ;
        void   *Yx = NULL ; uint64_t Yx_size = 0, Yx_len = 0 ;
        int64_t yvdim = 0 ;

        // these are revised below:
        bool Ap_is_32 = false ; // controls Ap
        bool Aj_is_32 = false ; // controls Ah, Yp, Yi, Yx
        bool Ai_is_32 = false ; // controls Ai
        size_t psize = Ap_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
        size_t jsize = Aj_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
        size_t isize = Ai_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
        GrB_Type Ap_type = Ap_is_32 ? GrB_UINT32 : GrB_UINT64 ;
        GrB_Type Aj_type = Aj_is_32 ? GrB_UINT32 : GrB_UINT64 ;
        GrB_Type Ai_type = Ai_is_32 ? GrB_UINT32 : GrB_UINT64 ;

        if (sparsity_status == GxB_HYPERSPARSE || sparsity_status == GxB_SPARSE)
        {
            // A is hypersparse or sparse

            // get Ap
            mxArray *Ap_mx = mxGetField (X, 0, "p") ;
            IF (Ap_mx == NULL, ".p missing") ;
            IF (mxGetM (Ap_mx) != 1, ".p wrong size") ;
            mxClassID class = mxGetClassID (Ap_mx) ;
            IF (!(class == mxUINT64_CLASS || class == mxUINT32_CLASS ||
                  class == mxINT64_CLASS), ".p wrong class")
            Ap_is_32 = (class == mxUINT32_CLASS) ;
            psize = Ap_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
            Ap_type = Ap_is_32 ? GrB_UINT32 : GrB_UINT64 ;
            Ap = (void *) mxGetData (Ap_mx) ;
            Ap_size = mxGetN (Ap_mx) * psize ;
            IF (mxGetN (Ap_mx) < plen+1, ".p wrong size")
            if (GraphBLASv3)
            {
                uint64_t *Ap64 = (uint64_t *) Ap ;
                nvals = Ap64 [plen] ;
            }

            // get Ai
            mxArray *Ai_mx = mxGetField (X, 0, "i") ;
            IF (Ai_mx == NULL, ".i missing") ;
            IF (mxGetM (Ai_mx) != 1, ".i wrong size") ;
            class = mxGetClassID (Ai_mx) ;
            IF (!(class == mxUINT64_CLASS || class == mxUINT32_CLASS ||
                  class == mxINT64_CLASS), ".i wrong class")
            Ai_is_32 = (class == mxUINT32_CLASS) ;
            isize = Ai_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
            Ai_type = Ai_is_32 ? GrB_UINT32 : GrB_UINT64 ;
            Ai_size = mxGetN (Ai_mx) * isize ;
            IF (mxGetN (Ai_mx) < nvals, ".i wrong size") ;
            Ai = (Ai_size == 0) ? NULL : ((void *) mxGetData (Ai_mx)) ;
        }

        // get the values
        mxArray *Ax_mx = mxGetField (X, 0, "x") ;
        IF (Ax_mx == NULL, ".x missing") ;
        IF (mxGetM (Ax_mx) != 1, ".x wrong size") ;
        Ax_size = mxGetN (Ax_mx) ;
        Ax_len = Ax_size / type_size ;
        Ax = (Ax_size == 0) ? NULL : ((void *) mxGetData (Ax_mx)) ;

        if (sparsity_status == GxB_SPARSE)
        {
            // A is sparse; determine Aj_is_32
            Aj_is_32 = (vdim <= ((int64_t) (1ULL << 31))) ;
            Aj_type = Aj_is_32 ? GrB_UINT32 : GrB_UINT64 ;
            jsize = Aj_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

        }
        else if (sparsity_status == GxB_HYPERSPARSE)
        { 
            // A is hypersparse
            // get the hyperlist
            mxArray *Ah_mx = mxGetField (X, 0, "h") ;
            IF (Ah_mx == NULL, ".h missing") ;
            IF (mxGetM (Ah_mx) != 1, ".h wrong size") ;
            mxClassID Ah_class = mxGetClassID (Ah_mx) ;
            IF (!(Ah_class == mxUINT64_CLASS || Ah_class == mxUINT32_CLASS ||
                  Ah_class == mxINT64_CLASS), ".h wrong class")
            Aj_is_32 = (Ah_class == mxUINT32_CLASS) ;
            Aj_type = Aj_is_32 ? GrB_UINT32 : GrB_UINT64 ;
            jsize = Aj_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
            Ah_size = mxGetN (Ah_mx) * jsize ;
            Ah = (Ah_size == 0) ? NULL : ((void *) mxGetData (Ah_mx)) ;

            // get the hyper_hash, if it exists

            if (nfields == 9)
            { 
                // get Yp, Yi, and Yx

                // Yp must be 1-by-(yvdim+1), with the same class as Ah
                mxArray *Yp_mx = mxGetField (X, 0, "Yp") ;
                IF (Yp_mx == NULL, ".Yp missing") ;
                IF (mxGetM (Yp_mx) != 1, ".Yp wrong size") ;
                yvdim = mxGetN (Yp_mx) - 1 ;
                IF (mxGetClassID (Yp_mx) != Ah_class, ".Yp wrong class") ;
                Yp_len = mxGetN (Yp_mx) ;
                Yp_size = Yp_len * jsize ;
                Yp = (Yp_size == 0) ? NULL : ((void *) mxGetData (Yp_mx)) ;

                // Yi must be 1-by-nvec, with the same class as Ah
                mxArray *Yi_mx = mxGetField (X, 0, "Yi") ;
                IF (Yi_mx == NULL, ".Yi missing") ;
                IF (mxGetM (Yi_mx) != 1, ".Yi wrong size") ;
                IF (mxGetN (Yi_mx) != nvec, ".Yi wrong size") ;
                IF (mxGetClassID (Yi_mx) != Ah_class, ".Yi wrong class") ;
                Yi_len = mxGetN (Yi_mx) ;
                Yi_size = Yi_len * jsize ;
                Yi = (Yi_size == 0) ? NULL : ((void *) mxGetData (Yi_mx)) ;

                // Yx must be 1-by-nvec
                mxArray *Yx_mx = mxGetField (X, 0, "Yx") ;
                IF (Yx_mx == NULL, ".Yx missing") ;
                IF (mxGetM (Yx_mx) != 1, ".Yx wrong size") ;
                IF (mxGetN (Yx_mx) != nvec, ".Yx wrong size") ;
                IF (mxGetClassID (Yx_mx) != Ah_class, ".Yx wrong class") ;
                Yx_len = mxGetN (Yx_mx) ;
                Yx_size = Yi_len * jsize ;
                Yx = (Yx_size == 0) ? NULL : ((void *) mxGetData (Yx_mx)) ;
            }
        }

        if (sparsity_status == GxB_BITMAP)
        { 
            // A is bitmap
            // get the bitmap
            mxArray *Ab_mx = mxGetField (X, 0, "b") ;
            IF (Ab_mx == NULL, ".b missing") ;
            IF (mxGetM (Ab_mx) != 1, ".b wrong size") ;
            IF (mxGetClassID (Ab_mx) != mxINT8_CLASS, ".Ab wrong class") ;
            Ab_len = mxGetN (Ab_mx) ;
            Ab_size = Ab_len ;
            Ab = (Ab_size == 0) ? NULL : ((int8_t *) mxGetData (Ab_mx)) ;
        }

        //----------------------------------------------------------------------
        // import the matrix
        //----------------------------------------------------------------------

        OK (GrB_Matrix_new (&A, GrB_BOOL, 0, 0)) ;

        GxB_Container Container = GB_helper_container ( ) ;

        if (Yp != NULL)
        { 
            // import the Y matrix using the Container
            OK (GrB_Matrix_new (&Y, GrB_UINT64, 0, 0)) ;
            Container->nrows = vdim ;
            Container->ncols = yvdim ;
            Container->nrows_nonempty = -1 ;
            Container->ncols_nonempty = -1 ;
            Container->nvals = nvec ;
            Container->format = GxB_SPARSE ;
            Container->orientation = GrB_COLMAJOR ;
            Container->iso = false ;
            Container->jumbled = false ;
            OK (GxB_Vector_load (Container->p, (void **) &Yp, Aj_type, Yp_len,
                Yp_size, GxB_IS_READONLY, NULL)) ;
            OK (GxB_Vector_load (Container->i, (void **) &Yi, Aj_type, Yi_len,
                Yi_size, GxB_IS_READONLY, NULL)) ;
            OK (GxB_Vector_load (Container->x, (void **) &Yx, Aj_type, nvec,
                Yx_size, GxB_IS_READONLY, NULL)) ;
            OK (GxB_load_Matrix_from_Container (Y, Container, NULL)) ;
        }

        // import the A matrix using the Container
        Container->nrows = (by_col) ? vlen : vdim ;
        Container->ncols = (by_col) ? vdim : vlen ;
        Container->nrows_nonempty = (by_col) ? -1 : nvec_nonempty ;
        Container->ncols_nonempty = (by_col) ? nvec_nonempty : -1 ;
        Container->nvals = nvals ;
        Container->format = sparsity_status ;
        Container->orientation = (by_col) ? GrB_COLMAJOR : GrB_ROWMAJOR ;
        Container->iso = iso ;
        Container->jumbled = false ;

        switch (sparsity_status)
        {
            case GxB_HYPERSPARSE : 
                Container->Y = Y ;
                Y = NULL ;
                OK (GxB_Vector_load (Container->h, (void **) &Ah, Aj_type,
                    plen, Ah_size, GxB_IS_READONLY, NULL)) ;
                // fall through to sparse case

            case GxB_SPARSE : 
                OK (GxB_Vector_load (Container->p, (void **) &Ap, Ap_type,
                    plen+1, Ap_size, GxB_IS_READONLY, NULL)) ;
                OK (GxB_Vector_load (Container->i, (void **) &Ai, Ai_type,
                    nvals, Ai_size, GxB_IS_READONLY, NULL)) ;
                break ;

            case GxB_BITMAP : 
                OK (GxB_Vector_load (Container->b, (void **) &Ab, GrB_INT8,
                    Ab_len, Ab_size, GxB_IS_READONLY, NULL)) ;
                break ;

            case GxB_FULL : 
                break ;

            default: ;
        }

        OK (GxB_Vector_load (Container->x, (void **) &Ax, Ax_type, Ax_len,
            Ax_size, GxB_IS_READONLY, NULL)) ;

        OK (GxB_load_Matrix_from_Container (A, Container, NULL)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // construct a shallow GrB_Matrix copy of a built-in MATLAB matrix
        //----------------------------------------------------------------------

        // get the type and dimensions
        bool X_is_sparse = mxIsSparse (X) ;
        GrB_Type Ax_type = gb_mxarray_type (X) ;
        uint64_t nrows = (uint64_t) mxGetM (X) ;
        uint64_t ncols = (uint64_t) mxGetN (X) ;
        uint64_t nvals ;
        OK (GrB_Matrix_new (&A, Ax_type, nrows, ncols)) ;

        // get Xp, Xi, and nvals
        uint64_t *Xp, *Xi ;
        if (X_is_sparse)
        { 
            Xp = (uint64_t *) mxGetJc (X) ;
            Xi = (uint64_t *) mxGetIr (X) ;
            nvals = Xp [ncols] ;
        }
        else
        { 
            Xp = NULL ;
            Xi = NULL ;
            nvals = nrows * ncols ;
        }

        // get the numeric data
        void *Xx = NULL ;
        size_t type_size = 0 ;
        if (Ax_type == GrB_FP64)
        { 
            // built-in sparse or full double matrix
            Xx = mxGetData (X) ;
            type_size = sizeof (double) ;
        }
        else if (Ax_type == GxB_FC64)
        { 
            // built-in sparse or full double complex matrix
            Xx = mxGetData (X) ;
            type_size = 2 * sizeof (double) ;
        }
        else if (Ax_type == GrB_BOOL)
        { 
            // built-in sparse or full logical matrix
            Xx = mxGetData (X) ;
            type_size = sizeof (bool) ;
        }
        else if (X_is_sparse)
        {
            // Built-in sparse matrices do not support any other kinds
            ERROR ("unsupported type") ;
        }
        else if (Ax_type == GrB_INT8)
        { 
            // full int8 matrix
            Xx = mxGetData (X) ;
            type_size = sizeof (int8_t) ;
        }
        else if (Ax_type == GrB_INT16)
        { 
            // full int16 matrix
            Xx = mxGetData (X) ;
            type_size = sizeof (int16_t) ;
        }
        else if (Ax_type == GrB_INT32)
        { 
            // full int32 matrix
            Xx = mxGetData (X) ;
            type_size = sizeof (int32_t) ;
        }
        else if (Ax_type == GrB_INT64)
        { 
            // full int64 matrix
            Xx = mxGetData (X) ;
            type_size = sizeof (int64_t) ;
        }
        else if (Ax_type == GrB_UINT8)
        { 
            // full uint8 matrix
            Xx = mxGetData (X) ;
            type_size = sizeof (uint8_t) ;
        }
        else if (Ax_type == GrB_UINT16)
        { 
            // full uint16 matrix
            Xx = mxGetData (X) ;
            type_size = sizeof (uint16_t) ;
        }
        else if (Ax_type == GrB_UINT32)
        { 
            // full uint32 matrix
            Xx = mxGetData (X) ;
            type_size = sizeof (uint32_t) ;
        }
        else if (Ax_type == GrB_UINT64)
        { 
            // full uint64 matrix
            Xx = mxGetData (X) ;
            type_size = sizeof (uint64_t) ;
        }
        else if (Ax_type == GrB_FP32)
        { 
            // full single matrix
            Xx = mxGetData (X) ;
            type_size = sizeof (float) ;
        }
        else if (Ax_type == GxB_FC32)
        { 
            // full single complex matrix
            Xx = mxGetData (X) ;
            type_size = 2 * sizeof (float) ;
        }
        else
        {
            ERROR ("unsupported type") ;
        }

        uint64_t Xx_size = (nvals) * type_size  ;

        GxB_Container Container = GB_helper_container ( ) ;
        Container->nrows = nrows ;
        Container->ncols = ncols ;
        Container->nvals = nvals ;
        Container->nrows_nonempty = -1 ;
        Container->ncols_nonempty = -1 ;
        Container->orientation = GrB_COLMAJOR ;
        Container->iso = false ;
        Container->jumbled = false ;

        if (X_is_sparse)
        { 
            // import the matrix in CSC format (all-64-bit)
            uint64_t Xp_size = (ncols + 1) * sizeof (uint64_t) ;
            uint64_t Xi_size = (nvals) * sizeof (uint64_t) ;
            OK (GxB_Vector_load (Container->p, (void **) &Xp, GrB_UINT64,
                ncols+1, Xp_size, GxB_IS_READONLY, NULL)) ;
            OK (GxB_Vector_load (Container->i, (void **) &Xi, GrB_UINT64,
                nvals, Xi_size, GxB_IS_READONLY, NULL)) ;
            Container->format = GxB_SPARSE ;
        }
        else
        { 
            // import a full matrix
            Container->format = GxB_FULL ;
        }

        OK (GxB_Vector_load (Container->x, (void **) &Xx, Ax_type, nvals,
            Xx_size, GxB_IS_READONLY, NULL)) ;

        OK (GxB_load_Matrix_from_Container (A, Container, NULL)) ;
    }

    //--------------------------------------------------------------------------
    // restore the burble and return result
    //--------------------------------------------------------------------------

    OK (GrB_Global_set_INT32 (GrB_GLOBAL, burble, GxB_BURBLE)) ;
    return (A) ;
}

