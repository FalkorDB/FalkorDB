//------------------------------------------------------------------------------
// gbmx_get_grb_matrix: get content for a shallow GrB matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2026, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// For v4, iso is false, and the s component has length 9.
// For v5, iso is present but false, and the s component has length 10.
// For v5_1, iso is true/false, and the s component has length 10.
// For v7_3: the same content as v5_1, except that Yp, Yi, and Yx are added.
// For v10: Ap, Ah, Ai, Yp, Yi, and Yx can be 32-bit or 64-bit

// mxGetData is used instead of the MATLAB-recommended mxGetDoubles, etc,
// because mxGetData works best for Octave, and it works fine for MATLAB
// since GraphBLAS requires R2018a with the interleaved complex data type.

#define IF(error,message) \
    CHECK_ERROR (error, "invalid GraphBLAS struct (" message ")" ) ;

void gbmx_get_grb_matrix
(
    // output
    gb_matrix matrix,
    // input
    const mxArray *X        // a struct containing a GrB value matrix
)
{

    //--------------------------------------------------------------------------
    // get the content of the GrB matrix from the struct
    //--------------------------------------------------------------------------

    char err [ERRLEN] ;
    err [0] = '\0' ;
    memset (matrix, 0, sizeof (struct gb_matrix_struct)) ;
    CHECK_ERROR (X == NULL, "matrix is missing; internal error 899") ;

    if (mxIsClass (X, "GrB"))
    {
        // X is a GrB object; get its opaque content (which must be a struct).
        // mxGetProperty works here, but is insanely slow; it creates a copy of
        // the entire opaque GrB struct.  The MATLAB/Octave interface does not
        // rely on this in the tests, but it might occur in other uses.  The
        // user application might pass in a scalar to a GrB method that is
        // itself a GrB object, which works fine, and is reasonably fast since
        // a scalar is small.  This call to mxGetProperty is left here to
        // handle that case.
        //
        // In the *.m files in the MATLAB/Octave interface, this case is
        // avoided with statements such as these:
        //
        //      if (gb_is_grb (A))
        //          A = struct (A) ;
        //      end
        //
        // The above m-file code does not make a full copy of the A matrix.  It
        // just makes a shallow copy, so it is very fast.
        //
        // See also gbmx_get_ghb_handle, which also calls mxGetProperty.  That
        // usage is very fast since the entire GhB opaque struct contains a
        // single uint8 array of size 8, and making a copy of that is fast.
        X = mxGetProperty (X, 0, "opaque") ;
        #ifdef GBCOV
        // make sure it doesn't occur in the coverage tests
        mexErrMsgTxt ("gotcha! (GrB passed to a mexFunction as an object)") ;
        #endif
    }

    CHECK_ERROR (!mxIsStruct (X), "input matrix is mangled") ;

    bool GraphBLASv10 = false ;
    bool GraphBLASv4 = false ;
    bool GraphBLASv3 = false ;

    // get the type
    mxArray *mx_type = mxGetField (X, 0, "GraphBLASv10") ;
    GraphBLASv10 = (mx_type != NULL) ;

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
        GraphBLASv4 = (mx_type != NULL) ;
    }

    if (mx_type == NULL)
    { 
        // check if it is a GraphBLASv3 struct
        mx_type = mxGetField (X, 0, "GraphBLAS") ;
        GraphBLASv3 = (mx_type != NULL) ;
    }

    CHECK_ERROR (mx_type == NULL, "not a GraphBLAS struct") ;

    char typename [LEN+2] ;
    gbmx_mxstring_to_string (typename, LEN, mx_type, "type") ;
    GrB_Type Ax_type = gb_string_to_type (typename) ;
    size_t type_size ;
    OK (GxB_Type_size (&type_size, Ax_type)) ;

    // get the scalar info
    mxArray *opaque = mxGetField (X, 0, "s") ;
    IF (opaque == NULL, ".s missing") ;
    IF (mxGetM (opaque) != 1, ".s wrong size") ;
    size_t s_size = mxGetN (opaque) ;
    int64_t *s = (int64_t *) mxGetData (opaque) ;
    int64_t plen, vlen, vdim, nvec, nvec_nonempty, nzmax ;
    bool by_col ;
    if (GraphBLASv3 && s_size == 9)
    {
        // v3.1.1 had 9 items in s, starting with s [0] = hyper_ratio.
        // s was also saved as double.
        double *sdouble = (double *) mxGetData (opaque) ;
        plen          = (int64_t) sdouble [1] ;
        vlen          = (int64_t) sdouble [2] ;
        vdim          = (int64_t) sdouble [3] ;
        nvec          = (int64_t) sdouble [4] ;
        nvec_nonempty = (int64_t) sdouble [5] ;
        by_col        = (bool) (sdouble [7]) ;
        nzmax         = (int64_t) sdouble [8] ;
    }
    else
    {
        if (GraphBLASv3)
        {
            // v3.2.2 had 8 items in s, all int64
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
        plen          = s [0] ;
        vlen          = s [1] ;
        vdim          = s [2] ;
        nvec          = s [3] ;
        nvec_nonempty = s [4] ;
        by_col        = (bool) (s [6]) ;
        nzmax         = s [7] ;
    }

    int sparsity_status, sparsity_control ;
    int64_t nvals ;
    bool iso ;

    if (GraphBLASv3)
    { 
        // GraphBLASv3 struct: sparse or hypersparse only
        sparsity_control = GxB_AUTO_SPARSITY ;
        nvals = 0 ;
        iso = false ;
    }
    else
    { 
        // GraphBLASv4 or later struct: sparse, hypersparse, bitmap, or full
        sparsity_control = (int) (s [5]) ;
        nvals = s [8] ; // for bitmap case only, zero otherwise
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
            // C is full, with 3 fields: GraphBLAS*, s, x
            sparsity_status = GxB_FULL ;
            break ;

        case 5 : 
            // C is sparse, with 5 fields: GraphBLAS*, s, x, p, i
            sparsity_status = GxB_SPARSE ;
            break ;

        case 6 : 
        case 9 : 
            // C is hypersparse, with 6 fields: GraphBLAS*, s, x, p, i, h
            // or with 9 fields: Yp, Yi, and Yx added.
            sparsity_status = GxB_HYPERSPARSE ;
            // GraphBLAS v9 and earlier can export a matrix to the MATLAB
            // struct with plen of 1 but nvec of 0.  Fix it here.
            plen = nvec ;
            break ;

        case 4 : 
            // C is bitmap, with 4 fields: GraphBLAS*, s, x, b
            sparsity_status = GxB_BITMAP ;
            break ;

        default : ERROR ("invalid GraphBLAS struct", GrB_INVALID_VALUE) ;
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
        // C is hypersparse or sparse

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
        IF ((int64_t) mxGetN (Ap_mx) < plen+1, ".p wrong size")

        if (!GraphBLASv10)
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
        IF ((int64_t) mxGetN (Ai_mx) < nvals, ".i wrong size") ;
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
        // C is sparse; determine Aj_is_32
        Aj_is_32 = (vdim <= ((int64_t) (1ULL << 31))) ;
        Aj_type = Aj_is_32 ? GrB_UINT32 : GrB_UINT64 ;
        jsize = Aj_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;

    }
    else if (sparsity_status == GxB_HYPERSPARSE)
    { 
        // C is hypersparse
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
            mxClassID Yp_class = mxGetClassID (Yp_mx) ;
            IF (!(Yp_class == mxUINT64_CLASS || Yp_class == mxUINT32_CLASS ||
                  Yp_class == mxINT64_CLASS), ".Yp wrong class")
            bool Yp_is_32 = (Yp_class == mxUINT32_CLASS) ;
            IF (Yp_is_32 != Aj_is_32, ".Yp wrong class 32/64") ;
            Yp_len = mxGetN (Yp_mx) ;
            Yp_size = Yp_len * jsize ;
            Yp = (Yp_size == 0) ? NULL : ((void *) mxGetData (Yp_mx)) ;

            // Yi must be 1-by-nvec, with the same class as Ah
            mxArray *Yi_mx = mxGetField (X, 0, "Yi") ;
            IF (Yi_mx == NULL, ".Yi missing") ;
            IF (mxGetM (Yi_mx) != 1, ".Yi wrong size") ;
            IF (mxGetN (Yi_mx) != nvec, ".Yi wrong size") ;
            mxClassID Yi_class = mxGetClassID (Yi_mx) ;
            IF (!(Yi_class == mxUINT64_CLASS || Yi_class == mxUINT32_CLASS ||
                  Yi_class == mxINT64_CLASS), ".Yi wrong class")
            bool Yi_is_32 = (Yi_class == mxUINT32_CLASS) ;
            IF (Yi_is_32 != Aj_is_32, ".Yi wrong class 32/64") ;
            Yi_len = mxGetN (Yi_mx) ;
            Yi_size = Yi_len * jsize ;
            Yi = (Yi_size == 0) ? NULL : ((void *) mxGetData (Yi_mx)) ;

            // Yx must be 1-by-nvec
            mxArray *Yx_mx = mxGetField (X, 0, "Yx") ;
            IF (Yx_mx == NULL, ".Yx missing") ;
            IF (mxGetM (Yx_mx) != 1, ".Yx wrong size") ;
            IF (mxGetN (Yx_mx) != nvec, ".Yx wrong size") ;
            mxClassID Yx_class = mxGetClassID (Yx_mx) ;
            IF (!(Yx_class == mxUINT64_CLASS || Yx_class == mxUINT32_CLASS ||
                  Yx_class == mxINT64_CLASS), ".Yx wrong class")
            bool Yx_is_32 = (Yx_class == mxUINT32_CLASS) ;
            IF (Yx_is_32 != Aj_is_32, ".Yx wrong class 32/64") ;
            Yx_len = mxGetN (Yx_mx) ;
            Yx_size = Yi_len * jsize ;
            Yx = (Yx_size == 0) ? NULL : ((void *) mxGetData (Yx_mx)) ;
        }
    }

    if (sparsity_status == GxB_BITMAP)
    { 
        // C is bitmap
        // get the bitmap
        mxArray *Ab_mx = mxGetField (X, 0, "b") ;
        IF (Ab_mx == NULL, ".b missing") ;
        IF (mxGetM (Ab_mx) != 1, ".b wrong size") ;
        IF (mxGetClassID (Ab_mx) != mxINT8_CLASS, ".Ab wrong class") ;
        Ab_len = mxGetN (Ab_mx) ;
        Ab_size = Ab_len ;
        Ab = (Ab_size == 0) ? NULL : ((int8_t *) mxGetData (Ab_mx)) ;
    }

    //--------------------------------------------------------------------------
    // load the results into the matrix struct
    //--------------------------------------------------------------------------

    matrix->nvals = nvals ;
    matrix->type = Ax_type ;
    matrix->nrows = (by_col) ? vlen : vdim ;
    matrix->ncols = (by_col) ? vdim : vlen ;
    matrix->typesize = type_size ;

    matrix->p = Ap ;
    matrix->h = Ah ;
    matrix->b = Ab ;
    matrix->i = Ai ;
    matrix->x = Ax ;

    matrix->Yp = Yp ;
    matrix->Yi = Yi ;
    matrix->Yx = Yx ;

    matrix->plen = plen ;
    matrix->nvec = nvec ;
    matrix->nvec_nonempty = nvec_nonempty ;
    matrix->ynrows = vdim ;
    matrix->yncols = yvdim ;

    matrix->sparsity = sparsity_status ;

    matrix->by_col = by_col ;
    matrix->p_is_32 = Ap_is_32 ;
    matrix->j_is_32 = Aj_is_32 ;
    matrix->i_is_32 = Ai_is_32 ;

    matrix->iso = iso ;

    matrix->is_empty = false ;
    matrix->will_wait = false ;
    matrix->kind = KIND_GRB ;   // matrix holds a GrB value matrix
}

#undef IF

