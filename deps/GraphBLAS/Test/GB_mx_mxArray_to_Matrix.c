//------------------------------------------------------------------------------
// GB_mx_mxArray_to_Matrix: convert a MATLAB matrix or struct to a GrB_Matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// Convert a MATLAB sparse or full matrix, or a struct with the contents of a
// GraphBLAS matrix, to a GrB_Matrix.  The mxArray is either a struct
// containing two terms: a sparse or full matrix or vector, and type (a string,
// "logical", "double", etc), or it is just a plain sparse or full matrix.  If
// A.class is present, it is used to typecast the MATLAB matrix into the
// corresponding type in GraphBLAS.

// That is: A = sparse (...) ;   % a sparse double or logical GraphBLAS matrix

// A.matrix = A ; A.class = 'int8' ; Represents a MATLAB sparse or full matrix
// that represents a GraphBLAS int8 matrix.  On input, the MATLAB sparse or
// full matrix is typecasted.

// The MATLAB matrix or struct is not modified.  If deep_copy is true, the
// GraphBLAS matrix is always a deep copy and can be modified by GraphBLAS.
// Otherwise, its pattern (A->p, A->h, and A->i) may be a shallow copy, and
// A->x is a shallow copy if the MATLAB matrix is 'logical' or 'double'.

// If the MATLAB matrix is double complex, it becomes a GraphBLAS Complex or
// GxB_FC64 matrix.

// A->x is always a deep copy for other types, since it must be typecasted from
// a MATLAB matrix to GraphBLAS.

// Like GB_mx_Matrix_to_mxArray, this could be done using only user-callable
// GraphBLAS functions, but the method used here is faster.

// A.sparsity sets the GxB_SPARSITY_CONTROL option: 0 to 15 (see GB_conform.c),
// which is any sum of these 4 flags:
//
//    // GxB_SPARSITY_CONTROL can be any sum or bitwise OR of these 4 values:
//    #define GxB_HYPERSPARSE 1   // hypersparse form
//    #define GxB_SPARSE      2   // sparse form
//    #define GxB_BITMAP      4   // a bitmap
//    #define GxB_FULL        8   // full (all entries must be present)

#include "GB_mex.h"

#define FREE_ALL            \
{                           \
    GrB_Matrix_free_(&A) ;  \
    GrB_Matrix_free_(&T) ;  \
}

GrB_Matrix GB_mx_mxArray_to_Matrix     // returns GraphBLAS version of A
(
    const mxArray *A_matlab,           // MATLAB version of A
    const char *name,                   // name of the argument
    bool deep_copy,                     // if true, return a deep copy
    const bool empty    // if false, 0-by-0 matrices are returned as NULL.
                        // if true, a 0-by-0 matrix is returned.
)
{

    //--------------------------------------------------------------------------
    // check for empty matrix
    //--------------------------------------------------------------------------

    GrB_Matrix A = NULL ;
    struct GB_Matrix_opaque T_header ;
    GrB_Matrix T = NULL ;

    if (A_matlab == NULL)
    {
        // input is not present; this is not an error if A is an
        // optional input
        return (NULL) ;
    }

    if ((mxGetM (A_matlab) == 0) && (mxGetN (A_matlab) == 0))
    {
        // input is "[ ]", zero-by-zero.
        if (empty)
        {
            // treat as a sparse 0-by-0 matrix, not NULL
            GrB_Matrix_new (&A, GrB_FP64, 0, 0) ;
            ASSERT_MATRIX_OK (A, "got A = [ ] from MATLAB", GB0) ;
            return (A) ;
        }
        else
        {
            // Treat as NULL in GraphBLAS.  Useful for mask matrices
            return (NULL) ;
        }
    }

    //--------------------------------------------------------------------------
    // get the matrix
    //--------------------------------------------------------------------------

    bool A_iso = false ;
    bool A_no_hyper_hash = false ;
    const mxArray *Amatrix = NULL ;
    GrB_Type atype_in, atype_out ;
    GB_Type_code atype_in_code, atype_out_code ;

    if (mxIsStruct (A_matlab))
    {
        // look for A.matrix
        int fieldnumber = mxGetFieldNumber (A_matlab, "matrix") ;
        if (fieldnumber >= 0)
        {
            Amatrix = mxGetFieldByNumber (A_matlab, 0, fieldnumber) ;
        }
        else
        {
            // A.matrix not present, try A.vector
            fieldnumber = mxGetFieldNumber (A_matlab, "vector") ;
            if (fieldnumber < 0)
            {
                FREE_ALL ;
                mexWarnMsgIdAndTxt ("GB:warn", "invalid matrix/vector struct") ;
                return (NULL) ;
            }
            Amatrix = mxGetFieldByNumber (A_matlab, 0, fieldnumber) ;
            if (mxGetN (Amatrix) != 1)
            {
                FREE_ALL ;
                mexWarnMsgIdAndTxt ("GB:warn", "vector must be n-by-1") ;
                return (NULL) ;
            }
        }

        // get the type
        ASSERT (Amatrix != NULL) ;

        atype_in = GB_mx_Type (Amatrix) ;
        atype_out = atype_in ;
        fieldnumber = mxGetFieldNumber (A_matlab, "class") ;
        if (fieldnumber >= 0)
        {
            mxArray *s = mxGetFieldByNumber (A_matlab, 0, fieldnumber) ;
            atype_out = GB_mx_string_to_Type (s, atype_in) ;
            if (atype_out == NULL) mexErrMsgTxt ("unknown class") ;
        }

        // get the iso property (false if not present)
        fieldnumber = mxGetFieldNumber (A_matlab, "iso") ;
        if (fieldnumber >= 0)
        {
            mxArray *s = mxGetFieldByNumber (A_matlab, 0, fieldnumber) ;
            if (mxIsLogicalScalar (s))
            {
                A_iso = mxIsLogicalScalarTrue (s) ;
            }
            else
            {
                A_iso = (mxGetScalar (s) != 0) ;
            }
        }

        // get the no_hyper_hash property (false if not present)
        fieldnumber = mxGetFieldNumber (A_matlab, "no_hyper_hash") ;
        if (fieldnumber >= 0)
        {
            mxArray *s = mxGetFieldByNumber (A_matlab, 0, fieldnumber) ;
            if (mxIsLogicalScalar (s))
            {
                A_no_hyper_hash = mxIsLogicalScalarTrue (s) ;
            }
            else
            {
                A_no_hyper_hash = (mxGetScalar (s) != 0) ;
            }
        }

    }
    else
    {
        // just a matrix
        Amatrix = A_matlab ;
        atype_in = GB_mx_Type (Amatrix) ;
        atype_out = atype_in ;
    }

    bool A_is_sparse = mxIsSparse (Amatrix) ;

    //--------------------------------------------------------------------------
    // get the matrix type
    //--------------------------------------------------------------------------

    atype_in_code  = atype_in->code ;
    atype_out_code = atype_out->code ;

    //--------------------------------------------------------------------------
    // get the size and content of the MATLAB matrix
    //--------------------------------------------------------------------------

    int64_t nrows = mxGetM (Amatrix) ;
    int64_t ncols = mxGetN (Amatrix) ;
    uint64_t *Mp ;
    int64_t *Mi, anz, anzmax ;

    if (A_is_sparse)
    {
        Mp = (uint64_t *) mxGetJc (Amatrix) ;
        Mi = (int64_t *) mxGetIr (Amatrix) ;
        anz = Mp [ncols] ;
        anzmax = mxGetNzmax (Amatrix) ;
    }
    else
    {
        Mp = NULL ;
        Mi = NULL ;
        anz = nrows * ncols ;
        anzmax = anz ;
    }

    GB_void *MatlabX = mxGetData (Amatrix) ;
    if (anz == 0)
    {
        // an empty matrix cannot be iso
        A_iso = false ;
    }

    //--------------------------------------------------------------------------
    // look for A.values
    //--------------------------------------------------------------------------

    if (mxIsStruct (A_matlab))
    {
        // this is used for int64 and uint64 only
        int fieldnumber = mxGetFieldNumber (A_matlab, "values") ;
        if (fieldnumber >= 0)
        {
            mxArray *values = mxGetFieldByNumber (A_matlab, 0, fieldnumber) ;
            if (mxIsComplex (values))
            {
                mexErrMsgTxt ("A.values must be real") ;
            }
            if (mxGetNumberOfElements (values) >= anz)
            {
                MatlabX = mxGetData (values) ;
                atype_in = GB_mx_Type (values) ;
                atype_in_code = atype_in->code ;
                anzmax = mxGetNumberOfElements (values) ;
            }
        }
    }

    ASSERT_TYPE_OK (atype_in,  "A type in", GB0) ;
    ASSERT_TYPE_OK (atype_out, "A type out", GB0) ;

    if (atype_in == NULL || atype_out == NULL)
    {
        FREE_ALL ;
        mexWarnMsgIdAndTxt ("GB:warn", "types must be numeric") ;
        return (NULL) ;
    }

    GrB_Info info ;

    // MATLAB matrices are sparse or full CSC, not hypersparse or bitmap
    bool is_csc = true ;
    int sparsity = (A_is_sparse) ? GxB_SPARSE : GxB_FULL ;

    //--------------------------------------------------------------------------
    // get the integer sizes
    //--------------------------------------------------------------------------

    // MATLAB matrices are all-64-bit
    bool p_is_32 = false ;
    bool j_is_32 = false ;
    bool i_is_32 = false ;

    if (mxIsStruct (A_matlab))
    {

        // look for A.p_is_32
        int fieldnumber = mxGetFieldNumber (A_matlab, "p_is_32") ;
        if (fieldnumber >= 0)
        {
            p_is_32 = mxGetScalar (mxGetFieldByNumber (A_matlab,
                0, fieldnumber)) ;
        }

        // look for A.j_is_32
        fieldnumber = mxGetFieldNumber (A_matlab, "j_is_32") ;
        if (fieldnumber >= 0)
        {
            j_is_32 = mxGetScalar (mxGetFieldByNumber (A_matlab,
                0, fieldnumber)) ;
        }

        // look for A.i_is_32
        fieldnumber = mxGetFieldNumber (A_matlab, "i_is_32") ;
        if (fieldnumber >= 0)
        {
            i_is_32 = mxGetScalar (mxGetFieldByNumber (A_matlab,
                0, fieldnumber)) ;
        }
    }

    // can only do a shallow copy if the resulting GrB_Matrix is all-64
    deep_copy = deep_copy || p_is_32 || j_is_32 || i_is_32 ;

    GB_Type_code apcode = (p_is_32) ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code aicode = (i_is_32) ? GB_UINT32_code : GB_UINT64_code ;

    //--------------------------------------------------------------------------
    // get the pattern of A
    //--------------------------------------------------------------------------

    if (deep_copy)
    {

        // create the GraphBLAS matrix
        info = GB_new (&A, // sparse or full, new header
            atype_out, (uint64_t) nrows, (uint64_t) ncols,
            GB_ph_calloc, is_csc, sparsity, GxB_HYPER_DEFAULT, 0,
            p_is_32, j_is_32, i_is_32) ;
        if (info != GrB_SUCCESS)
        {
            FREE_ALL ;
            mexWarnMsgIdAndTxt ("GB:warn", "new deep matrix failed") ;
            return (NULL) ;
        }

        // A is a deep copy and can be modified by GraphBLAS
        info = GB_bix_alloc (A, anz, sparsity, false, true, false) ;
        if (info != GrB_SUCCESS)
        {
            FREE_ALL ;
            mexWarnMsgIdAndTxt ("GB:warn", "out of memory") ;
            return (NULL) ;
        }

        if (sparsity != GxB_FULL)
        {
//          memcpy (A->p, Mp, (ncols+1) * sizeof (int64_t)) ;
            GB_cast_int (A->p, apcode, Mp, GB_UINT64_code, ncols+1, 1) ;
//          memcpy (A->i, Mi, anz * sizeof (int64_t)) ;
            GB_cast_int (A->i, aicode, Mi, GB_UINT64_code, anz, 1) ;
        }

    }
    else
    {

        // the GraphBLAS pattern (A->p and A->i) are pointers into the
        // MATLAB sparse mxArray, and must not be modified.

        // create the GraphBLAS matrix, do not allocate A->p
        info = GB_new (&A, // sparse or full, new header
            atype_out, (uint64_t) nrows, (uint64_t) ncols,
            GB_ph_null, is_csc, sparsity, GxB_HYPER_DEFAULT, 0,
            /* must be false (MATLAB matrices are 64/64 bit): */
            false, false, false) ;
        if (info != GrB_SUCCESS)
        {
            FREE_ALL ;
            mexWarnMsgIdAndTxt ("GB:warn", "new shallow matrix failed") ;
            return (NULL) ;
        }

        if (sparsity != GxB_FULL)
        {
            A->p = Mp ;
            A->i = Mi ;
            A->p_size = (ncols+1) * sizeof (int64_t) ;
            A->i_size = GB_IMAX (anz, 1) * sizeof (int64_t) ;
            A->p_shallow = true ;
            A->i_shallow = true ;
        }
        else
        {
            A->p = NULL ;
            A->i = NULL ;
            A->p_size = 0 ;
            A->i_size = 0 ;
            A->p_shallow = false ;
            A->i_shallow = false ;
        }

        A->h_shallow = false ;      // A->h is NULL
    }

    A->nvals = anz ;
    A->magic = GB_MAGIC ;

    //--------------------------------------------------------------------------
    // copy the numerical values from MATLAB to the GraphBLAS matrix
    //--------------------------------------------------------------------------

    if (sparsity == GxB_FULL)
    {
        A->x_shallow = (!deep_copy && (atype_out_code == atype_in_code)) ;
    }
    else
    {
        A->x_shallow = (!deep_copy &&
               ((atype_out_code == GB_BOOL_code ||
                 atype_out_code == GB_FP64_code ||
                 atype_out_code == GB_FC64_code)
             && (atype_out_code == atype_in_code))) ;
    }

    if (A->x_shallow)
    {
        // the MATLAB matrix and GraphBLAS matrix have the same type;
        // (logical, double, or double complex), and a deep copy is not
        // requested.  Just make a shallow copy.
        A->x = MatlabX ;
        A->x_size = anzmax * atype_out->size ;
    }
    else
    {
        if (!deep_copy)
        {
            // allocate new space for the GraphBLAS values
            A->x = (GB_void *) GB_malloc_memory (anz * atype_out->size,
                sizeof (GB_void), &(A->x_size)) ;
            if (A->x == NULL)
            {
                FREE_ALL ;
                mexWarnMsgIdAndTxt ("GB:warn", "out of memory") ;
                return (NULL) ;
            }
        }

        if (A->x == NULL && anz > 0)
        {
            mexErrMsgTxt ("A->x is NULL!\n") ;
        }

        GB_Type_code code1 =
            (atype_out_code == GB_UDT_code) ? GB_FC64_code : atype_out_code ;

        GB_Type_code code2 =
            (atype_in_code == GB_UDT_code) ? GB_FC64_code : atype_in_code ;

        size_t asize =
            (atype_in_code == GB_UDT_code) ? sizeof(GxB_FC64_t) :atype_in->size;

        if (code1 == code2)
        {
            memcpy (A->x, MatlabX, anz * asize) ;
        }
        else
        {
            // create a shallow cnz-by-1 matrix T to wrap the array MatlabX
            T = NULL ;
            GrB_Type ttype = (atype_in_code == GB_UDT_code) ?
                GxB_FC64 : atype_in ;
            void *Tx = MatlabX ;
            uint64_t nrows = anz, ncols = 1, Tx_size = anz * asize ;
            GxB_Matrix_import_FullC (&T, ttype, nrows, ncols, &Tx, Tx_size,
                false, NULL) ;
            GB_cast_array (A->x, code1, T, 1) ;
            // GB_cast_array (A->x, code1, MatlabX, code2, NULL, anz, 1) ;
            bool iso ;
            GxB_Matrix_export_FullC (&T, &ttype, &nrows, &ncols, &Tx,
                &Tx_size, &iso, NULL) ;
        }
    }

    //--------------------------------------------------------------------------
    // compute the # of non-empty vectors in A only when needed
    //--------------------------------------------------------------------------

    if (sparsity != GxB_FULL)
    {
        A->nvec_nonempty = -1 ;
    }

    //--------------------------------------------------------------------------
    // set the iso property
    //--------------------------------------------------------------------------

    if (A_iso)
    {
        if (!A->x_shallow)
        {
            // convert A to iso, and reduce the size of A->x to a single entry
            if (A->x_size >= atype_out->size)
            {
                // use the first entry of A->x as the iso value of A
                A->iso = true ;
            }
            else
            {
                // A is converted to iso, but it doesn't have enough space in
                // A->x for the iso value, so set it to zero
                A->iso = false ;
            }
            GB_convert_any_to_iso (A, NULL) ;       // OK
        }
        else
        {
            // just set the iso flag, leave A->x unmodified.  A can be iso
            // only if A->x is large enough to hold at least 1 entry.
            A->iso = (A->x_size >= atype_out->size) ;
        }

        ASSERT_MATRIX_OK (A, "got iso A from MATLAB", GB0) ;
    }

    ASSERT_MATRIX_OK (A, "got natural A from MATLAB", GB0) ;
    ASSERT (A->h == NULL) ;

    //--------------------------------------------------------------------------
    // set A->no_hyper_hash
    //--------------------------------------------------------------------------

    if (A_no_hyper_hash)
    {
        A->no_hyper_hash = A_no_hyper_hash ;
    }

    //--------------------------------------------------------------------------
    // look for CSR/CSC and hyper/non-hyper format
    //--------------------------------------------------------------------------

    bool A_is_hyper = false ;
    bool has_hyper_switch = false ;
    bool has_sparsity_control = false ;
    int sparsity_control = GxB_AUTO_SPARSITY ;
    double hyper_switch = GxB_HYPER_DEFAULT ;

    if (mxIsStruct (A_matlab))
    {
        // look for A.is_csc
        int fieldnumber = mxGetFieldNumber (A_matlab, "is_csc") ;
        if (fieldnumber >= 0)
        {
            is_csc = mxGetScalar (mxGetFieldByNumber (A_matlab,
                0, fieldnumber)) ;
        }

        // look for A.is_hyper (ignored if hyper_switch present
        // or if A is full)
        fieldnumber = mxGetFieldNumber (A_matlab, "is_hyper") ;
        if (fieldnumber >= 0)
        {
            A_is_hyper = mxGetScalar (mxGetFieldByNumber (A_matlab,
                0, fieldnumber)) ;
        }

        // look for A.hyper_switch (ignored if A is full)
        fieldnumber = mxGetFieldNumber (A_matlab, "hyper_switch") ;
        if (fieldnumber >= 0)
        {
            has_hyper_switch = true ;
            hyper_switch = mxGetScalar (mxGetFieldByNumber (A_matlab,
                0, fieldnumber)) ;
        }

        // look for A.sparsity
        fieldnumber = mxGetFieldNumber (A_matlab, "sparsity") ;
        if (fieldnumber >= 0)
        {
            has_sparsity_control = true ;
            sparsity_control = mxGetScalar (mxGetFieldByNumber (A_matlab,
                0, fieldnumber)) ;
//          printf ("sparsity control is %d\n", sparsity_control) ;
        }
    }

    //--------------------------------------------------------------------------
    // convert to CSR if requested
    //--------------------------------------------------------------------------

    int64_t nrows_old = GB_NROWS (A) ;
    int64_t ncols_old = GB_NCOLS (A) ;

    #define ENSURE_DEEP             \
    if (GB_is_shallow (A))          \
    {                               \
        GrB_Matrix_dup (&T, A) ;    \
        GrB_Matrix_free (&A) ;      \
        A = T ;                     \
        T = NULL ;                  \
    }

    if (!is_csc)
    {
        // this might convert A to hypersparse
        ENSURE_DEEP ;
        GxB_Matrix_Option_set_(A, GxB_FORMAT, GxB_BY_ROW) ;
        // so convert it back; hypersparsity is defined below
        if (sparsity != GxB_FULL)
        {
            bool burble = GB_Global_burble_get ( ) ;
            if (burble) printf (" [ GB_mx_mxArray_to_Matrix ") ;
            GB_convert_hyper_to_sparse (A, true) ;
            if (burble) printf ("]\n") ;
        }
        ASSERT (!A->is_csc) ;
    }

    ASSERT_MATRIX_OK (A, "conformed from MATLAB", GB0) ;
    ASSERT (A->h == NULL) ;
    ASSERT (A->is_csc == is_csc) ;

    //--------------------------------------------------------------------------
    // convert to hypersparse or set hypersparse ratio, if requested
    //--------------------------------------------------------------------------

    if (sparsity == GxB_FULL)
    {
        // leave as-is
        ;
    }
    else if (has_hyper_switch)
    {
        // this sets the hyper_switch and then conforms the matrix to its
        // desired hypersparsity.  It may stay non-hypersparse.
        ENSURE_DEEP ;
        GxB_Matrix_Option_set_(A, GxB_HYPER_SWITCH, hyper_switch) ;
    }
    else if (A_is_hyper)
    {
        // this forces the matrix to be always hypersparse
        ENSURE_DEEP ;
        ASSERT_MATRIX_OK (A, "to always hyper", GB0) ;
        GxB_Matrix_Option_set_(A, GxB_SPARSITY_CONTROL, GxB_HYPERSPARSE) ;
        ASSERT_MATRIX_OK (A, "always hyper", GB0) ;
    }

    //--------------------------------------------------------------------------
    // set the sparsity control and conform the matrix
    //--------------------------------------------------------------------------

    if (has_sparsity_control)
    {
        ASSERT_MATRIX_OK (A, "setting sparsity", GB0) ;
        ENSURE_DEEP ;
        GxB_Matrix_Option_set_(A, GxB_SPARSITY_CONTROL, sparsity_control) ;
        ASSERT_MATRIX_OK (A, "set sparsity", GB0) ;
    }

    ASSERT (A->is_csc == is_csc) ;
    ASSERT (nrows_old == GB_NROWS (A)) ;
    ASSERT (ncols_old == GB_NCOLS (A)) ;

    //--------------------------------------------------------------------------
    // return the GraphBLAS matrix
    //--------------------------------------------------------------------------

    info = GrB_Matrix_wait (A, GrB_MATERIALIZE) ;
    if (info != GrB_SUCCESS)
    {
        FREE_ALL ;
        mexWarnMsgIdAndTxt ("GB:warn", "matrix wait failed") ;
        return (NULL) ;
    }

    ASSERT_MATRIX_OK (A, "got A from MATLAB", GB0) ;
    return (A) ;
}

