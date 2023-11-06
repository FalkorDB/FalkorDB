//------------------------------------------------------------------------------
// GB_mx_build_template: build a sparse vector or matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// This is not a stand-alone function; it is #include'd in
// GB_mex_Matrix_build.c and GB_mex_Vector_build.c.

// The function works the same as C = sparse (I,J,X,nrows,ncols)
// except an optional operator, dup, can be provided.  The operator defines
// how duplicates are assembled.

// The type of dup and C may differ.  A matrix T is first built that has the
// same type as dup, and then typecasted to the desired type of C, given by
// the last argument, "type".

// This particular GraphBLAS implementation provides a well-defined order of
// 'summation'.  Entries in [I,J,X] are first sorted in increasing order of row
// index via a stable sort, with ties broken by the position of the tuple in
// the [I,J,X] list.  If duplicates appear, they are 'summed' in the order they
// appear in the [I,J,X] input.  That is, if the same indices i and j appear in
// positions k1, k2, k3, and k3 in [I,J,X], where k1 < k2 < k3 < k4, then the
// following operations will occur in order:

//      T (i,j) = X (k1) ;
//      T (i,j) = dup (T (i,j), X (k2)) ;
//      T (i,j) = dup (T (i,j), X (k3)) ;
//      T (i,j) = dup (T (i,j), X (k4)) ;

// Thus, dup need not be associative, and the results are still well-defined.
// Using the FIRST operator (a string 'first') means the first value (X (k1))
// is used and the rest are ignored.  The SECOND operator means the last value
// (X (k4)) is used instead.

// X must be the same length as I, in which case GrB_Matrix_build_[TYPE] or
// GrB_Vector_build_[TYPE] is used, or X must have length 1, in which case
// GxB_Matrix_build_Scalar or GxB_Vector_build_Scalar is used.  The dup
// operator is ignored in that case.

#ifdef MATRIX
#define MAX_NARGIN 8
#define MIN_NARGIN 3
#define USAGE "GB_mex_Matrix_build (I,J,X,nrows,ncols,dup,type,csc)"
#define I_ARG 0
#define J_ARG 1
#define X_ARG 2
#define NROWS_ARG 3
#define NCOLS_ARG 4
#define DUP_ARG 5
#define TYPE_ARG 6
#define CSC_ARG 7
#define FREE_WORK                   \
{                                   \
    GrB_Scalar_free_(&scalar) ;     \
    GrB_Matrix_free_(&C) ;          \
}
#else
#define MAX_NARGIN 5
#define MIN_NARGIN 2
#define USAGE "GB_mex_Vector_build (I,X,nrows,dup,type)"
#define I_ARG 0
#define X_ARG 1
#define NROWS_ARG 2
#define DUP_ARG 3
#define TYPE_ARG 4
#define FREE_WORK                   \
{                                   \
    GrB_Scalar_free_(&scalar) ;     \
    GrB_Vector_free_(&C) ;          \
}
#endif

#define FREE_ALL                    \
{                                   \
    FREE_WORK ;                     \
    GB_mx_put_global (true) ;       \
}

#define OK1(method)                 \
{                                   \
    info = method ;                 \
    if (info != GrB_SUCCESS)        \
    {                               \
        FREE_WORK ;                 \
        return (info) ;             \
    }                               \
}

#define GET_DEEP_COPY ;
#define FREE_DEEP_COPY ;

bool malloc_debug = false ;

GrB_Info builder
(
    #ifdef MATRIX
    GrB_Matrix *Chandle,
    #else
    GrB_Vector *Chandle,
    #endif
    GrB_Type ctype,
    GrB_Index nrows,
    GrB_Index ncols,
    GrB_Index *I,
    GrB_Index *J,
    GB_void *X, bool scalar_build,
    GrB_Index ni,
    GrB_BinaryOp dup,
    bool C_is_csc,
    GrB_Type xtype
) ;

//------------------------------------------------------------------------------

GrB_Info builder
(
    #ifdef MATRIX
    GrB_Matrix *Chandle,
    #else
    GrB_Vector *Chandle,
    #endif
    GrB_Type ctype,
    GrB_Index nrows,
    GrB_Index ncols,
    GrB_Index *I,
    GrB_Index *J,
    GB_void *X, bool scalar_build,
    GrB_Index ni,
    GrB_BinaryOp dup,
    bool C_is_csc,
    GrB_Type xtype
)
{

    GrB_Info info ;
    GrB_Scalar scalar = NULL ;
    (*Chandle) = NULL ;

    // create the GraphBLAS output object C
    int sparsity = GxB_SPARSE + GxB_HYPERSPARSE ;
    #ifdef MATRIX
    if (C_is_csc)
    {
        // create a hypersparse CSC matrix
        info = GB_new (Chandle, // sparse/hyper, new header
            ctype, nrows, ncols, GB_Ap_calloc,
            true, sparsity, GxB_HYPER_DEFAULT, 1) ;
    }
    else
    {
        // create a hypersparse CSR matrix
        info = GB_new (Chandle, // sparse/hyper, new header
            ctype, ncols, nrows, GB_Ap_calloc,
            false, sparsity, GxB_HYPER_DEFAULT, 1) ;
    }
    #else
    info = GrB_Vector_new (Chandle, ctype, nrows) ;
    #endif

    #ifdef MATRIX
    GrB_Matrix C = (*Chandle) ;
    #else
    GrB_Vector C = (*Chandle) ;
    #endif

    OK1 (info) ;

    ASSERT_TYPE_OK (ctype, "ctype for build", GB0) ;
    ASSERT_BINARYOP_OK (dup, "dup for build", GB0) ;

    if (scalar_build)
    {

        OK1 (GrB_Scalar_new (&scalar, xtype)) ;

        // build an iso matrix or vector from the tuples and the scalar
        #ifdef MATRIX
        #define BUILD(prefix,suffix,type)                                   \
            OK1 (prefix ## Scalar_setElement ## suffix (scalar, * (type *) X)) ;  \
            OK1 (GxB_Matrix_build_Scalar (C, I, J, scalar, ni)) ;
        #else
        #define BUILD(prefix,suffix,type)                                   \
            OK1 (prefix ## Scalar_setElement ## suffix (scalar, * (type *) X)) ;  \
            OK1 (GxB_Vector_build_Scalar (C, I,    scalar, ni)) ;
        #endif

        switch (xtype->code)
        {
            case GB_BOOL_code    : BUILD (GrB_, _BOOL,   bool    ) ; break ;
            case GB_INT8_code    : BUILD (GrB_, _INT8,   int8_t  ) ; break ;
            case GB_INT16_code   : BUILD (GrB_, _INT16,  int16_t ) ; break ;
            case GB_INT32_code   : BUILD (GrB_, _INT32,  int32_t ) ; break ;
            case GB_INT64_code   : BUILD (GrB_, _INT64,  int64_t ) ; break ;
            case GB_UINT8_code   : BUILD (GrB_, _UINT8,  uint8_t ) ; break ;
            case GB_UINT16_code  : BUILD (GrB_, _UINT16, uint16_t) ; break ;
            case GB_UINT32_code  : BUILD (GrB_, _UINT32, uint32_t) ; break ;
            case GB_UINT64_code  : BUILD (GrB_, _UINT64, uint64_t) ; break ;
            case GB_FP32_code    : BUILD (GrB_, _FP32,   float   ) ; break ;
            case GB_FP64_code    : BUILD (GrB_, _FP64,   double  ) ; break ;
            case GB_FC32_code    : BUILD (GxB_, _FC32,   GxB_FC32_t) ; break ;
            case GB_FC64_code    : BUILD (GxB_, _FC64,   GxB_FC64_t) ; break ;
            default              :
                FREE_WORK ;
                mexErrMsgTxt ("xtype not supported")  ;
        }

    }
    else
    {

        // build a non-iso matrix or vector from the tuples
        #undef BUILD
        #ifdef MATRIX
        #define BUILD(prefix,suffix,type)               \
            OK1 (prefix ## Matrix_build ## suffix       \
                (C, I, J, (const type *) X, ni, dup))
        #else
        #define BUILD(prefix,suffix,type)               \
            OK1 (prefix ## Vector_build ## suffix       \
                (C, I,    (const type *) X, ni, dup))
        #endif

        switch (xtype->code)
        {
            case GB_BOOL_code    : BUILD (GrB_, _BOOL,   bool    ) ; break ;
            case GB_INT8_code    : BUILD (GrB_, _INT8,   int8_t  ) ; break ;
            case GB_INT16_code   : BUILD (GrB_, _INT16,  int16_t ) ; break ;
            case GB_INT32_code   : BUILD (GrB_, _INT32,  int32_t ) ; break ;
            case GB_INT64_code   : BUILD (GrB_, _INT64,  int64_t ) ; break ;
            case GB_UINT8_code   : BUILD (GrB_, _UINT8,  uint8_t ) ; break ;
            case GB_UINT16_code  : BUILD (GrB_, _UINT16, uint16_t) ; break ;
            case GB_UINT32_code  : BUILD (GrB_, _UINT32, uint32_t) ; break ;
            case GB_UINT64_code  : BUILD (GrB_, _UINT64, uint64_t) ; break ;
            case GB_FP32_code    : BUILD (GrB_, _FP32,   float   ) ; break ;
            case GB_FP64_code    : BUILD (GrB_, _FP64,   double  ) ; break ;
            case GB_FC32_code    : BUILD (GxB_, _FC32,   GxB_FC32_t) ; break ;
            case GB_FC64_code    : BUILD (GxB_, _FC64,   GxB_FC64_t) ; break ;
            default              :
                FREE_WORK ;
                mexErrMsgTxt ("xtype not supported")  ;
        }
    }

    GrB_Scalar_free_(&scalar) ;
    return (GrB_SUCCESS) ;
}

//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    GrB_Info info ;
    malloc_debug = GB_mx_get_global (true) ;
    GrB_Index *I = NULL, ni = 0, I_range [3] ;
    GrB_Index *J = NULL, nj = 0, J_range [3] ;
    GrB_Scalar scalar = NULL ;
    bool is_list ; 
    #ifdef MATRIX
    GrB_Matrix C = NULL ;
    #else
    GrB_Vector C = NULL ;
    #endif

    // check inputs
    if (nargout > 1 || nargin < MIN_NARGIN || nargin > MAX_NARGIN)
    {
        mexErrMsgTxt ("Usage: C = " USAGE) ;
    }

    // get I
    if (!GB_mx_mxArray_to_indices (&I, pargin [I_ARG], &ni, I_range, &is_list))
    {
        FREE_ALL ;
        mexErrMsgTxt ("I failed") ;
    }
    if (!is_list)
    {
        mexErrMsgTxt ("I is invalid; must be a list") ;
    }

    #ifdef MATRIX
    // get J for a matrix
    if (!GB_mx_mxArray_to_indices (&J, pargin [J_ARG], &nj, J_range, &is_list))
    {
        FREE_ALL ;
        mexErrMsgTxt ("J failed") ;
    }
    if (!is_list)
    {
        mexErrMsgTxt ("J is invalid; must be a list") ;
    }

    if (ni != nj)
    {
        FREE_ALL ;
        mexErrMsgTxt ("I and J must be the same size") ;
    }
    #endif

    // get X
    GrB_Index nx = mxGetNumberOfElements (pargin [X_ARG]) ;
    bool scalar_build = (nx == 1) ;
    if (!scalar_build && ni != nx)
    {
        FREE_ALL ;
        mexErrMsgTxt ("I and X must be the same size") ;
    }
    if (!(mxIsNumeric (pargin [X_ARG]) || mxIsLogical (pargin [X_ARG])))
    {
        FREE_ALL ;
        mexErrMsgTxt ("X must be a numeric or logical array") ;
    }
    if (mxIsSparse (pargin [X_ARG]))
    {
        FREE_ALL ;
        mexErrMsgTxt ("X cannot be sparse") ;
    }
    GB_void *X = mxGetData (pargin [X_ARG]) ;
    GrB_Type xtype = GB_mx_Type (pargin [X_ARG]) ;
    if (xtype == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("X must be numeric") ;
    }

    // get the number of rows
    uint64_t nrows = 0 ;
    if (nargin > NROWS_ARG)
    {
        nrows = (uint64_t) mxGetScalar (pargin [NROWS_ARG]) ;
    }
    else
    {
        for (int64_t k = 0 ; k < ni ; k++)
        {
            nrows = GB_IMAX (nrows, I [k]) ;
        }
        nrows++ ;
    }

    // get the number of columns of a matrix
    uint64_t ncols = 1 ;
    #ifdef MATRIX
    if (nargin > NCOLS_ARG)
    {
        ncols = (uint64_t) mxGetScalar (pargin [NCOLS_ARG]) ;
    }
    else
    {
        ncols = 0 ;
        for (int64_t k = 0 ; k < ni ; k++)
        {
            ncols = GB_IMAX (ncols, J [k]) ;
        }
        ncols++ ;
    }
    #endif

    // get dup; default is xtype
    bool user_complex = (Complex != GxB_FC64) && (xtype == Complex) ;
    GrB_BinaryOp dup ;
    if (!GB_mx_mxArray_to_BinaryOp (&dup, PARGIN (DUP_ARG), "dup",
        xtype, user_complex))
    {
        FREE_ALL ;
        mexErrMsgTxt ("dup failed") ;
    }

    // get the type for C, default is same as xtype
    GrB_Type ctype = GB_mx_string_to_Type (PARGIN (TYPE_ARG), xtype) ;

    if (dup == NULL && !scalar_build)
    {
        switch (xtype->code)
        {
            case GB_BOOL_code    : dup = GrB_PLUS_BOOL   ; break ;
            case GB_INT8_code    : dup = GrB_PLUS_INT8   ; break ;
            case GB_INT16_code   : dup = GrB_PLUS_INT16  ; break ;
            case GB_INT32_code   : dup = GrB_PLUS_INT32  ; break ;
            case GB_INT64_code   : dup = GrB_PLUS_INT64  ; break ;
            case GB_UINT8_code   : dup = GrB_PLUS_UINT8  ; break ;
            case GB_UINT16_code  : dup = GrB_PLUS_UINT16 ; break ;
            case GB_UINT32_code  : dup = GrB_PLUS_UINT32 ; break ;
            case GB_UINT64_code  : dup = GrB_PLUS_UINT64 ; break ;
            case GB_FP32_code    : dup = GrB_PLUS_FP32   ; break ;
            case GB_FP64_code    : dup = GrB_PLUS_FP64   ; break ;
            case GB_FC32_code    : dup = GxB_PLUS_FC32   ; break ;
            case GB_FC64_code    : dup = GxB_PLUS_FC64   ; break ;
            default              : 
                mexErrMsgTxt ("unknown operator") ;
        }
    }

    bool C_is_csc = true ;
    #ifdef MATRIX
    // get the CSC/CSR format
    if (nargin > CSC_ARG)
    {
        C_is_csc = (bool) mxGetScalar (pargin [CSC_ARG]) ;
    }
    #endif

    METHOD (builder (&C, ctype, nrows, ncols, I, J, X, scalar_build, ni, dup,
        C_is_csc, xtype)) ;

    ASSERT_MATRIX_OK (C, "C built", GB0) ;

    // return C as a struct and free the GraphBLAS C
    #ifdef MATRIX
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C output", true) ;
    #else
    pargout [0] = GB_mx_Vector_to_mxArray (&C, "C output", true) ;
    #endif

    FREE_ALL ;
}

