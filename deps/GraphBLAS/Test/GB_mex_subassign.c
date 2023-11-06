//------------------------------------------------------------------------------
// GB_mex_subassign: C(I,J)<M> = accum (C (I,J), A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// This function is a wrapper for all GxB_*_subassign functions.
// For these uses, the mask M must always be the same size as C(I,J) and A.

// GxB_Matrix_subassign: M has the same size as C(I,J) and A
// GxB_Matrix_subassign_TYPE: M has the same size as C(I,J).  A is scalar.

// GxB_Vector_subassign: M has the same size as C(I,J) and A
// GxB_Vector_subassign_TYPE: M has the same size as C(I,J).  A is scalar.

// GxB_Col_subassign: on input to GB_mex_subassign, M and A are a single
// columns, the same size as the single subcolumn C(I,j).  They are not column
// vectors.

// GxB_Row_subassign: on input to GB_mex_subassign, M and A are single ROWS,
// the same size as the single subrow C(i,J).  They are not column vectors.
// Before GxB_Row_subassign is called, the A and the mask M (if present) are
// transposed.

// Thus, in all cases, A and the mask M (if present) have the same size as
// C(I,J), except in the case where A is a scalar.  In that case, A is
// implicitly expanded into a matrix the same size as C(I,J), but this occurs
// inside GxB_*subassign, not here.

// This function does the same thing as the mimic GB_spec_subassign.m.

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "[C,s,t] = GB_mex_subassign " \
              "(C, M, accum, A, I, J, desc, reduce) or (C, Work, control)"

#define FREE_ALL                        \
{                                       \
    bool A_is_M = (A == M) ;            \
    bool A_is_C = (A == C) ;            \
    bool C_is_M = (C == M) ;            \
    GrB_Matrix_free_(&A) ;              \
    if (A_is_C) C = NULL ;              \
    if (A_is_M) M = NULL ;              \
    GrB_Matrix_free_(&C) ;              \
    if (C_is_M) M = NULL ;              \
    GrB_Matrix_free_(&M) ;              \
    GrB_Descriptor_free_(&desc) ;       \
    if (!user_complex) GrB_Monoid_free_(&reduce) ;                \
    GB_mx_put_global (true) ;           \
}

#define GET_DEEP_COPY                                                   \
{                                                                       \
    C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;   \
    if (have_sparsity_control)                                          \
    {                                                                   \
        GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, C_sparsity_control) ; \
    }                                                                   \
    if (nargin > 3 && mxIsChar (pargin [1]))                            \
    {                                                                   \
        M = GB_mx_alias ("M", pargin [1], "C", C, "A", A) ;             \
    }                                                                   \
    if (nargin > 3 && mxIsChar (pargin [3]))                            \
    {                                                                   \
        A = GB_mx_alias ("A", pargin [3], "C", C, "M", M) ;             \
    }                                                                   \
}

#define FREE_DEEP_COPY          \
{                               \
    if (A == C) A = NULL ;      \
    if (M == C) M = NULL ;      \
    GrB_Matrix_free_(&C) ;      \
}

GrB_Matrix C = NULL ;
GrB_Matrix M = NULL ;
GrB_Matrix A = NULL ;
GrB_Matrix mask = NULL, u = NULL ;
GrB_Descriptor desc = NULL ;
GrB_BinaryOp accum = NULL ;
GrB_Index *I = NULL, ni = 0, I_range [3] ;
GrB_Index *J = NULL, nj = 0, J_range [3] ;
bool ignore ;
bool malloc_debug = false ;
GrB_Info info = GrB_SUCCESS ;
GrB_Monoid reduce = NULL ;
GrB_BinaryOp op = NULL ;
bool user_complex = false ;
int C_sparsity_control ;
int M_sparsity_control ;
bool have_sparsity_control = false ;
bool use_GrB_Scalar = false ;

GrB_Info assign (void) ;

GrB_Info many_subassign
(
    int nwork,
    int fA,
    int fI,
    int fJ,
    int faccum,
    int fM,
    int fdesc,
    int fscalar,
    GrB_Type ctype,
    const mxArray *pargin [ ]
) ;

//------------------------------------------------------------------------------
// assign: perform a single assignment
//------------------------------------------------------------------------------

#define OK(method)                      \
{                                       \
    info = method ;                     \
    if (info != GrB_SUCCESS)            \
    {                                   \
        GrB_Matrix_free_(&mask) ;       \
        GrB_Matrix_free_(&u) ;          \
        return (info) ;                 \
    }                                   \
}

GrB_Info assign (void)
{
    bool at = (desc != NULL && desc->in0 == GrB_TRAN) ;
    GrB_Info info ;

    int pr = GB0 ;
    bool ph = (pr > 0) ;

    ASSERT_MATRIX_OK (C, "C before mex assign", pr) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for mex assign", pr) ;
    ASSERT_MATRIX_OK (A, "A for mex assign", pr) ;

    if (GB_NROWS (A) == 1 && GB_NCOLS (A) == 1 && use_GrB_Scalar)
    {
        // use GxB_Matrix_subassign_Scalar or GxB_Vector_subassign_Scalar
        GrB_Scalar S = (GrB_Scalar) A ;
        if (GB_VECTOR_OK (C) && GB_VECTOR_OK (M))
        {
            OK (GxB_Vector_subassign_Scalar ((GrB_Vector) C, (GrB_Vector) M,
                accum, S, I, ni, desc)) ;
        }
        else
        {
            OK (GxB_Matrix_subassign_Scalar ((GrB_Matrix) C, (GrB_Matrix) M,
                accum, S, I, ni, J, nj, desc)) ;
        }

    }
    else if (GB_NROWS (A) == 1 && GB_NCOLS (A) == 1 && GB_nnz (A) == 1)
    {
        GB_void *Ax = A->x ; // OK: A is a scalar with exactly one entry

        if (ni == 1 && nj == 1 && M == NULL && I != GrB_ALL && J != GrB_ALL
            && GB_op_is_second (accum, C->type) && A->type->code <= GB_FC64_code
            && desc == NULL)
        {
            if (ph) printf ("setElement\n") ;
            // test GrB_Matrix_setElement
            #define ASSIGN(prefix,suffix,type)                          \
            {                                                           \
                type x = ((type *) Ax) [0] ;                            \
                OK (prefix ## Matrix_setElement ## suffix               \
                    (C, x, I [0], J [0])) ;                             \
            } break ;

            switch (A->type->code)
            {

                case GB_BOOL_code   : ASSIGN (GrB_, _BOOL,   bool) ;
                case GB_INT8_code   : ASSIGN (GrB_, _INT8,   int8_t) ;
                case GB_UINT8_code  : ASSIGN (GrB_, _UINT8,  uint8_t) ;
                case GB_INT16_code  : ASSIGN (GrB_, _INT16,  int16_t) ;
                case GB_UINT16_code : ASSIGN (GrB_, _UINT16, uint16_t) ;
                case GB_INT32_code  : ASSIGN (GrB_, _INT32,  int32_t) ;
                case GB_UINT32_code : ASSIGN (GrB_, _UINT32, uint32_t) ;
                case GB_INT64_code  : ASSIGN (GrB_, _INT64,  int64_t) ;
                case GB_UINT64_code : ASSIGN (GrB_, _UINT64, uint64_t) ;
                case GB_FP32_code   : ASSIGN (GrB_, _FP32,   float) ;
                case GB_FP64_code   : ASSIGN (GrB_, _FP64,   double) ;
                case GB_FC32_code   : ASSIGN (GxB_, _FC32,   GxB_FC32_t) ;
                case GB_FC64_code   : ASSIGN (GxB_, _FC64,   GxB_FC64_t) ;
                case GB_UDT_code    :
                default:
                    FREE_ALL ;
                    mexErrMsgTxt ("unsupported type") ;
            }
            #undef ASSIGN

        }
        else if (GB_VECTOR_OK (C) && GB_VECTOR_OK (M))
        {

            // test GxB_Vector_subassign_scalar functions
            if (ph) printf ("scalar assign to vector\n") ;
            #define ASSIGN(suffix,type)                 \
            {                                           \
                type x = ((type *) Ax) [0] ;            \
                OK (GxB_Vector_subassign ## suffix      \
                    ((GrB_Vector) C, (GrB_Vector) M,    \
                    accum, x, I, ni, desc)) ;           \
            } break ;

            switch (A->type->code)
            {

                case GB_BOOL_code   : ASSIGN (_BOOL,   bool) ;
                case GB_INT8_code   : ASSIGN (_INT8,   int8_t) ;
                case GB_UINT8_code  : ASSIGN (_UINT8,  uint8_t) ;
                case GB_INT16_code  : ASSIGN (_INT16,  int16_t) ;
                case GB_UINT16_code : ASSIGN (_UINT16, uint16_t) ;
                case GB_INT32_code  : ASSIGN (_INT32,  int32_t) ;
                case GB_UINT32_code : ASSIGN (_UINT32, uint32_t) ;
                case GB_INT64_code  : ASSIGN (_INT64,  int64_t) ;
                case GB_UINT64_code : ASSIGN (_UINT64, uint64_t) ;
                case GB_FP32_code   : ASSIGN (_FP32,   float) ;
                case GB_FP64_code   : ASSIGN (_FP64,   double) ;
                case GB_FC32_code   : ASSIGN (_FC32,   GxB_FC32_t) ;
                case GB_FC64_code   : ASSIGN (_FC64,   GxB_FC64_t) ;
                case GB_UDT_code    :
                {
                    // user-defined Complex type
                    OK (GxB_Vector_subassign_UDT
                        ((GrB_Vector) C, (GrB_Vector) M,
                        accum, Ax, I, ni, desc)) ;
                }
                break ;
                default:
                    FREE_ALL ;
                    mexErrMsgTxt ("unsupported type") ;
            }
            #undef ASSIGN

        }
        else
        {

            // test Matrix_subassign_scalar functions
            if (ph) printf ("scalar assign to matrix\n") ;
            #define ASSIGN(suffix,type)                     \
            {                                               \
                type x = ((type *) Ax) [0] ;                \
                OK (GxB_Matrix_subassign ## suffix          \
                    (C, M, accum, x, I, ni, J, nj,desc)) ;  \
            } break ;

            switch (A->type->code)
            {

                case GB_BOOL_code   : ASSIGN (_BOOL,   bool) ;
                case GB_INT8_code   : ASSIGN (_INT8,   int8_t) ;
                case GB_UINT8_code  : ASSIGN (_UINT8,  uint8_t) ;
                case GB_INT16_code  : ASSIGN (_INT16,  int16_t) ;
                case GB_UINT16_code : ASSIGN (_UINT16, uint16_t) ;
                case GB_INT32_code  : ASSIGN (_INT32,  int32_t) ;
                case GB_UINT32_code : ASSIGN (_UINT32, uint32_t) ;
                case GB_INT64_code  : ASSIGN (_INT64,  int64_t) ;
                case GB_UINT64_code : ASSIGN (_UINT64, uint64_t) ;
                case GB_FP32_code   : ASSIGN (_FP32,   float) ;
                case GB_FP64_code   : ASSIGN (_FP64,   double) ;
                case GB_FC32_code   : ASSIGN (_FC32,   GxB_FC32_t) ;
                case GB_FC64_code   : ASSIGN (_FC64,   GxB_FC64_t) ;
                case GB_UDT_code    :
                {
                    // user-defined Complex type
                    OK (GxB_Matrix_subassign_UDT
                        (C, M, accum, Ax, I, ni, J, nj, desc)) ;
                }
                break ;

                default:
                    FREE_ALL ;
                    mexErrMsgTxt ("unsupported type") ;
            }
            #undef ASSIGN

        }
    }
    else if (GB_VECTOR_OK (C) && GB_VECTOR_OK (A) &&
        (M == NULL || GB_VECTOR_OK (M)) && !at)
    {
        // test GxB_Vector_subassign
        if (ph) printf ("vector assign\n") ;
        OK (GxB_Vector_subassign_((GrB_Vector) C, (GrB_Vector) M, accum,
            (GrB_Vector) A, I, ni, desc)) ;
    }
    else if (GB_VECTOR_OK (A) && nj == 1 &&
        (M == NULL || GB_VECTOR_OK (M)) && !at)
    {
        // test GxB_Col_subassign
        if (ph) printf ("col assign\n") ;
        OK (GxB_Col_subassign_(C, (GrB_Vector) M, accum, (GrB_Vector) A,
            I, ni, J [0], desc)) ;
    }
    else if (A->vlen == 1 && ni == 1 && nj > 0 &&
        (M == NULL || M->vlen == 1) && !at)
    {
        // test GxB_Row_subassign; this is not meant to be efficient,
        // just for testing
        if (ph) printf ("row assign\n") ;
        if (M != NULL)
        {
            // mask = M'
            int64_t mnrows, mncols ;
            OK (GrB_Matrix_nrows (&mnrows, M)) ;
            OK (GrB_Matrix_ncols (&mncols, M)) ;
            OK (GrB_Matrix_new (&mask, M->type, mncols, mnrows)) ;
            OK (GrB_transpose (mask, NULL, NULL, M, NULL)) ;
            mask->is_csc = true ;
            ASSERT (GB_VECTOR_OK (mask)) ;
        }
        // u = A'
        int64_t ancols, anrows ;
        OK (GrB_Matrix_nrows (&anrows, A)) ;
        OK (GrB_Matrix_ncols (&ancols, A)) ;
        OK (GrB_Matrix_new (&u, A->type, ancols, anrows)) ;
        OK (GrB_transpose (u, NULL, NULL, A, NULL)) ;
        u->is_csc = true ;
        ASSERT (GB_VECTOR_OK (u)) ;
        OK (GxB_Row_subassign_(C, (GrB_Vector) mask, accum, (GrB_Vector) u,
            I [0], J, nj, desc)) ;
        GrB_Matrix_free_(&mask) ;
        GrB_Matrix_free_(&u) ;
    }
    else
    {
        // standard submatrix assignment
        if (ph) printf ("submatrix assign\n") ;
        OK (GxB_Matrix_subassign_(C, M, accum, A, I, ni, J, nj, desc)) ;
    }

    ASSERT_MATRIX_OK (C, "C after assign", pr) ;
    return (info) ;
}

//------------------------------------------------------------------------------
// many_subassign: do a sequence of assignments
//------------------------------------------------------------------------------

// The list of assignments is in a struct array

GrB_Info many_subassign
(
    int nwork,
    int fA,
    int fI,
    int fJ,
    int faccum,
    int fM,
    int fdesc,
    int fscalar,
    GrB_Type ctype,
    const mxArray *pargin [ ]
)
{
    GrB_Info info = GrB_SUCCESS ;

    for (int64_t k = 0 ; k < nwork ; k++)
    {

        //----------------------------------------------------------------------
        // get the kth work to do
        //----------------------------------------------------------------------

        // each struct has fields A, I, J, and optionally Mask, accum, and desc

        mxArray *p ;

        // [ turn off malloc debugging
        bool save = GB_Global_malloc_debug_get ( ) ;
        GB_Global_malloc_debug_set (false) ;

        // get M (deep copy)
        M = NULL ;
        if (fM >= 0)
        {
            p = mxGetFieldByNumber (pargin [1], k, fM) ;
            M = GB_mx_mxArray_to_Matrix (p, "Mask", true, false) ;
            if (M == NULL && !mxIsEmpty (p))
            {
                FREE_ALL ;
                mexErrMsgTxt ("M failed") ;
            }
            if (have_sparsity_control)
            {
                GxB_Matrix_Option_set (M, GxB_SPARSITY_CONTROL,
                    M_sparsity_control) ;
            }
        }

        // get A (true copy)
        p = mxGetFieldByNumber (pargin [1], k, fA) ;
        A = GB_mx_mxArray_to_Matrix (p, "A", true, true) ;
        if (A == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("A failed") ;
        }

        // get accum, if present
        accum = NULL ;
        if (faccum >= 0)
        {
            p = mxGetFieldByNumber (pargin [1], k, faccum) ;
            user_complex = (Complex != GxB_FC64)
                && (C->type == Complex || A->type == Complex) ;
            if (!GB_mx_mxArray_to_BinaryOp (&accum, p, "accum",
                C->type, user_complex))
            {
                FREE_ALL ;
                mexErrMsgTxt ("accum failed") ;
            }
        }

        // get I
        p = mxGetFieldByNumber (pargin [1], k, fI) ;
        if (!GB_mx_mxArray_to_indices (&I, p, &ni, I_range, &ignore))
        {
            FREE_ALL ;
            mexErrMsgTxt ("I failed") ;
        }

        // get J
        p = mxGetFieldByNumber (pargin [1], k, fJ) ;
        if (!GB_mx_mxArray_to_indices (&J, p, &nj, J_range, &ignore))
        {
            FREE_ALL ;
            mexErrMsgTxt ("J failed") ;
        }

        // get desc
        desc = NULL ;
        if (fdesc > 0)
        {
            p = mxGetFieldByNumber (pargin [1], k, fdesc) ;
            if (!GB_mx_mxArray_to_Descriptor (&desc, p, "desc"))
            {
                FREE_ALL ;
                mexErrMsgTxt ("desc failed") ;
            }
        }

        // get use_GrB_Scalar
        use_GrB_Scalar = false ;
        if (fscalar > 0)
        {
            p = mxGetFieldByNumber (pargin [1], k, fscalar) ;
            use_GrB_Scalar = (bool) (mxGetScalar (p) == 2) ;
        }

        // restore malloc debugging to test the method
        GB_Global_malloc_debug_set (save) ; // ]

        //----------------------------------------------------------------------
        // C(I,J)<M> = A
        //----------------------------------------------------------------------

        info = assign ( ) ;

        GrB_Matrix_free_(&A) ;
        GrB_Matrix_free_(&M) ;
        GrB_Descriptor_free_(&desc) ;

        if (info != GrB_SUCCESS)
        {
            return (info) ;
        }
    }

    OK (GrB_Matrix_wait_(C, GrB_MATERIALIZE)) ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_mex_subassign mexFunction
//------------------------------------------------------------------------------

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{

    C = NULL ;
    M = NULL ;
    A = NULL ;
    mask = NULL ;
    u = NULL ;
    desc = NULL ;
    accum = NULL ;
    I = NULL ; ni = 0 ;
    J = NULL ; nj = 0 ;
    malloc_debug = false ;
    info = GrB_SUCCESS ;
    reduce = NULL ;
    op = NULL ;
    user_complex = false ;
    C_sparsity_control = GxB_AUTO_SPARSITY ;
    M_sparsity_control = GxB_AUTO_SPARSITY ;
    have_sparsity_control = false ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    malloc_debug = GB_mx_get_global (true) ;
    A = NULL ;
    C = NULL ;
    M = NULL ;
    desc = NULL ;
    user_complex = false ;
    op = NULL ;
    reduce = NULL ;

    if (!((nargout == 1 && (nargin == 2 || nargin == 3 ||
            nargin == 6 || nargin == 7)) ||
          ((nargout == 2 || nargout == 3) && nargin == 8)))
    {
        GB_mx_put_global (true) ;
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    if (nargin == 2 || nargin == 3)
    {

        // get control if present:
        // [C_sparsity_control M_sparsity_control]
        if (nargin == 3)
        {
            int n = mxGetNumberOfElements (pargin [2]) ;
            if (n != 2) mexErrMsgTxt ("invalid control") ;
            have_sparsity_control = true ;
            double *p = mxGetDoubles (pargin [2]) ;
            C_sparsity_control = (int) p [0] ;
            M_sparsity_control = (int) p [1] ;
        }

        // get C (deep copy)
        GET_DEEP_COPY ;
        if (C == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("C failed") ;
        }

        //----------------------------------------------------------------------
        // get a list of work to do: a struct array of length nwork
        //----------------------------------------------------------------------

        // each entry is a struct with fields:
        // Mask, accum, A, I, J, desc

        if (!mxIsStruct (pargin [1]))
        {
            FREE_ALL ;
            mexErrMsgTxt ("2nd argument must be a struct") ;
        }

        int nwork = mxGetNumberOfElements (pargin [1]) ;
        int nf = mxGetNumberOfFields (pargin [1]) ;
        for (int f = 0 ; f < nf ; f++)
        {
            mxArray *p ;
            for (int k = 0 ; k < nwork ; k++)
            {
                p = mxGetFieldByNumber (pargin [1], k, f) ;
            }
        }

        int fA = mxGetFieldNumber (pargin [1], "A") ;
        int fI = mxGetFieldNumber (pargin [1], "I") ;
        int fJ = mxGetFieldNumber (pargin [1], "J") ;
        int faccum = mxGetFieldNumber (pargin [1], "accum") ;
        int fM = mxGetFieldNumber (pargin [1], "Mask") ;
        int fdesc = mxGetFieldNumber (pargin [1], "desc") ;
        int fscalar = mxGetFieldNumber (pargin [1], "scalar") ;

        if (fA < 0 || fI < 0 || fJ < 0) mexErrMsgTxt ("A,I,J required") ;

        METHOD (many_subassign (nwork, fA, fI, fJ, faccum, fM, fdesc,
            fscalar, C->type, pargin)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // C(I,J)<M> = A, with a single assignment
        //----------------------------------------------------------------------

        // get M (deep copy)
        if (!mxIsChar (pargin [1]))
        {
            M = GB_mx_mxArray_to_Matrix (pargin [1], "M", true, false) ;
            if (M == NULL && !mxIsEmpty (pargin [1]))
            {
                FREE_ALL ;
                mexErrMsgTxt ("M failed") ;
            }
        }

        // get A (deep copy)
        if (!mxIsChar (pargin [3]))
        {
            A = GB_mx_mxArray_to_Matrix (pargin [3], "A", true, true) ;
            if (A == NULL)
            {
                FREE_ALL ;
                mexErrMsgTxt ("A failed") ;
            }
        }

        // get C (deep copy)
        GET_DEEP_COPY ;
        if (C == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("C failed") ;
        }

        // get accum, if present
        user_complex = (Complex != GxB_FC64)
            && (C->type == Complex || A->type == Complex) ;
        accum = NULL ;
        if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
            C->type, user_complex))
        {
            FREE_ALL ;
            mexErrMsgTxt ("accum failed") ;
        }

        // get I
        if (!GB_mx_mxArray_to_indices (&I, pargin [4], &ni, I_range, &ignore))
        {
            FREE_ALL ;
            mexErrMsgTxt ("I failed") ;
        }

        // get J
        if (!GB_mx_mxArray_to_indices (&J, pargin [5], &nj, J_range, &ignore))
        {
            FREE_ALL ;
            mexErrMsgTxt ("J failed") ;
        }

        // get desc
        if (!GB_mx_mxArray_to_Descriptor (&desc, PARGIN (6), "desc"))
        {
            FREE_ALL ;
            mexErrMsgTxt ("desc failed") ;
        }

        if (nargin == 8 && (nargout == 2 || nargout == 3))
        {
            // get reduce operator
            user_complex = (Complex != GxB_FC64) && (C->type == Complex) ;
            if (!GB_mx_mxArray_to_BinaryOp (&op, PARGIN (7), "op",
                C->type, user_complex) || op == NULL)
            {
                FREE_ALL ;
                mexErrMsgTxt ("op failed") ;
            }

            // get the reduce monoid
            if (user_complex)
            {
                if (op == Complex_plus)
                {
                    reduce = Complex_plus_monoid ;
                }
                else if (op == Complex_times)
                {
                    reduce = Complex_times_monoid ;
                }
                else
                {
                    FREE_ALL ;
                    mexErrMsgTxt ("user reduce failed") ;
                }
            }
            else
            {
                // create the reduce monoid
                if (!GB_mx_Monoid (&reduce, op, malloc_debug))
                {
                    FREE_ALL ;
                    mexErrMsgTxt ("reduce failed") ;
                }
            }
        }

        // C(I,J)<M> = A
        METHOD (assign ( )) ;

        // apply the reduce monoid
        if (nargin == 8 && (nargout == 2 || nargout == 3))
        {

            pargout [1] = GB_mx_create_full (1, 1, C->type) ;
            GB_void *p = mxGetData (pargout [1]) ;

            #define REDUCE(prefix,suffix,type,zero)                            \
            {                                                                  \
                type c = zero ;                                                \
                prefix ## Matrix_reduce ## suffix (&c, NULL, reduce, C, NULL) ;\
                memcpy (p, &c, sizeof (type)) ;                                \
            }                                                                  \
            break ;

            double d = 0 ;

            switch (C->type->code)
            {

                case GB_BOOL_code   : REDUCE (GrB_, _BOOL,   bool      , false);
                case GB_INT8_code   : REDUCE (GrB_, _INT8,   int8_t    , 0) ;
                case GB_INT16_code  : REDUCE (GrB_, _INT16,  int16_t   , 0) ;
                case GB_INT32_code  : REDUCE (GrB_, _INT32,  int32_t   , 0) ;
                case GB_INT64_code  : REDUCE (GrB_, _INT64,  int64_t   , 0) ;
                case GB_UINT8_code  : REDUCE (GrB_, _UINT8,  uint8_t   , 0) ;
                case GB_UINT16_code : REDUCE (GrB_, _UINT16, uint16_t  , 0) ;
                case GB_UINT32_code : REDUCE (GrB_, _UINT32, uint32_t  , 0) ;
                case GB_UINT64_code : REDUCE (GrB_, _UINT64, uint64_t  , 0) ;
                case GB_FP32_code   : REDUCE (GrB_, _FP32,   float     , 0) ;
                case GB_FP64_code   : REDUCE (GrB_, _FP64,   double    , 0) ;
                case GB_FC32_code   :
                    REDUCE (GxB_, _FC32, GxB_FC32_t, GxB_CMPLXF (0,0)) ;
                case GB_FC64_code   :
                    REDUCE (GxB_, _FC64,   GxB_FC64_t, GxB_CMPLX  (0,0)) ;
                case GB_UDT_code    :
                    {
                        // user-defined Complex type
                        GxB_FC64_t c = GxB_CMPLX (0,0) ;
                        GrB_Matrix_reduce_UDT_((void *) &c, NULL, reduce,
                            C, NULL) ;
                        memcpy (p, &c, sizeof (GxB_FC64_t)) ;
                    }
                    break ;

                default             :
                    FREE_ALL ;
                    mexErrMsgTxt ("unknown type: subassign reduce") ;
            }

            GrB_Matrix_reduce_FP64_(&d, NULL, GxB_PLUS_FP64_MONOID, C, NULL) ;
            if (nargout > 2) pargout [2] = mxCreateDoubleScalar (d) ;

        }
    }

    //--------------------------------------------------------------------------
    // return C as a struct
    //--------------------------------------------------------------------------

    ASSERT_MATRIX_OK (C, "Final C before wait", GB0) ;
    GrB_Matrix_wait_(C, GrB_MATERIALIZE) ;

    if (C == A) A = NULL ;      // do not free A if it is aliased to C
    if (C == M) M = NULL ;      // do not free M if it is aliased to C
    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C assign result", true) ;
    FREE_ALL ;
}

