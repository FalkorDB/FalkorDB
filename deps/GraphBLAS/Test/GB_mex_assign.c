//------------------------------------------------------------------------------
// GB_mex_assign: C<Mask>(I,J) = accum (C (I,J), A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// This function is a wrapper for GrB_Matrix_assign, GrB_Matrix_assign_T
// GrB_Vector_assign, and GrB_Vector_assign_T (when kind=0 or by default).  For
// these uses, the Mask must always be the same size as C.

// This mexFunction does calls GrB_Row_assign (when kind=2) or GrB_Col_assign
// (when kind=1).  In these cases, the Mask is a single row or column,
// respectively.  C is not modified outside that single row (for
// GrB_Row_assign) or column (for GrB_Col_assign).

// This function does the same thing as the mimics GB_spec_assign.m
// (when kind=0), GB_spec_Col_assign (when kind=1), and GB_spec_Row_assign
// (when kind=2).

// When kind=3, the indices I and J are passed into GxB_Matrix_assign_Vector
// as GrB_Vectors, not plain uint64_t * arrays.

// kind=4: GxB_Col_assign_Vector
// kind=5: GxB_Row_assign_Vector
// kind=6: GxB_Matrix_assign_Scalar_Vector or GxB_Vector_assign_Scalar_Vector
// kind=7: GxB_Vector_assign_Vector or GxB_Matrix_assign_Vector

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C = GB_mex_assign(C,Mask,acc,A,I,J,desc,kind) or (C,Work,ctrl)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&Mask) ;           \
    GrB_Matrix_free_(&C) ;              \
    GrB_Vector_free_(&I_vector) ;       \
    GrB_Vector_free_(&J_vector) ;       \
    GrB_Descriptor_free_(&desc) ;       \
    GB_mx_put_global (true) ;           \
}

#define GET_DEEP_COPY \
    C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;         \
    if (have_sparsity_control)                                                \
    {                                                                         \
        GxB_Matrix_Option_set (C, GxB_SPARSITY_CONTROL, C_sparsity_control) ; \
    }

#define FREE_DEEP_COPY GrB_Matrix_free_(&C) ;

GrB_Matrix C = NULL ;
GrB_Matrix Mask = NULL ;
GrB_Matrix A = NULL ;
GrB_Descriptor desc = NULL ;
GrB_BinaryOp accum = NULL ;
uint64_t *I = NULL, ni = 0, I_range [3] ; GrB_Vector I_vector = NULL ;
uint64_t *J = NULL, nj = 0, J_range [3] ; GrB_Vector J_vector = NULL ;
bool ignore ;
bool malloc_debug = false ;
GrB_Info info = GrB_SUCCESS ;
int kind = 0 ;
GrB_Info assign (void) ;
int C_sparsity_control ;
int M_sparsity_control ;
bool have_sparsity_control = false ;
bool use_GrB_Scalar = false ;

GrB_Info many_assign
(
    int nwork,
    int fA,
    int fI,
    int fJ,
    int faccum,
    int fMask,
    int fdesc,
    int fscalar,
    int fkind,
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
        return (info) ;                 \
    }                                   \
}

GrB_Info assign ( )
{
    bool at = (desc != NULL && desc->in0 == GrB_TRAN) ;
    GrB_Info info ;

    int pr = 0 ;
    bool ph = (pr > 0) ;

    ASSERT_MATRIX_OK (C, "C for GB_mex_assign", pr) ;
    ASSERT_MATRIX_OK_OR_NULL (Mask, "Mask for GB_mex_assign", pr) ;
    ASSERT_MATRIX_OK (A, "A for GB_mex_assign", pr) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum for GB_mex_assign", pr) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc for GB_mex_assign", pr) ;

    if (kind == 1)
    {

        //----------------------------------------------------------------------
        // test GrB_Col_assign
        //----------------------------------------------------------------------

        ASSERT (GB_VECTOR_OK (A)) ;
        ASSERT (Mask == NULL || GB_VECTOR_OK (Mask)) ;
        OK (GrB_Col_assign_(C, (GrB_Vector) Mask, accum, (GrB_Vector) A,
            I, ni, J [0], desc)) ;

    }
    else if (kind == 4)
    {

        //----------------------------------------------------------------------
        // test GxB_Col_assign_Vector
        //----------------------------------------------------------------------

        uint64_t j0 = 0 ;
        OK (GrB_Vector_extractElement_UINT64_(&j0, J_vector, 0)) ;

        ASSERT (GB_VECTOR_OK (A)) ;
        ASSERT (Mask == NULL || GB_VECTOR_OK (Mask)) ;
        OK (GxB_Col_assign_Vector(C, (GrB_Vector) Mask, accum, (GrB_Vector) A,
            I_vector, j0, desc)) ;

    }
    else if (kind == 2)
    {

        //----------------------------------------------------------------------
        // test GrB_Row_assign
        //----------------------------------------------------------------------

        ASSERT (GB_VECTOR_OK (A)) ;
        ASSERT (Mask == NULL || GB_VECTOR_OK (Mask)) ;
        ASSERT_VECTOR_OK_OR_NULL ((GrB_Vector) Mask, "row mask", GB0) ;
        ASSERT_VECTOR_OK ((GrB_Vector) A, "row u", GB0) ;

        OK (GrB_Row_assign_(C, (GrB_Vector) Mask, accum, (GrB_Vector) A,
            I [0], J, nj, desc)) ;

    }
    else if (kind == 5)
    {

        //----------------------------------------------------------------------
        // test GxB_Row_assign_Vector
        //----------------------------------------------------------------------

        uint64_t i0 = 0 ;
        OK (GrB_Vector_extractElement_UINT64_(&i0, I_vector, 0)) ;

        ASSERT (GB_VECTOR_OK (A)) ;
        ASSERT (Mask == NULL || GB_VECTOR_OK (Mask)) ;
        ASSERT_VECTOR_OK_OR_NULL ((GrB_Vector) Mask, "row mask", GB0) ;
        ASSERT_VECTOR_OK ((GrB_Vector) A, "row u", GB0) ;

        OK (GxB_Row_assign_Vector_(C, (GrB_Vector) Mask, accum, (GrB_Vector) A,
            i0, J_vector, desc)) ;

    }
    else if (kind == 3)
    {

        //----------------------------------------------------------------------
        // test GxB_*_assign_Vector, with GrB_Vectors as I and J
        //----------------------------------------------------------------------

        if (GB_VECTOR_OK (C) && GB_VECTOR_OK (Mask))
        {
            OK (GxB_Vector_assign_Vector_((GrB_Vector) C, (GrB_Vector) Mask,
                accum, (GrB_Vector) A, I_vector, desc)) ;
        }
        else
        {
            OK (GxB_Matrix_assign_Vector_((GrB_Matrix) C, (GrB_Matrix) Mask,
                accum, A, I_vector, J_vector, desc)) ;
        }

    }
    else if (GB_NROWS (A) == 1 && GB_NCOLS (A) == 1 && use_GrB_Scalar)
    {

        //----------------------------------------------------------------------
        // use GrB_Matrix_assign_Scalar or GrB_Vector_assign_Scalar
        //----------------------------------------------------------------------

        GrB_Scalar S = (GrB_Scalar) A ;

        // OK but not used; see GB_mex_assign_scalar.c instead
        if (kind == 6)
        {
            // test _Vector variant
            if (GB_VECTOR_OK (C) && GB_VECTOR_OK (Mask))
            {
                OK (GxB_Vector_assign_Scalar_Vector_((GrB_Vector) C,
                    (GrB_Vector) Mask, accum, S, I_vector, desc)) ;
            }
            else
            {
                OK (GxB_Matrix_assign_Scalar_Vector_((GrB_Matrix) C,
                    (GrB_Matrix) Mask, accum, S, I_vector, J_vector, desc)) ;
            }
        }
        else
        {
            // test original variants with (uint64_t *) arrays I and J
            if (GB_VECTOR_OK (C) && GB_VECTOR_OK (Mask))
            {
                OK (GrB_Vector_assign_Scalar_((GrB_Vector) C, (GrB_Vector) Mask,
                    accum, S, I, ni, desc)) ;
            }
            else
            {
                OK (GrB_Matrix_assign_Scalar_((GrB_Matrix) C, (GrB_Matrix) Mask,
                    accum, S, I, ni, J, nj, desc)) ;
            }
        }

    }
    else if (GB_NROWS (A) == 1 && GB_NCOLS (A) == 1 && GB_nnz (A) == 1
        && kind == 0)
    {

        //----------------------------------------------------------------------
        // test GrB_Matrix_assign_TYPE or GrB_Vector_assign_TYPE
        //----------------------------------------------------------------------

        GB_void *Ax = A->x ; // OK: A is a scalar with exactly one entry

        if (ni == 1 && nj == 1 && Mask == NULL && I != GrB_ALL && J != GrB_ALL
            && GB_op_is_second (accum, C->type) && A->type->code <= GB_FC64_code
            && desc == NULL)
        {

            //------------------------------------------------------------------
            // test GrB_Matrix_setElement
            //------------------------------------------------------------------

            #define ASSIGN(prefix,suffix, type)                         \
            {                                                           \
                type x = ((type *) Ax) [0] ;                            \
                OK (prefix ## Matrix_setElement ## suffix               \
                    (C, x, I [0], J [0])) ;                             \
            } break ;

            switch (A->type->code)
            {
                case GB_BOOL_code   : ASSIGN (GrB_, _BOOL,   bool) ;
                case GB_INT8_code   : ASSIGN (GrB_, _INT8,   int8_t) ;
                case GB_INT16_code  : ASSIGN (GrB_, _INT16,  int16_t) ;
                case GB_INT32_code  : ASSIGN (GrB_, _INT32,  int32_t) ;
                case GB_INT64_code  : ASSIGN (GrB_, _INT64,  int64_t) ;
                case GB_UINT8_code  : ASSIGN (GrB_, _UINT8,  uint8_t) ;
                case GB_UINT16_code : ASSIGN (GrB_, _UINT16, uint16_t) ;
                case GB_UINT32_code : ASSIGN (GrB_, _UINT32, uint32_t) ;
                case GB_UINT64_code : ASSIGN (GrB_, _UINT64, uint64_t) ;
                case GB_FP32_code   : ASSIGN (GrB_, _FP32,   float) ;
                case GB_FP64_code   : ASSIGN (GrB_, _FP64,   double) ;
                case GB_FC32_code   : ASSIGN (GxB_, _FC32,   GxB_FC32_t) ;
                case GB_FC64_code   : ASSIGN (GxB_, _FC64,   GxB_FC64_t) ;
                case GB_UDT_code    :
                default:
                    FREE_ALL ;
                    mexErrMsgTxt ("GB_mex_assign: unknown type, setEl") ;
            }

            ASSERT_MATRIX_OK (C, "C after setElement", GB0) ;

        }
        else if (GB_VECTOR_OK (C) && (Mask == NULL || GB_VECTOR_OK (Mask)))
        {

            //------------------------------------------------------------------
            // test GrB_Vector_assign_scalar functions
            //------------------------------------------------------------------

            #undef  ASSIGN
            #define ASSIGN(prefix,suffix,type)                          \
            {                                                           \
                type x = ((type *) Ax) [0] ;                            \
                OK (prefix ## Vector_assign ## suffix ((GrB_Vector) C,  \
                    (GrB_Vector) Mask, accum, x, I, ni, desc)) ;        \
            } break ;

            switch (A->type->code)
            {
                case GB_BOOL_code   : ASSIGN (GrB_, _BOOL,   bool) ;
                case GB_INT8_code   : ASSIGN (GrB_, _INT8,   int8_t) ;
                case GB_INT16_code  : ASSIGN (GrB_, _INT16,  int16_t) ;
                case GB_INT32_code  : ASSIGN (GrB_, _INT32,  int32_t) ;
                case GB_INT64_code  : ASSIGN (GrB_, _INT64,  int64_t) ;
                case GB_UINT8_code  : ASSIGN (GrB_, _UINT8,  uint8_t) ;
                case GB_UINT16_code : ASSIGN (GrB_, _UINT16, uint16_t) ;
                case GB_UINT32_code : ASSIGN (GrB_, _UINT32, uint32_t) ;
                case GB_UINT64_code : ASSIGN (GrB_, _UINT64, uint64_t) ;
                case GB_FP32_code   : ASSIGN (GrB_, _FP32,   float) ;
                case GB_FP64_code   : ASSIGN (GrB_, _FP64,   double) ;
                case GB_FC32_code   : ASSIGN (GxB_, _FC32,   GxB_FC32_t) ;
                case GB_FC64_code   : ASSIGN (GxB_, _FC64,   GxB_FC64_t) ;
                case GB_UDT_code    :
                    {
                        OK (GrB_Vector_assign_UDT ((GrB_Vector) C,
                            (GrB_Vector) Mask, accum, Ax, I, ni, desc)) ;
                    }
                    break ;
                default:
                    FREE_ALL ;
                    mexErrMsgTxt ("GB_mex_assign: unknown type") ;
            }

        }
        else
        {

            //------------------------------------------------------------------
            // test Matrix_assign_scalar functions
            //------------------------------------------------------------------

            #undef  ASSIGN
            #define ASSIGN(prefix,suffix,type)                  \
            {                                                   \
                type x = ((type *) Ax) [0] ;                    \
                OK (prefix ## Matrix_assign ## suffix           \
                    (C, Mask, accum, x, I, ni, J, nj,desc)) ;   \
            } break ;

            switch (A->type->code)
            {
                case GB_BOOL_code   : ASSIGN (GrB_, _BOOL,   bool) ;
                case GB_INT8_code   : ASSIGN (GrB_, _INT8,   int8_t) ;
                case GB_INT16_code  : ASSIGN (GrB_, _INT16,  int16_t) ;
                case GB_INT32_code  : ASSIGN (GrB_, _INT32,  int32_t) ;
                case GB_INT64_code  : ASSIGN (GrB_, _INT64,  int64_t) ;
                case GB_UINT8_code  : ASSIGN (GrB_, _UINT8,  uint8_t) ;
                case GB_UINT16_code : ASSIGN (GrB_, _UINT16, uint16_t) ;
                case GB_UINT32_code : ASSIGN (GrB_, _UINT32, uint32_t) ;
                case GB_UINT64_code : ASSIGN (GrB_, _UINT64, uint64_t) ;
                case GB_FP32_code   : ASSIGN (GrB_, _FP32,   float) ;
                case GB_FP64_code   : ASSIGN (GrB_, _FP64,   double) ;
                case GB_FC32_code   : ASSIGN (GxB_, _FC32,   GxB_FC32_t) ;
                case GB_FC64_code   : ASSIGN (GxB_, _FC64,   GxB_FC64_t) ;
                case GB_UDT_code    :
                {
                    OK (GrB_Matrix_assign_UDT
                        (C, Mask, accum, Ax, I, ni, J, nj, desc)) ;
                }
                break ;

                default:
                    FREE_ALL ;
                    mexErrMsgTxt ("unknown type: mtx assign") ;
            }
        }

    }
    else if (GB_VECTOR_OK (C) && GB_VECTOR_OK (A) &&
        (Mask == NULL || GB_VECTOR_OK (Mask)) && !at)
    {

        //----------------------------------------------------------------------
        // test GrB_Vector_assign and GxB_Vector_assign_Vector
        //----------------------------------------------------------------------

        if (kind == 7)
        {
            OK (GxB_Vector_assign_Vector_((GrB_Vector) C, (GrB_Vector) Mask,
                accum, (GrB_Vector) A, I_vector, desc)) ;
        }
        else
        {
            OK (GrB_Vector_assign_((GrB_Vector) C, (GrB_Vector) Mask, accum,
                (GrB_Vector) A, I, ni, desc)) ;
        }
    }
    else
    {

        //----------------------------------------------------------------------
        // standard submatrix assignment
        //----------------------------------------------------------------------

        if (kind == 7)
        {
            OK (GxB_Matrix_assign_Vector_(C, Mask, accum, A, I_vector,
                J_vector, desc)) ;
        }
        else
        {
            OK (GrB_Matrix_assign_(C, Mask, accum, A, I, ni, J, nj, desc)) ;
        }
    }

    ASSERT_MATRIX_OK (C, "Final C before wait", GB0) ;
    OK (GrB_Matrix_wait_(C, GrB_MATERIALIZE)) ;
    return (info) ;
}

//------------------------------------------------------------------------------
// many_assign: do a sequence of assignments
//------------------------------------------------------------------------------

// The list of assignments is in a struct array

GrB_Info many_assign
(
    int nwork,
    int fA,
    int fI,
    int fJ,
    int faccum,
    int fMask,
    int fdesc,
    int fscalar,
    int fkind,
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

        // get Mask (deep copy)
        Mask = NULL ;
        if (fMask >= 0)
        {
            p = mxGetFieldByNumber (pargin [1], k, fMask) ;
            Mask = GB_mx_mxArray_to_Matrix (p, "Mask", true, false) ;
            if (Mask == NULL && !mxIsEmpty (p))
            {
                FREE_ALL ;
                mexErrMsgTxt ("Mask failed") ;
            }
            if (have_sparsity_control)
            {
                GxB_Matrix_Option_set (Mask, GxB_SPARSITY_CONTROL,
                    M_sparsity_control) ;
            }
        }

        // get A (deep copy)
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
            bool user_complex = (Complex != GxB_FC64)
                && (C->type == Complex || A->type == Complex) ;
            if (!GB_mx_mxArray_to_BinaryOp (&accum, p, "accum",
                C->type, user_complex))
            {
                FREE_ALL ;
                mexErrMsgTxt ("accum failed") ;
            }
        }

        // get kind (0: matrix/vector, 1: col_assign, 2: row_assign,
        // 3 or more: matrix/vector with I and J as GrB_Vectors)
        kind = 0 ;
        if (fkind > 0)
        {
            p = mxGetFieldByNumber (pargin [1], k, fkind) ;
            kind = (int) mxGetScalar (p) ;
        }

        // get I: may be a GrB_Vector
        p = mxGetFieldByNumber (pargin [1], k, fI) ;
        if (!GB_mx_mxArray_to_indices (p, &I, &ni, I_range, &ignore,
            (kind >= 3) ? (&I_vector) : NULL))
        {
            FREE_ALL ;
            mexErrMsgTxt ("I failed") ;
        }

        // get J: may be a GrB_Vector
        p = mxGetFieldByNumber (pargin [1], k, fJ) ;
        if (!GB_mx_mxArray_to_indices (p, &J, &nj, J_range, &ignore,
            (kind >= 3) ? (&J_vector) : NULL))
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
        // C<Mask>(I,J) = A
        //----------------------------------------------------------------------

        info = assign ( ) ;

        GrB_Matrix_free_(&A) ;
        GrB_Matrix_free_(&Mask) ;
        GrB_Vector_free_(&I_vector) ;
        GrB_Vector_free_(&J_vector) ;
        GrB_Descriptor_free_(&desc) ;

        if (info != GrB_SUCCESS)
        {
            return (info) ;
        }
    }

    ASSERT_MATRIX_OK (C, "Final C before wait", GB0) ;
    OK (GrB_Matrix_wait_(C, GrB_MATERIALIZE)) ;
    return (info) ;
}

//------------------------------------------------------------------------------
// GB_mex_assign mexFunction
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
    Mask = NULL ;
    A = NULL ;
    desc = NULL ;
    accum = NULL ;
    I = NULL ; ni = 0 ;
    J = NULL ; nj = 0 ;
    malloc_debug = false ;
    info = GrB_SUCCESS ;
    kind = 0 ;
    C_sparsity_control = GxB_AUTO_SPARSITY ;
    M_sparsity_control = GxB_AUTO_SPARSITY ;
    have_sparsity_control = false ;
    I_vector = NULL ;
    J_vector = NULL ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    malloc_debug = GB_mx_get_global (true) ;
    A = NULL ;
    C = NULL ;
    Mask = NULL ;
    desc = NULL ;

    // check inputs
    if (nargout > 1 || !
        (nargin == 2 || nargin == 3 || nargin == 6 || nargin == 7 ||
         nargin == 8 || nargin == 9))
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
    }

    // get control if present: [C_sparsity_control M_sparsity_control]
    if (nargin == 3)
    {
        int n = mxGetNumberOfElements (pargin [2]) ;
        if (n != 2) mexErrMsgTxt ("invalid control") ;
        have_sparsity_control = true ;
        double *p = mxGetDoubles (pargin [2]) ;
        C_sparsity_control = (int) p [0] ;
        M_sparsity_control = (int) p [1] ;
    }

    //--------------------------------------------------------------------------
    // get C (make a deep copy)
    //--------------------------------------------------------------------------

    GET_DEEP_COPY ;
    if (C == NULL)
    {
        FREE_ALL ;
        mexErrMsgTxt ("C failed") ;
    }

    if (nargin == 2 || nargin == 3)
    {

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
        int fMask = mxGetFieldNumber (pargin [1], "Mask") ;
        int fdesc = mxGetFieldNumber (pargin [1], "desc") ;
        int fkind = mxGetFieldNumber (pargin [1], "kind") ;
        int fscalar = mxGetFieldNumber (pargin [1], "scalar") ;

        if (fA < 0 || fI < 0 || fJ < 0) mexErrMsgTxt ("A,I,J required") ;

        METHOD (many_assign (nwork, fA, fI, fJ, faccum, fMask, fdesc,
            fscalar, fkind, pargin)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // C<Mask>(I,J) = A, with a single assignment
        //----------------------------------------------------------------------

        // get Mask (deep copy)
        Mask = GB_mx_mxArray_to_Matrix (pargin [1], "Mask", true, false) ;
        if (Mask == NULL && !mxIsEmpty (pargin [1]))
        {
            FREE_ALL ;
            mexErrMsgTxt ("Mask failed") ;
        }

        // get A (deep copy)
        A = GB_mx_mxArray_to_Matrix (pargin [3], "A", true, true) ;
        if (A == NULL)
        {
            FREE_ALL ;
            mexErrMsgTxt ("A failed") ;
        }

        // get accum, if present
        bool user_complex = (Complex != GxB_FC64)
            && (C->type == Complex || A->type == Complex) ;
        accum = NULL ;
        if (!GB_mx_mxArray_to_BinaryOp (&accum, pargin [2], "accum",
            C->type, user_complex))
        {
            FREE_ALL ;
            mexErrMsgTxt ("accum failed") ;
        }

        // get kind (0: matrix/vector, 1: col_assign, 2: row_assign,
        // 3 or more: matrix/vector with I and J as GrB_Vectors)
        kind = 0 ;
        if (nargin > 7)
        {
            kind = (int) mxGetScalar (pargin [7]) ;
        }

        // get I: may be a GrB_Vector
        if (!GB_mx_mxArray_to_indices (pargin [4], &I, &ni, I_range, &ignore,
            (kind >= 3) ? (&I_vector) : NULL))
        {
            FREE_ALL ;
            mexErrMsgTxt ("I failed") ;
        }

        // get J: may be a GrB_Vector
        if (!GB_mx_mxArray_to_indices (pargin [5], &J, &nj, J_range, &ignore,
            (kind >= 3) ? (&J_vector) : NULL))
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

        // C<Mask>(I,J) = A
        METHOD (assign ( )) ;
    }

    //--------------------------------------------------------------------------
    // return C as a struct
    //--------------------------------------------------------------------------

    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C assign result", true) ;
    FREE_ALL ;
}

