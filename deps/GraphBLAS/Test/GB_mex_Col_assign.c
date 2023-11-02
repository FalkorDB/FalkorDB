//------------------------------------------------------------------------------
// GB_mex_assign: C<Mask>(I,J) = accum (C (I,J), A)
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// This function is a wrapper for GrB_Matrix_assign, GrB_Matrix_assign_T
// GrB_Vector_assign, and GrB_Vector_assign_T.  For these uses, the Mask must
// always be the same size as C.

// This mexFunction does not call GrB_Row_assign or GrB_Col_assign, since
// the Mask is a single row or column in these cases, and C is not modified
// outside that single row (for GrB_Row_assign) or column (for GrB_Col_assign).

// This function does the same thing as the mimic GB_spec_assign.m.

//------------------------------------------------------------------------------

#include "GB_mex.h"

#define USAGE "C =GB_mex_assign (C, Mask, accum, A, I, J, desc) or (C, Work)"

#define FREE_ALL                        \
{                                       \
    GrB_Matrix_free_(&A) ;              \
    GrB_Matrix_free_(&Mask) ;           \
    GrB_Matrix_free_(&C) ;              \
    GrB_Descriptor_free_(&desc) ;       \
    GB_mx_put_global (true) ;           \
}

#define GET_DEEP_COPY \
    C = GB_mx_mxArray_to_Matrix (pargin [0], "C input", true, true) ;

#define FREE_DEEP_COPY GrB_Matrix_free_(&C) ;

GrB_Matrix C = NULL ;
GrB_Matrix Mask = NULL ;
GrB_Matrix A = NULL ;
GrB_Descriptor desc = NULL ;
GrB_BinaryOp accum = NULL ;
GrB_Index *I = NULL, ni = 0, I_range [3] ;
GrB_Index *J = NULL, nj = 0, J_range [3] ;
bool ignore ;
bool malloc_debug = false ;
GrB_Info info = GrB_SUCCESS ;
GrB_Info assign (void) ;

GrB_Info many_assign
(
    int nwork,
    int fA,
    int fI,
    int fJ,
    int faccum,
    int fMask,
    int fdesc,
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

    ASSERT_MATRIX_OK (C, "C", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (Mask, "Mask", GB0) ;
    ASSERT_MATRIX_OK (A, "A", GB0) ;
    ASSERT_BINARYOP_OK_OR_NULL (accum, "accum", GB0) ;
    ASSERT_DESCRIPTOR_OK_OR_NULL (desc, "desc", GB0) ;

    if (GB_NROWS (A) == 1 && GB_NCOLS (A) == 1 && GB_nnz (A) == 1)
    {
        GB_void *Ax = A->x ; // OK: A is a scalar with exactly one entry

        if (ni == 1 && nj == 1 && Mask == NULL && I != GrB_ALL && J != GrB_ALL
            && GB_op_is_second (accum, C->type) && A->type->code < GB_FC64_code
            && desc == NULL)
        {
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
                    mexErrMsgTxt ("unknown type: col setEl") ;
            }

            ASSERT_MATRIX_OK (C, "C after setElement", GB0) ;

        }
        else if (C->vdim == 1)
        {

            // test GrB_Vector_assign_scalar functions
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
                    mexErrMsgTxt ("unknown type: vec assign") ;
            }

        }
        else
        {

            // test Matrix_assign_scalar functions
            #undef  ASSIGN
            #define ASSIGN(prefix,suffix,type)                          \
            {                                                           \
                type x = ((type *) Ax) [0] ;                            \
                OK (prefix ## Matrix_assign ## suffix (C, Mask, accum,  \
                    x, I, ni, J, nj,desc)) ;                            \
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
                    OK (GrB_Matrix_assign_UDT (C, Mask, accum,
                        Ax, I, ni, J, nj, desc)) ;
                }
                break ;

                default:
                    FREE_ALL ;
                    mexErrMsgTxt ("unknown type: mtx assign") ;
            }
        }

    }
    else if (C->vdim == 1 && A->vdim == 1 &&
        (Mask == NULL || Mask->vdim == 1) && !at)
    {
        // test GrB_Vector_assign
        OK (GrB_Vector_assign_((GrB_Vector) C, (GrB_Vector) Mask, accum,
            (GrB_Vector) A, I, ni, desc)) ;
    }
    else
    {
        // standard submatrix assignment
        OK (GrB_Matrix_assign_(C, Mask, accum, A, I, ni, J, nj, desc)) ;
    }

    ASSERT_MATRIX_OK (C, "Final C before wait", GB0) ;
    OK (GrB_Matrix_wait (C, GrB_MATERIALIZE)) ;
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

        // get Mask (shallow copy)
        Mask = NULL ;
        if (fMask >= 0)
        {
            p = mxGetFieldByNumber (pargin [1], k, fMask) ;
            Mask = GB_mx_mxArray_to_Matrix (p, "Mask", false, false) ;
            if (Mask == NULL && !mxIsEmpty (p))
            {
                FREE_ALL ;
                mexErrMsgTxt ("Mask failed") ;
            }
        }

        // get A (shallow copy)
        p = mxGetFieldByNumber (pargin [1], k, fA) ;
        A = GB_mx_mxArray_to_Matrix (p, "A", false, true) ;
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

        // restore malloc debugging to test the method
        GB_Global_malloc_debug_set (save) ; // ]

        //----------------------------------------------------------------------
        // C<Mask>(I,J) = A
        //----------------------------------------------------------------------

        info = assign ( ) ;

        GrB_Matrix_free_(&A) ;
        GrB_Matrix_free_(&Mask) ;
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

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    malloc_debug = GB_mx_get_global (true) ;
    A = NULL ;
    C = NULL ;
    Mask = NULL ;
    desc = NULL ;

    if (nargout > 1 || ! (nargin == 2 || nargin == 6 || nargin == 7))
    {
        mexErrMsgTxt ("Usage: " USAGE) ;
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

    if (nargin == 2)
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

        if (fA < 0 || fI < 0 || fJ < 0) mexErrMsgTxt ("A,I,J required") ;

        METHOD (many_assign (nwork, fA, fI, fJ, faccum, fMask, fdesc, pargin)) ;

    }
    else
    {

        //----------------------------------------------------------------------
        // C<Mask>(I,J) = A, with a single assignment
        //----------------------------------------------------------------------

        // get Mask (shallow copy)
        Mask = GB_mx_mxArray_to_Matrix (pargin [1], "Mask", false, false) ;
        if (Mask == NULL && !mxIsEmpty (pargin [1]))
        {
            FREE_ALL ;
            mexErrMsgTxt ("Mask failed") ;
        }

        // get A (shallow copy)
        A = GB_mx_mxArray_to_Matrix (pargin [3], "A", false, true) ;
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

        // C<Mask>(I,J) = A

        METHOD (assign ( )) ;
    }

    //--------------------------------------------------------------------------
    // return C as a struct
    //--------------------------------------------------------------------------

    pargout [0] = GB_mx_Matrix_to_mxArray (&C, "C assign result", true) ;
    FREE_ALL ;
}

