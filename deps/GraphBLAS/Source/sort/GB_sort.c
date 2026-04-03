//------------------------------------------------------------------------------
// GB_sort: sort all vectors in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

#include "sort/GB_sort.h"
#include "transpose/GB_transpose.h"
#include "jitifyer/GB_stringify.h"

//  macros:

//  GB_SORT (func)      defined as GB_sort_XX_func_TYPE_ascend or _descend,
//                      or GB_sort_XX_func_UDT where XX = 32 or 64.
//  GB_C_TYPE           bool, int8_, ... or GB_void for UDT
//  GB_Ci_TYPE          the type of C->i (uint32_t or uint64_t)
//  GB_ADDR(A,p)        A+p for builtin, A + p * GB_SIZE otherwise
//  GB_SIZE             size of each entry: sizeof (GB_C_TYPE) for built-in
//  GB_GETX(x,X,i)      x = (op->xtype) X [i]
//  GB_COPY(A,i,C,k)    A [i] = C [k]
//  GB_SWAP(A,i,k)      swap A [i] and A [k]
//  GB_LT               compare two entries, x < y

//------------------------------------------------------------------------------
// macros for all built-in types
//------------------------------------------------------------------------------

#define GB_SORT_UDT         0
#define GB_ADDR(A,i)        ((A) + (i))
#define GB_GETX(x,A,i)      GB_C_TYPE x = A [i]
#define GB_COPY(A,i,B,j)    A [i] = B [j]
#define GB_SIZE             sizeof (GB_C_TYPE)
#define GB_SWAP(A,i,j)      \
{                           \
    GB_C_TYPE t = A [i] ;   \
    A [i] = A [j] ;         \
    A [j] = t ;             \
}

//------------------------------------------------------------------------------
// ascending sort for built-in types
//------------------------------------------------------------------------------

#define GB_LT(less,a,i,b,j)  \
    less = (((a) < (b)) ? true : (((a) == (b)) ? ((i) < (j)) : false))

#define GB_Ci_TYPE          uint32_t

#define GB_C_TYPE           bool
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _ascend_BOOL)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int8_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _ascend_INT8)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int16_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _ascend_INT16)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int32_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _ascend_INT32)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int64_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _ascend_INT64)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint8_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _ascend_UINT8)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint16_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _ascend_UINT16)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint32_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _ascend_UINT32)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint64_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _ascend_UINT64)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           float
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _ascend_FP32)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           double
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _ascend_FP64)
#include "sort/template/GB_sort_template.c"

#undef  GB_Ci_TYPE
#define GB_Ci_TYPE          uint64_t

#define GB_C_TYPE           bool
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _ascend_BOOL)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int8_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _ascend_INT8)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int16_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _ascend_INT16)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int32_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _ascend_INT32)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int64_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _ascend_INT64)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint8_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _ascend_UINT8)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint16_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _ascend_UINT16)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint32_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _ascend_UINT32)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint64_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _ascend_UINT64)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           float
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _ascend_FP32)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           double
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _ascend_FP64)
#include "sort/template/GB_sort_template.c"


//------------------------------------------------------------------------------
// descending sort for built-in types
//------------------------------------------------------------------------------

#undef  GB_LT
#define GB_LT(less,a,i,b,j)  \
    less = (((a) > (b)) ? true : (((a) == (b)) ? ((i) < (j)) : false))

#undef  GB_Ci_TYPE
#define GB_Ci_TYPE          uint32_t

#define GB_C_TYPE           bool
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _descend_BOOL)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int8_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _descend_INT8)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int16_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _descend_INT16)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int32_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _descend_INT32)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int64_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _descend_INT64)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint8_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _descend_UINT8)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint16_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _descend_UINT16)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint32_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _descend_UINT32)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint64_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _descend_UINT64)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           float
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _descend_FP32)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           double
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _descend_FP64)
#include "sort/template/GB_sort_template.c"

#undef  GB_Ci_TYPE
#define GB_Ci_TYPE          uint64_t

#define GB_C_TYPE           bool
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _descend_BOOL)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int8_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _descend_INT8)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int16_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _descend_INT16)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int32_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _descend_INT32)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           int64_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _descend_INT64)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint8_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _descend_UINT8)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint16_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _descend_UINT16)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint32_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _descend_UINT32)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           uint64_t
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _descend_UINT64)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           float
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _descend_FP32)
#include "sort/template/GB_sort_template.c"

#define GB_C_TYPE           double
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _descend_FP64)
#include "sort/template/GB_sort_template.c"

//------------------------------------------------------------------------------
// macros for the generic kernel
//------------------------------------------------------------------------------

#undef  GB_ADDR
#undef  GB_GETX
#undef  GB_COPY
#undef  GB_SIZE
#undef  GB_SWAP
#undef  GB_LT

#define GB_ADDR(A,i)        ((A) + (i) * csize)
#define GB_GETX(x,A,i)      GB_void x [GB_VLA(xsize)] ;                     \
                            fcast (x, GB_ADDR (A, i), csize)
#define GB_COPY(A,i,B,j)    memcpy (GB_ADDR (A, i), GB_ADDR (B, j), csize)
#define GB_SIZE             csize

#define GB_SWAP(A,i,j)                                                      \
{                                                                           \
    GB_void t [GB_VLA(csize)] ;         /* declare the scalar t */          \
    memcpy (t, GB_ADDR (A, i), csize) ; /* t = A [i] */                     \
    GB_COPY (A, i, A, j) ;              /* A [i] = A [j] */                 \
    memcpy (GB_ADDR (A, j), t, csize) ; /* A [j] = t */                     \
}

#define GB_LT(less,a,i,b,j)                                                 \
{                                                                           \
    flt (&less, a, b) ;         /* less = (a < b) */                        \
    if (!less)                                                              \
    {                                                                       \
        /* check for equality and tie-break on index */                     \
        bool more ;                                                         \
        flt (&more, b, a) ;     /* more = (b < a) */                        \
        less = (more) ? false : ((i) < (j)) ;                               \
    }                                                                       \
}

#undef  GB_SORT_UDT
#define GB_SORT_UDT 1

#undef  GB_Ci_TYPE
#define GB_Ci_TYPE          uint32_t
#define GB_C_TYPE           GB_void
#define GB_SORT(func)       GB_EVAL3 (GB (sort_32_), func, _UDT)
#include "sort/template/GB_sort_template.c"

#undef  GB_Ci_TYPE
#define GB_Ci_TYPE          uint64_t
#define GB_C_TYPE           GB_void
#define GB_SORT(func)       GB_EVAL3 (GB (sort_64_), func, _UDT)
#include "sort/template/GB_sort_template.c"

//------------------------------------------------------------------------------
// GB_sort
//------------------------------------------------------------------------------

#undef  GB_FREE_WORKSPACE
#define GB_FREE_WORKSPACE                   \
{                                           \
    GB_WERK_POP (C_ek_slicing, int64_t) ;   \
    GB_Matrix_free (&T) ;                   \
}

#undef  GB_FREE_ALL
#define GB_FREE_ALL                         \
{                                           \
    GB_FREE_WORKSPACE ;                     \
    if (!C_is_NULL) GB_phybix_free (C) ;    \
    GB_phybix_free (P) ;                    \
}

GrB_Info GB_sort
(
    // output:
    GrB_Matrix C,               // matrix with sorted vectors on output
    GrB_Matrix P,               // matrix with permutations on output
    // input:
    GrB_BinaryOp op,            // comparator for the sort
    GrB_Matrix A,               // matrix to sort
    const bool A_transpose,     // false: sort each row, true: sort each column
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (A, "A for GB_sort", GB0) ;
    ASSERT_BINARYOP_OK (op, "op for GB_sort", GB0) ;

    GrB_Matrix T = NULL ;
    struct GB_Matrix_opaque T_header ;
    GB_WERK_DECLARE (C_ek_slicing, int64_t) ;

    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;

    bool C_is_NULL = (C == NULL) ;
    if (C_is_NULL && P == NULL)
    { 
        // either C, or P, or both must be present
        return (GrB_NULL_POINTER) ;
    }

    GrB_Type atype = A->type ;
    GrB_Type ctype = (C_is_NULL) ? atype : C->type ;
    GrB_Type ptype = (P == NULL) ? NULL : P->type ;
    bool ptype_ok = true ;

    if (ptype != NULL)
    { 
        // if present, P must have a int32, uint32, int64, uint64 data type
        // that is large enough to hold the row/column indices of C
        int64_t n = GB_IMAX (A->vlen, A->vdim) ;
        int64_t nmax = -1 ;
        switch (ptype->code)
        {
            case GB_INT32_code  : nmax =  INT32_MAX ; break ;
            case GB_UINT32_code : nmax = UINT32_MAX ; break ;
            case GB_INT64_code  : nmax =  INT64_MAX ; break ;
            case GB_UINT64_code : nmax =  INT64_MAX ; break ;
            default             : nmax = -1         ; break ;
        }
        ptype_ok = (n <= nmax) ;
    }

    if (op->ztype != GrB_BOOL || op->xtype != op->ytype || atype != ctype
        || !ptype_ok
        || !GB_Type_compatible (atype, op->xtype)
        || GB_IS_INDEXBINARYOP_CODE (op->opcode))
    { 
        // op must return bool, and its inputs x and y must have the same type.
        // The types of A and C must match exactly.  P must have a valid data
        // type if present.  A and C must be typecasted to the input type of
        // the op.  Positional ops are not allowed.
        return (GrB_DOMAIN_MISMATCH) ;
    }

    int64_t anrows = GB_NROWS (A) ;
    int64_t ancols = GB_NCOLS (A) ;
    if ((C != NULL && (GB_NROWS (C) != anrows || GB_NCOLS (C) != ancols)) ||
        (P != NULL && (GB_NROWS (P) != anrows || GB_NCOLS (P) != ancols)))
    { 
        // C and P must have the same dimensions as A
        return (GrB_DIMENSION_MISMATCH) ;
    }

    bool A_iso = A->iso ;
    bool sort_in_place = (A == C) ;

    // free any prior content of C and P
    if (A == P)
    { 
        return (GrB_NOT_IMPLEMENTED) ;  // A and P cannot be aliased
    }
    GB_phybix_free (P) ;    // this frees A if A == P are aliased
    if (!sort_in_place)
    { 
        GB_phybix_free (C) ;
    }

    //--------------------------------------------------------------------------
    // make a copy of A, unless it is aliased with C
    //--------------------------------------------------------------------------

    if (C_is_NULL)
    { 
        // C is a temporary matrix, which is freed when done
        GB_CLEAR_MATRIX_HEADER (T, &T_header) ;
        C = T ;
    }

    if (A_transpose)
    {
        // ensure C is in sparse or hypersparse CSC format
        if (A->is_csc)
        {
            // A is already CSC
            if (!sort_in_place)
            { 
                // C = A
                GB_OK (GB_dup_worker (&C, A_iso, A, true, atype)) ;
            }
        }
        else
        {
            // A is CSR but C must be CSC
            if (sort_in_place)
            { 
                // C = C'
                GB_OK (GB_transpose_in_place (C, true, Werk)) ;
            }
            else
            { 
                // C = A'
                GB_OK (GB_transpose_cast (C, atype, true, A, false, Werk)) ;
            }
        }
    }
    else
    {
        // ensure C is in sparse or hypersparse CSR format
        if (!A->is_csc)
        {
            // A is already CSR
            if (!sort_in_place)
            { 
                // C = A
                GB_OK (GB_dup_worker (&C, A_iso, A, true, atype)) ;
            }
        }
        else
        {
            // A is CSC but C must be CSR
            if (sort_in_place)
            { 
                // C = C'
                GB_OK (GB_transpose_in_place (C, false, Werk)) ;
            }
            else
            { 
                // C = A'
                GB_OK (GB_transpose_cast (C, atype, false, A, false, Werk)) ;
            }
        }
    }

    // ensure C is sparse or hypersparse
    if (GB_IS_BITMAP (C) || GB_IS_FULL (C))
    { 
        GB_OK (GB_convert_any_to_sparse (C, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // sort C in place
    //--------------------------------------------------------------------------

    int64_t cnz = GB_nnz (C) ;
    int nthreads = GB_nthreads (cnz, chunk, nthreads_max) ;

    GB_Opcode opcode = op->opcode ;
    GB_Type_code acode = atype->code ;

    ASSERT_MATRIX_OK (C, "C to sort", GB0) ;

    if (C->iso || cnz <= 1)
    { 

        //----------------------------------------------------------------------
        // C is iso: nothing to do
        //----------------------------------------------------------------------

        ;

    }
    else if ((op->xtype == atype) && (op->ytype == atype) &&
        (opcode == GB_LT_binop_code || opcode == GB_GT_binop_code) &&
        (acode < GB_UDT_code))
    {

        //----------------------------------------------------------------------
        // no typecasting, using built-in < or > operators, builtin types
        //----------------------------------------------------------------------

        if (opcode == GB_LT_binop_code && C->i_is_32)
        { 
            // ascending sort, 32-bit integers
            switch (acode)
            {
                case GB_BOOL_code : 
                    GB_OK (GB (sort_32_mtx_ascend_BOOL   )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT8_code : 
                    GB_OK (GB (sort_32_mtx_ascend_INT8   )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT16_code : 
                    GB_OK (GB (sort_32_mtx_ascend_INT16  )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT32_code : 
                    GB_OK (GB (sort_32_mtx_ascend_INT32  )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT64_code : 
                    GB_OK (GB (sort_32_mtx_ascend_INT64  )(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT8_code : 
                    GB_OK (GB (sort_32_mtx_ascend_UINT8  )(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT16_code : 
                    GB_OK (GB (sort_32_mtx_ascend_UINT16 )(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT32_code : 
                    GB_OK (GB (sort_32_mtx_ascend_UINT32 )(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT64_code : 
                    GB_OK (GB (sort_32_mtx_ascend_UINT64 )(C, nthreads, Werk)) ;
                    break ;
                case GB_FP32_code : 
                    GB_OK (GB (sort_32_mtx_ascend_FP32   )(C, nthreads, Werk)) ;
                    break ;
                case GB_FP64_code : 
                    GB_OK (GB (sort_32_mtx_ascend_FP64   )(C, nthreads, Werk)) ;
                    break ;
                default:;
            }
        }
        else if (opcode == GB_GT_binop_code && C->i_is_32)
        { 
            // descending sort
            switch (acode)
            {
                case GB_BOOL_code : 
                    GB_OK (GB (sort_32_mtx_descend_BOOL  )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT8_code : 
                    GB_OK (GB (sort_32_mtx_descend_INT8  )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT16_code : 
                    GB_OK (GB (sort_32_mtx_descend_INT16 )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT32_code : 
                    GB_OK (GB (sort_32_mtx_descend_INT32 )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT64_code : 
                    GB_OK (GB (sort_32_mtx_descend_INT64 )(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT8_code : 
                    GB_OK (GB (sort_32_mtx_descend_UINT8 )(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT16_code : 
                    GB_OK (GB (sort_32_mtx_descend_UINT16)(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT32_code : 
                    GB_OK (GB (sort_32_mtx_descend_UINT32)(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT64_code : 
                    GB_OK (GB (sort_32_mtx_descend_UINT64)(C, nthreads, Werk)) ;
                    break ;
                case GB_FP32_code : 
                    GB_OK (GB (sort_32_mtx_descend_FP32  )(C, nthreads, Werk)) ;
                    break ;
                case GB_FP64_code : 
                    GB_OK (GB (sort_32_mtx_descend_FP64  )(C, nthreads, Werk)) ;
                    break ;
                default:;
            }
        }
        else if (opcode == GB_LT_binop_code && !C->i_is_32)
        { 
            // ascending sort, 64-bit integers
            switch (acode)
            {
                case GB_BOOL_code : 
                    GB_OK (GB (sort_64_mtx_ascend_BOOL   )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT8_code : 
                    GB_OK (GB (sort_64_mtx_ascend_INT8   )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT16_code : 
                    GB_OK (GB (sort_64_mtx_ascend_INT16  )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT32_code : 
                    GB_OK (GB (sort_64_mtx_ascend_INT32  )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT64_code : 
                    GB_OK (GB (sort_64_mtx_ascend_INT64  )(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT8_code : 
                    GB_OK (GB (sort_64_mtx_ascend_UINT8  )(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT16_code : 
                    GB_OK (GB (sort_64_mtx_ascend_UINT16 )(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT32_code : 
                    GB_OK (GB (sort_64_mtx_ascend_UINT32 )(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT64_code : 
                    GB_OK (GB (sort_64_mtx_ascend_UINT64 )(C, nthreads, Werk)) ;
                    break ;
                case GB_FP32_code : 
                    GB_OK (GB (sort_64_mtx_ascend_FP32   )(C, nthreads, Werk)) ;
                    break ;
                case GB_FP64_code : 
                    GB_OK (GB (sort_64_mtx_ascend_FP64   )(C, nthreads, Werk)) ;
                    break ;
                default:;
            }
        }
        else // if (opcode == GB_GT_binop_code && !C->i_is_32)
        { 
            // descending sort
            switch (acode)
            {
                case GB_BOOL_code : 
                    GB_OK (GB (sort_64_mtx_descend_BOOL  )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT8_code : 
                    GB_OK (GB (sort_64_mtx_descend_INT8  )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT16_code : 
                    GB_OK (GB (sort_64_mtx_descend_INT16 )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT32_code : 
                    GB_OK (GB (sort_64_mtx_descend_INT32 )(C, nthreads, Werk)) ;
                    break ;
                case GB_INT64_code : 
                    GB_OK (GB (sort_64_mtx_descend_INT64 )(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT8_code : 
                    GB_OK (GB (sort_64_mtx_descend_UINT8 )(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT16_code : 
                    GB_OK (GB (sort_64_mtx_descend_UINT16)(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT32_code : 
                    GB_OK (GB (sort_64_mtx_descend_UINT32)(C, nthreads, Werk)) ;
                    break ;
                case GB_UINT64_code : 
                    GB_OK (GB (sort_64_mtx_descend_UINT64)(C, nthreads, Werk)) ;
                    break ;
                case GB_FP32_code : 
                    GB_OK (GB (sort_64_mtx_descend_FP32  )(C, nthreads, Werk)) ;
                    break ;
                case GB_FP64_code : 
                    GB_OK (GB (sort_64_mtx_descend_FP64  )(C, nthreads, Werk)) ;
                    break ;
                default:;
            }
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // typecasting, user-defined types, or unconventional operators
        //----------------------------------------------------------------------

        // via the JIT kernel
        info = GB_sort_jit (C, op, nthreads, Werk) ;

        // via the generic kernel
        if (info == GrB_NO_VALUE)
        {
            if (C->i_is_32)
            { 
                info = GB (sort_32_mtx_UDT) (C, op, nthreads, Werk) ;
            }
            else
            { 
                info = GB (sort_64_mtx_UDT) (C, op, nthreads, Werk) ;
            }
        }

        GB_OK (info) ;
    }

    //--------------------------------------------------------------------------
    // constuct the final indices
    //--------------------------------------------------------------------------

    int64_t cnvec = C->nvec ;
    GB_MDECL (Ti, ,) ;
    bool Ti_is_32 ;

    if (P == NULL)
    { 
        // P is not constructed; use C->i to construct the new indices
        ASSERT (!C_is_NULL) ;
        Ti = C->i ;
        Ti_is_32 = C->i_is_32 ;
    }
    else
    {
        // allocate P->i and use it to construct the new indices
        size_t pisize = P->i_is_32 ? sizeof (uint32_t) : sizeof (uint64_t) ;
        P->i = GB_MALLOC_MEMORY (cnz, pisize, &(P->i_size)) ;
        if (P->i == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        Ti = P->i ;
        Ti_is_32 = P->i_is_32 ;
    }

    GB_IPTR (Ti, Ti_is_32) ;

    int C_nthreads, C_ntasks ;
    GB_SLICE_MATRIX (C, 1) ;
    GB_Cp_DECLARE (Cp, ) ; GB_Cp_PTR (Cp, C) ;

    int tid ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(static,1)
    for (tid = 0 ; tid < C_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Cslice [tid] ;
        int64_t klast  = klast_Cslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            const int64_t pC0 = GB_IGET (Cp, k) ;
            GB_GET_PA (pC_start, pC_end, tid, k, kfirst, klast, pstart_Cslice,
                pC0, GB_IGET (Cp, k+1)) ;
            for (int64_t pC = pC_start ; pC < pC_end ; pC++)
            { 
                int64_t i = pC - pC0 ;
                GB_ISET (Ti, pC, i) ;   // Ti [pC] = i
            }
        }
    }

    //--------------------------------------------------------------------------
    // construct P
    //--------------------------------------------------------------------------

    bool C_is_hyper = GB_IS_HYPERSPARSE (C) ;
    size_t cisize = (C->i_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
    GB_Type_code cpcode = (C->p_is_32) ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code cjcode = (C->j_is_32) ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code cicode = (C->i_is_32) ? GB_UINT32_code : GB_UINT64_code ;
    GB_Type_code picode, pjcode ;

    if (P != NULL)
    {
        P->is_csc = C->is_csc ;
        P->nvec = C->nvec ;
//      P->nvec_nonempty = C->nvec_nonempty ;
        GB_nvec_nonempty_set (P, GB_nvec_nonempty_get (C)) ;
        P->iso = false ;
        P->vlen = C->vlen ;
        P->vdim = C->vdim ;

        size_t ppsize = (P->p_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
        size_t pjsize = (P->j_is_32) ? sizeof (uint32_t) : sizeof (uint64_t) ;
        size_t pxsize = ptype->size ;
        GB_Type_code pxcode = ptype->code ;
        GB_Type_code ppcode = (P->p_is_32) ? GB_UINT32_code : GB_UINT64_code ;
        pjcode = (P->j_is_32) ? GB_UINT32_code : GB_UINT64_code ;
        picode = (P->i_is_32) ? GB_UINT32_code : GB_UINT64_code ;

        if (C_is_NULL && pxsize == cisize &&
            P->p_is_32 == C->p_is_32 &&
            P->j_is_32 == C->j_is_32 &&
            P->i_is_32 == C->i_is_32)
        { 
            // C is a temporary matrix T, and its contents are not needed.  The
            // indices of C become the values of P, Cp becomes Pp, and Ch (if
            // present) becomes Ph.
            P->x = C->i ; C->i = NULL ; P->x_size = C->i_size ;
            P->p = C->p ; C->p = NULL ; P->p_size = C->p_size ;
            P->h = C->h ; C->h = NULL ; P->h_size = C->h_size ;
            P->plen = C->plen ;
        }
        else
        {
            // C is required on output, or it has integers of the wrong size.
            // The indices of C are copied and become the values of P.  Cp is
            // copied to Pp, and Ch (if present) is copied to Ph.
            int64_t pplen = GB_IMAX (1, cnvec) ;
            P->plen = pplen ;
            P->x = GB_MALLOC_MEMORY (cnz, pxsize, &(P->x_size)) ;
            P->p = GB_MALLOC_MEMORY (pplen+1, ppsize, &(P->p_size)) ;
            P->h = NULL ;
            if (C_is_hyper)
            { 
                P->h = GB_MALLOC_MEMORY (pplen, pjsize, &(P->h_size)) ;
            }
            if (P->x == NULL || P->p == NULL || (C_is_hyper && P->h == NULL))
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }

            // copy from C to P
//          GB_memcpy (P->x, C->i, cnz * sizeof (int64_t), nthreads_max) ;
            GB_cast_int (P->x, pxcode, C->i, cicode, cnz, nthreads_max) ;
//          GB_memcpy (P->p, C->p, (cnvec+1) * sizeof (int64_t), nthreads_max) ;
            GB_cast_int (P->p, ppcode, C->p, cpcode, cnvec+1, nthreads_max) ;
            if (C_is_hyper)
            { 
//              GB_memcpy (P->h, C->h, cnvec * sizeof (int64_t), nthreads_max) ;
                GB_cast_int (P->h, pjcode, C->h, cjcode, cnvec, nthreads_max) ;
            }
        }

        P->nvals = cnz ;
        P->magic = GB_MAGIC ;
    }

    //--------------------------------------------------------------------------
    // finalize the pattern of C
    //--------------------------------------------------------------------------

    if (!C_is_NULL && P != NULL)
    { 
        // copy P->i into C->i
//      GB_memcpy (C->i, P->i, cnz * sizeof (int64_t), nthreads_max) ;
        GB_cast_int (C->i, cicode, P->i, picode, cnz, nthreads_max) ;
    }

    //--------------------------------------------------------------------------
    // free workspace, and comform/return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;

    if (!C_is_NULL)
    { 
        GB_OK (GB_conform (C, Werk)) ;
        ASSERT_MATRIX_OK (C, "C output of GB_sort", GB0) ;
    }
    if (P != NULL)
    { 
        GB_OK (GB_conform (P, Werk)) ;
        ASSERT_MATRIX_OK (P, "P output of GB_sort", GB0) ;
    }
    return (GrB_SUCCESS) ;
}

