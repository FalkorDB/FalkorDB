//------------------------------------------------------------------------------
// GB_sort: sort all vectors in a matrix
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// JIT: needed.

#include "GB_sort.h"
#include "GB_transpose.h"
#include "GB_ek_slice.h"

//  macros:

//  GB_SORT (func)      defined as GB_sort_func_TYPE_ascend or _descend,
//                      GB_msort_ISO_ascend or _descend,
//                      or GB_msort_func_UDT
//  GB_TYPE             bool, int8_, ... or GB_void for UDT

//  GB_ADDR(A,p)        A+p for builtin, A + p * GB_SIZE otherwise
//  GB_SIZE             size of each entry: sizeof (GB_TYPE) for built-in
//  GB_GET(x,X,i)       x = (op->xtype) X [i]
//  GB_COPY(A,i,C,k)    A [i] = C [k]
//  GB_SWAP(A,i,k)      swap A [i] and A [k]
//  GB_LT               compare two entries, x < y

//------------------------------------------------------------------------------
// macros for all built-in types
//------------------------------------------------------------------------------

#define GB_SORT_UDT         0
#define GB_ADDR(A,i)        ((A) + (i))
#define GB_GET(x,A,i)       GB_TYPE x = A [i]
#define GB_COPY(A,i,B,j)    A [i] = B [j]
#define GB_SIZE             sizeof (GB_TYPE)
#define GB_SWAP(A,i,j)      { GB_TYPE t = A [i] ; A [i] = A [j] ; A [j] = t ; }

//------------------------------------------------------------------------------
// ascending sort for built-in types
//------------------------------------------------------------------------------

#define GB_LT(less,a,i,b,j)  \
    less = (((a) < (b)) ? true : (((a) == (b)) ? ((i) < (j)) : false))

#define GB_TYPE             bool
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _ascend_BOOL)
#include "GB_sort_template.c"

#define GB_TYPE             int8_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _ascend_INT8)
#include "GB_sort_template.c"

#define GB_TYPE             int16_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _ascend_INT16)
#include "GB_sort_template.c"

#define GB_TYPE             int32_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _ascend_INT32)
#include "GB_sort_template.c"

#define GB_TYPE             int64_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _ascend_INT64)
#include "GB_sort_template.c"

#define GB_TYPE             uint8_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _ascend_UINT8)
#include "GB_sort_template.c"

#define GB_TYPE             uint16_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _ascend_UINT16)
#include "GB_sort_template.c"

#define GB_TYPE             uint32_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _ascend_UINT32)
#include "GB_sort_template.c"

#define GB_TYPE             uint64_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _ascend_UINT64)
#include "GB_sort_template.c"

#define GB_TYPE             float
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _ascend_FP32)
#include "GB_sort_template.c"

#define GB_TYPE             double
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _ascend_FP64)
#include "GB_sort_template.c"

//------------------------------------------------------------------------------
// descending sort for built-in types
//------------------------------------------------------------------------------

#undef  GB_LT
#define GB_LT(less,a,i,b,j)  \
    less = (((a) > (b)) ? true : (((a) == (b)) ? ((i) < (j)) : false))

#define GB_TYPE             bool
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _descend_BOOL)
#include "GB_sort_template.c"

#define GB_TYPE             int8_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _descend_INT8)
#include "GB_sort_template.c"

#define GB_TYPE             int16_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _descend_INT16)
#include "GB_sort_template.c"

#define GB_TYPE             int32_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _descend_INT32)
#include "GB_sort_template.c"

#define GB_TYPE             int64_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _descend_INT64)
#include "GB_sort_template.c"

#define GB_TYPE             uint8_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _descend_UINT8)
#include "GB_sort_template.c"

#define GB_TYPE             uint16_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _descend_UINT16)
#include "GB_sort_template.c"

#define GB_TYPE             uint32_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _descend_UINT32)
#include "GB_sort_template.c"

#define GB_TYPE             uint64_t
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _descend_UINT64)
#include "GB_sort_template.c"

#define GB_TYPE             float
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _descend_FP32)
#include "GB_sort_template.c"

#define GB_TYPE             double
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _descend_FP64)
#include "GB_sort_template.c"

//------------------------------------------------------------------------------
// macros for user-defined types and when typecasting is performed 
//------------------------------------------------------------------------------

#undef  GB_ADDR
#undef  GB_GET
#undef  GB_COPY
#undef  GB_SIZE
#undef  GB_SWAP
#undef  GB_LT

#define GB_ADDR(A,i)        ((A) + (i) * csize)
#define GB_GET(x,A,i)       GB_void x [GB_VLA(xsize)] ;                     \
                            fcast (x, GB_ADDR (A, i), csize)
#define GB_COPY(A,i,B,j)    memcpy (GB_ADDR (A, i), GB_ADDR (B, j), csize)
#define GB_SIZE             csize
#define GB_TYPE             GB_void

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
#define GB_SORT(func)       GB_EVAL3 (GB(sort_), func, _UDT)
#include "GB_sort_template.c"

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

// redefine to use the revised GB_FREE_ALL above:
#include "GB_static_header.h"

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
    GrB_Type ptype = (P == NULL) ? GrB_INT64 : P->type ;

    if (op->ztype != GrB_BOOL || op->xtype != op->ytype || atype != ctype
        || !(ptype == GrB_INT64 || ptype == GrB_UINT64)
        || !GB_Type_compatible (atype, op->xtype))
    { 
        // op must return bool, and its inputs x and y must have the same type;
        // the types of A and C must match exactly; P must be INT64 or UINT64;
        // A and C must be typecasted to the input type of the op.
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
    GB_phybix_free (P) ;
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
        GB_CLEAR_STATIC_HEADER (T, &T_header) ;
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
                // A = C
                GB_OK (GB_dup_worker (&C, A_iso, A, true, atype)) ;
            }
        }
        else
        {
            // A is CSR but C must be CSC
            if (sort_in_place)
            { 
                // A = A'
                GB_OK (GB_transpose_in_place (A, true, Werk)) ;
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
                // A = C
                GB_OK (GB_dup_worker (&C, A_iso, A, true, atype)) ;
            }
        }
        else
        {
            // A is CSC but C must be CSR
            if (sort_in_place)
            { 
                // A = A'
                GB_OK (GB_transpose_in_place (A, false, Werk)) ;
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

    GB_Opcode opcode = op->opcode ;
    GB_Type_code acode = atype->code ;

    if ((op->xtype == atype) && (op->ytype == atype) &&
        (opcode == GB_LT_binop_code || opcode == GB_GT_binop_code) &&
        (acode < GB_UDT_code))
    {

        //----------------------------------------------------------------------
        // no typecasting, using built-in < or > operators, builtin types
        //----------------------------------------------------------------------

        if (opcode == GB_LT_binop_code)
        { 
            // ascending sort
            switch (acode)
            {
                case GB_BOOL_code : 
                    GB_OK (GB(sort_matrix_ascend_BOOL   )(C, Werk)) ; break ;
                case GB_INT8_code : 
                    GB_OK (GB(sort_matrix_ascend_INT8   )(C, Werk)) ; break ;
                case GB_INT16_code : 
                    GB_OK (GB(sort_matrix_ascend_INT16  )(C, Werk)) ; break ;
                case GB_INT32_code : 
                    GB_OK (GB(sort_matrix_ascend_INT32  )(C, Werk)) ; break ;
                case GB_INT64_code : 
                    GB_OK (GB(sort_matrix_ascend_INT64  )(C, Werk)) ; break ;
                case GB_UINT8_code : 
                    GB_OK (GB(sort_matrix_ascend_UINT8  )(C, Werk)) ; break ;
                case GB_UINT16_code : 
                    GB_OK (GB(sort_matrix_ascend_UINT16 )(C, Werk)) ; break ;
                case GB_UINT32_code : 
                    GB_OK (GB(sort_matrix_ascend_UINT32 )(C, Werk)) ; break ;
                case GB_UINT64_code : 
                    GB_OK (GB(sort_matrix_ascend_UINT64 )(C, Werk)) ; break ;
                case GB_FP32_code : 
                    GB_OK (GB(sort_matrix_ascend_FP32   )(C, Werk)) ; break ;
                case GB_FP64_code : 
                    GB_OK (GB(sort_matrix_ascend_FP64   )(C, Werk)) ; break ;
                default:;
            }
        }
        else // opcode == GB_GT_binop_code
        { 
            // descending sort
            switch (acode)
            {
                case GB_BOOL_code : 
                    GB_OK (GB(sort_matrix_descend_BOOL  )(C, Werk)) ; break ;
                case GB_INT8_code : 
                    GB_OK (GB(sort_matrix_descend_INT8  )(C, Werk)) ; break ;
                case GB_INT16_code : 
                    GB_OK (GB(sort_matrix_descend_INT16 )(C, Werk)) ; break ;
                case GB_INT32_code : 
                    GB_OK (GB(sort_matrix_descend_INT32 )(C, Werk)) ; break ;
                case GB_INT64_code : 
                    GB_OK (GB(sort_matrix_descend_INT64 )(C, Werk)) ; break ;
                case GB_UINT8_code : 
                    GB_OK (GB(sort_matrix_descend_UINT8 )(C, Werk)) ; break ;
                case GB_UINT16_code : 
                    GB_OK (GB(sort_matrix_descend_UINT16)(C, Werk)) ; break ;
                case GB_UINT32_code : 
                    GB_OK (GB(sort_matrix_descend_UINT32)(C, Werk)) ; break ;
                case GB_UINT64_code : 
                    GB_OK (GB(sort_matrix_descend_UINT64)(C, Werk)) ; break ;
                case GB_FP32_code : 
                    GB_OK (GB(sort_matrix_descend_FP32  )(C, Werk)) ; break ;
                case GB_FP64_code : 
                    GB_OK (GB(sort_matrix_descend_FP64  )(C, Werk)) ; break ;
                default:;
            }
        }

    }
    else
    { 

        //----------------------------------------------------------------------
        // typecasting, user-defined types, or unconventional operators
        //----------------------------------------------------------------------

        GB_OK (GB (sort_matrix_UDT) (C, op, Werk)) ;
    }

    //--------------------------------------------------------------------------
    // constuct the final indices
    //--------------------------------------------------------------------------

    int64_t cnz = GB_nnz (C) ;
    int64_t cnvec = C->nvec ;
    int64_t *restrict Ti = NULL ;

    if (P == NULL)
    { 
        // P is not constructed; use C->i to construct the new indices
        Ti = C->i ;
    }
    else
    {
        // allocate P->i and use it to construct the new indices
        P->i = GB_MALLOC (cnz, int64_t, &(P->i_size)) ;
        if (P->i == NULL)
        { 
            // out of memory
            GB_FREE_ALL ;
            return (GrB_OUT_OF_MEMORY) ;
        }
        Ti = P->i ;
    }

    int C_nthreads, C_ntasks ;
    GB_SLICE_MATRIX (C, 1) ;
    int64_t *restrict Cp = C->p ;
    const int64_t cvlen = C->vlen ;
    int tid ;
    #pragma omp parallel for num_threads(C_nthreads) schedule(static,1)
    for (tid = 0 ; tid < C_ntasks ; tid++)
    {
        int64_t kfirst = kfirst_Cslice [tid] ;
        int64_t klast  = klast_Cslice  [tid] ;
        for (int64_t k = kfirst ; k <= klast ; k++)
        {
            const int64_t pC0 = Cp [k] ;
            int64_t pC_start, pC_end ;
            GB_get_pA (&pC_start, &pC_end, tid, k,
                kfirst, klast, pstart_Cslice, Cp, cvlen) ;
            for (int64_t pC = pC_start ; pC < pC_end ; pC++)
            { 
                Ti [pC] = pC - pC0 ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // construct P
    //--------------------------------------------------------------------------

    bool C_is_hyper = GB_IS_HYPERSPARSE (C) ;

    if (P != NULL)
    {
        P->is_csc = C->is_csc ;
        P->nvec = C->nvec ;
        P->nvec_nonempty = C->nvec_nonempty ;
        P->iso = false ;
        P->vlen = C->vlen ;
        P->vdim = C->vdim ;

        if (C_is_NULL)
        { 
            // the values of C are not needed.  The indices of C become the
            // values of P, Cp becomes Pp, and Ch (if present) becomes Ph.
            P->x = C->i ; C->i = NULL ; P->x_size = C->i_size ;
            P->p = C->p ; C->p = NULL ; P->p_size = C->p_size ;
            P->h = C->h ; C->h = NULL ; P->h_size = C->h_size ;
            P->plen = C->plen ;
        }
        else
        {
            // C is required on output.  The indices of C are copied and
            // become the values of P.  Cp is copied to Pp, and Ch (if present)
            // is copied to Ph.
            int64_t pplen = GB_IMAX (1, cnvec) ;
            P->plen = pplen ;
            P->x = GB_MALLOC (cnz, int64_t, &(P->x_size)) ; // x:OK
            P->p = GB_MALLOC (pplen+1, int64_t, &(P->p_size)) ;
            P->h = NULL ;
            if (C_is_hyper)
            { 
                P->h = GB_MALLOC (pplen, int64_t, &(P->h_size)) ;
            }
            if (P->x == NULL || P->p == NULL || (C_is_hyper && P->h == NULL))
            { 
                // out of memory
                GB_FREE_ALL ;
                return (GrB_OUT_OF_MEMORY) ;
            }

            // copy from C to P
            GB_memcpy (P->x, C->i, cnz * sizeof (int64_t), nthreads_max) ;
            GB_memcpy (P->p, C->p, (cnvec+1) * sizeof (int64_t), nthreads_max) ;
            if (C_is_hyper)
            { 
                GB_memcpy (P->h, C->h, cnvec * sizeof (int64_t), nthreads_max) ;
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
        GB_memcpy (C->i, P->i, cnz * sizeof (int64_t), nthreads_max) ;
    }

    //--------------------------------------------------------------------------
    // free workspace, and comform/return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    if (!C_is_NULL)
    { 
        ASSERT_MATRIX_OK (C, "C output of GB_sort (before conform)", GB0) ;
        GB_OK (GB_conform (C, Werk)) ;
        ASSERT_MATRIX_OK (C, "C output of GB_sort", GB0) ;
    }
    if (P != NULL)
    { 
        ASSERT_MATRIX_OK (P, "P output of GB_sort (before conform)", GB0) ;
        GB_OK (GB_conform (P, Werk)) ;
        ASSERT_MATRIX_OK (P, "P output of GB_sort", GB0) ;
    }
    return (GrB_SUCCESS) ;
}

