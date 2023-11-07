//------------------------------------------------------------------------------
// GB_AxB_dot4: compute C+=A'*B in-place
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_AxB_dot4 does its computation in a single phase, computing its result in
// the input matrix C, which is already as-if-full (in any format).  The mask M
// is not handled by this function.  C is not iso on output, but might be iso
// on input (if so, C is converted from iso on input to non-iso on output).

// The accum operator is the same as monoid operator semiring->add->op, and the
// type of C (C->type) matches the accum->ztype so no typecasting is needed
// from the monoid ztype to C.

// The ANY monoid is a special case: C is not modified at all.

// JIT: done.

//------------------------------------------------------------------------------

#include "GB_mxm.h"
#include "GB_binop.h"
#include "GB_unused.h"
#include "GB_stringify.h"
#ifndef GBCOMPACT
#include "GB_AxB__include2.h"
#endif

#define GB_FREE_WORKSPACE               \
{                                       \
    GB_WERK_POP (B_slice, int64_t) ;    \
    GB_WERK_POP (A_slice, int64_t) ;    \
}

#define GB_FREE_ALL                     \
{                                       \
    GB_FREE_WORKSPACE ;                 \
    GB_phybix_free (C) ;                \
}

//------------------------------------------------------------------------------
// GB_AxB_dot4: compute C+=A'*B in-place
//------------------------------------------------------------------------------

GrB_Info GB_AxB_dot4                // C+=A'*B, dot product method
(
    GrB_Matrix C,                   // input/output matrix, must be as-if-full
    const GrB_Matrix A,             // input matrix
    const GrB_Matrix B,             // input matrix
    const GrB_Semiring semiring,    // semiring that defines C+=A*B and accum
    const bool flipxy,              // if true, do z=fmult(b,a) vs fmult(a,b)
    bool *done_in_place,            // if true, dot4 has computed the result
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT_MATRIX_OK (C, "C for dot in-place += A'*B", GB0) ;
    ASSERT_MATRIX_OK (A, "A for dot in-place += A'*B", GB0) ;
    ASSERT_MATRIX_OK (B, "B for dot in-place += A'*B", GB0) ;
    ASSERT (GB_IS_FULL (C)) ;
    ASSERT (!GB_ZOMBIES (C)) ;
    ASSERT (!GB_JUMBLED (C)) ;
    ASSERT (!GB_PENDING (C)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;
    ASSERT (!GB_ZOMBIES (B)) ;
    ASSERT (!GB_JUMBLED (B)) ;
    ASSERT (!GB_PENDING (B)) ;
    ASSERT_SEMIRING_OK (semiring, "semiring for in-place += A'*B", GB0) ;
    ASSERT (A->vlen == B->vlen) ;

    GB_WERK_DECLARE (A_slice, int64_t) ;
    GB_WERK_DECLARE (B_slice, int64_t) ;

    //--------------------------------------------------------------------------
    // get the semiring operators
    //--------------------------------------------------------------------------

    GrB_BinaryOp mult = semiring->multiply ;
    GrB_Monoid add = semiring->add ;
    ASSERT (mult->ztype == add->op->ztype) ;
    ASSERT (C->type     == add->op->ztype) ;

    bool op_is_first  = mult->opcode == GB_FIRST_binop_code ;
    bool op_is_second = mult->opcode == GB_SECOND_binop_code ;
    bool op_is_pair   = mult->opcode == GB_PAIR_binop_code ;
    bool A_is_pattern = false ;
    bool B_is_pattern = false ;

    if (flipxy)
    { 
        // z = fmult (b,a) will be computed
        A_is_pattern = op_is_first  || op_is_pair ;
        B_is_pattern = op_is_second || op_is_pair ;
        ASSERT (GB_IMPLIES (!A_is_pattern,
            GB_Type_compatible (A->type, mult->ytype))) ;
        ASSERT (GB_IMPLIES (!B_is_pattern,
            GB_Type_compatible (B->type, mult->xtype))) ;
    }
    else
    { 
        // z = fmult (a,b) will be computed
        A_is_pattern = op_is_second || op_is_pair ;
        B_is_pattern = op_is_first  || op_is_pair ;
        ASSERT (GB_IMPLIES (!A_is_pattern,
            GB_Type_compatible (A->type, mult->xtype))) ;
        ASSERT (GB_IMPLIES (!B_is_pattern,
            GB_Type_compatible (B->type, mult->ytype))) ;
    }

    GB_Opcode mult_binop_code, add_binop_code ;
    GB_Type_code xcode, ycode, zcode ;
    bool builtin_semiring = GB_AxB_semiring_builtin (A, A_is_pattern, B,
        B_is_pattern, semiring, flipxy, &mult_binop_code, &add_binop_code,
        &xcode, &ycode, &zcode) ;

    if (add_binop_code == GB_ANY_binop_code)
    { 
        // no work to do
        // future:: when the JIT is extended to handle the case when
        // accum != monoid->op, this case must be modified.
        return (GrB_NO_VALUE) ;
    }

    GBURBLE ("(dot4: %s += %s'*%s) ",
        GB_sparsity_char_matrix (C),
        GB_sparsity_char_matrix (A),
        GB_sparsity_char_matrix (B)) ;

    //--------------------------------------------------------------------------
    // determine the number of threads to use
    //--------------------------------------------------------------------------

    int64_t anz = GB_nnz_held (A) ;
    int64_t bnz = GB_nnz_held (B) ;
    int nthreads_max = GB_Context_nthreads_max ( ) ;
    double chunk = GB_Context_chunk ( ) ;
    int nthreads = GB_nthreads (anz + bnz, chunk, nthreads_max) ;

    //--------------------------------------------------------------------------
    // slice A and B
    //--------------------------------------------------------------------------

    // A and B can have any sparsity: sparse/hyper/bitmap/full.
    // C is always as-if-full.

    int64_t anvec = A->nvec ;
    int64_t vlen  = A->vlen ;
    int64_t bnvec = B->nvec ;
    int naslice, nbslice ;

    if (nthreads == 1)
    { 
        naslice = 1 ;
        nbslice = 1 ;
    }
    else
    {
        bool A_is_sparse_or_hyper = GB_IS_SPARSE (A) || GB_IS_HYPERSPARSE (A) ;
        bool B_is_sparse_or_hyper = GB_IS_SPARSE (B) || GB_IS_HYPERSPARSE (B) ;
        if (A_is_sparse_or_hyper && B_is_sparse_or_hyper)
        { 
            // both A and B are sparse/hyper; split them finely
            naslice = 16 * nthreads ;
            nbslice = 16 * nthreads ;
        }
        else if (!A_is_sparse_or_hyper && B_is_sparse_or_hyper)
        { 
            // A is bitmap/full and B is sparse/hyper; only split B
            naslice = 1 ;
            nbslice = 16 * nthreads ;
        }
        else if (A_is_sparse_or_hyper && !B_is_sparse_or_hyper)
        { 
            // A is sparse/hyper and B is bitmap/full; is only split A
            naslice = 16 * nthreads ;
            nbslice = 1 ;
        }
        else
        { 
            // A and B are bitmap/full; split them coarsely
            naslice = nthreads ;
            nbslice = nthreads ;
        }
    }

    // ensure each slice has at least one vector
    naslice = GB_IMIN (naslice, anvec) ;
    nbslice = GB_IMIN (nbslice, bnvec) ;

    GB_WERK_PUSH (A_slice, naslice + 1, int64_t) ;
    GB_WERK_PUSH (B_slice, nbslice + 1, int64_t) ;
    if (A_slice == NULL || B_slice == NULL)
    { 
        // out of memory
        GB_FREE_WORKSPACE ;
        return (GrB_OUT_OF_MEMORY) ;
    }
    GB_pslice (A_slice, A->p, anvec, naslice, false) ;
    GB_pslice (B_slice, B->p, bnvec, nbslice, false) ;

    //--------------------------------------------------------------------------
    // convert C to non-iso
    //--------------------------------------------------------------------------

    bool C_in_iso = C->iso ;
    bool initialized = GB_IS_HYPERSPARSE (A) || GB_IS_HYPERSPARSE (B) ;
    if (C_in_iso)
    { 
        // allocate but do not initialize C->x unless A or B are hypersparse.
        // The initialization must be done if dot4 doesn't do the work;
        // see GB_expand_iso below.
        GB_OK (GB_convert_any_to_non_iso (C, initialized)) ;
    }

    //--------------------------------------------------------------------------
    // via the factory kernel
    //--------------------------------------------------------------------------

    info = GrB_NO_VALUE ;
    #ifndef GBCOMPACT
    GB_IF_FACTORY_KERNELS_ENABLED
    { 

        //----------------------------------------------------------------------
        // define the worker for the switch factory
        //----------------------------------------------------------------------

        #define GB_Adot4B(add,mult,xname) GB (_Adot4B_ ## add ## mult ## xname)
        #define GB_AxB_WORKER(add,mult,xname)                               \
        {                                                                   \
            info = GB_Adot4B (add,mult,xname) (C, C_in_iso, A, B,           \
                A_slice, B_slice, naslice, nbslice, nthreads, Werk) ;       \
        }                                                                   \
        break ;

        //----------------------------------------------------------------------
        // launch the switch factory
        //----------------------------------------------------------------------

        // disabled the ANY monoid
        #define GB_NO_ANY_MONOID
        #include "GB_AxB_factory.c"
    }
    #endif

    //--------------------------------------------------------------------------
    // via the JIT or PreJIT kernel
    //--------------------------------------------------------------------------

    if (info == GrB_NO_VALUE)
    { 
        // C+= A*B, C is full
        info = GB_AxB_dot4_jit (C, C_in_iso, A, B, semiring,
            flipxy, A_slice, B_slice, naslice, nbslice, nthreads, Werk) ;
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    GB_FREE_WORKSPACE ;
    if (info == GrB_NO_VALUE)
    { 
        // dot4 doesn't handle this case; punt to dot2 or dot3
        if (C_in_iso && !initialized)
        { 
            // C has been expanded to non-iso, but dot4 didn't do the work,
            // and C has been left incompletely expanded to non-iso.
            // Need to copy the iso value in Cx [0] to all of Cx.
            size_t csize = C->type->size ;
            GB_void cscalar [GB_VLA(csize)] ;
            int64_t cnz = GB_nnz_held (C) ;
            memcpy (cscalar, C->x, csize) ;
            GB_expand_iso (C->x, cnz, cscalar, csize) ;
        }
        GBURBLE ("(punt) ") ;
    }
    else if (info == GrB_SUCCESS)
    { 
        ASSERT_MATRIX_OK (C, "dot4: output", GB0) ;
        (*done_in_place) = true ;
    }
    else
    { 
        // out of memory, or other error
        GB_FREE_ALL ;
    }
    return (info) ;
}

