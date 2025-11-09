//------------------------------------------------------------------------------
// GB_bitmap_subref: C = A(I,J) where A is bitmap or full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C=A(I,J), where A is bitmap or full, symbolic and numeric.

#include "extract/GB_subref.h"
#include "jitifyer/GB_stringify.h"
#include "include/GB_unused.h"

#define GB_FREE_WORKSPACE                               \
{                                                       \
    GB_FREE_MEMORY (&TaskList_IxJ, TaskList_IxJ_size) ;   \
}

#define GB_FREE_ALL                                     \
{                                                       \
    GB_FREE_WORKSPACE                                   \
    GB_phybix_free (C) ;                                \
}

GrB_Info GB_bitmap_subref       // C = A(I,J): either symbolic or numeric
(
    // output:
    GrB_Matrix C,               // output matrix, static header
    // inputs, not modified:
    const GrB_Type ctype,       // type of C to create
    const bool C_iso,           // if true, C is iso
    const GB_void *cscalar,     // scalar value of C, if iso
    const bool C_is_csc,        // requested format of C
    const GrB_Matrix A,
    const void *I,              // index list for C = A(I,J), or GrB_ALL, etc.
    const bool I_is_32,         // if true, I is 32-bit; else 64-bit
    const int64_t ni,           // length of I, or special
    const void *J,              // index list for C = A(I,J), or GrB_ALL, etc.
    const bool J_is_32,         // if true, J is 32-bit; else 64-bit
    const int64_t nj,           // length of J, or special
    const bool symbolic,        // if true, construct C as symbolic
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL && (C->header_size == 0 || GBNSTATIC)) ;
    ASSERT_MATRIX_OK (A, "A for C=A(I,J) bitmap subref", GB0) ;
    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A)) ;
    ASSERT (!GB_IS_SPARSE (A)) ;
    ASSERT (!GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // workspace for assign/template/GB_bitmap_assign_IxJ_template.c
    //--------------------------------------------------------------------------

    GB_task_struct *TaskList_IxJ = NULL ; size_t TaskList_IxJ_size = 0 ;
    int ntasks_IxJ = 0, nthreads_IxJ = 0 ;

    //--------------------------------------------------------------------------
    // get A
    //--------------------------------------------------------------------------

    const int8_t *restrict Ab = A->b ;
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;

    //--------------------------------------------------------------------------
    // check the properties of I and J
    //--------------------------------------------------------------------------

    // C = A(I,J) so I is in range 0:avlen-1 and J is in range 0:avdim-1
    int64_t nI, nJ, Icolon [3], Jcolon [3] ;
    int Ikind, Jkind ;
    GB_ijlength (I, I_is_32, ni, avlen, &nI, &Ikind, Icolon) ;
    GB_ijlength (J, J_is_32, nj, avdim, &nJ, &Jkind, Jcolon) ;

    bool I_unsorted, I_has_dupl, I_contig, J_unsorted, J_has_dupl, J_contig ;
    int64_t imin, imax, jmin, jmax ;

    info = GB_ijproperties (I, I_is_32, ni, nI, avlen, &Ikind, Icolon,
        &I_unsorted, &I_has_dupl, &I_contig, &imin, &imax, Werk) ;
    if (info != GrB_SUCCESS)
    { 
        // I invalid
        return (info) ;
    }

    info = GB_ijproperties (J, J_is_32, nj, nJ, avdim, &Jkind, Jcolon,
        &J_unsorted, &J_has_dupl, &J_contig, &jmin, &jmax, Werk) ;
    if (info != GrB_SUCCESS)
    { 
        // J invalid
        return (info) ;
    }

    GB_IDECL (I, const, u) ; GB_IPTR (I, I_is_32) ;
    GB_IDECL (J, const, u) ; GB_IPTR (J, J_is_32) ;

    #define GB_I_KIND Ikind
    #define GB_J_KIND Jkind
    #define GB_C_IS_BITMAP (sparsity == GxB_BITMAP)
    #define GB_C_IS_FULL   (sparsity == GxB_FULL)

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    int64_t cnzmax ;
    bool ok = GB_int64_multiply ((uint64_t *) (&cnzmax), nI, nJ) ;
    if (!ok) cnzmax = INT64_MAX ;
    int sparsity = GB_IS_BITMAP (A) ? GxB_BITMAP : GxB_FULL ;
    GB_OK (GB_new_bix (&C, // bitmap or full, existing header
        ctype, nI, nJ, GB_ph_null, C_is_csc,
        sparsity, true, A->hyper_switch, -1, cnzmax, true, C_iso,
        /* OK: */ false, false, false)) ;

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    int8_t *restrict Cb = C->b ;

    // In assign/template/GB_bitmap_assign_IxJ_template, vlen is the vector
    // length of the submatrix C(I,J), but here the template is used to access
    // A(I,J), and so the vector length is A->vlen, not C->vlen.  The pointers
    // pA and pC are swapped in GB_IXJ_WORK macro below, since C=A(I,J) is
    // being computed, instead of C(I,J)=A for the bitmap assignment.

    int64_t vlen = avlen ;

    //--------------------------------------------------------------------------
    // C = A(I,J)
    //--------------------------------------------------------------------------

    if (symbolic )
    {

        //----------------------------------------------------------------------
        // symbolic subref, for GB_subassign_symbolic
        //----------------------------------------------------------------------

        // symbolic subref is only used by GB_subassign_symbolic, which only
        // operates on a matrix that is hypersparse, sparse, or full, but not
        // bitmap.  As a result, the symbolic subref C=A(I,J) where both A and
        // C are bitmap is not needed.

        ASSERT (GB_C_IS_FULL) ;
        ASSERT (ctype == GrB_UINT32 || ctype == GrB_UINT64) ;

        // cnvals must be declared for the omp #pragma, but it is not used
        int64_t cnvals = 0 ;

        #undef  GB_IXJ_WORK
        #define GB_IXJ_WORK(pA,pC)  \
        {                           \
            Cx [pC] = pA ;          \
        }

        if (ctype == GrB_UINT32)
        { 
            // C=A(I,J) symbolic (32-bit) with A and C full
            uint32_t *restrict Cx = (uint32_t *) C->x ;
            #define GB_NO_CNVALS
            #include "assign/template/GB_bitmap_assign_IxJ_template.c"
            #undef  GB_NO_CNVALS
        }
        else
        { 
            // C=A(I,J) symbolic (64-bit) with A and C full
            uint64_t *restrict Cx = (uint64_t *) C->x ;
            #define GB_NO_CNVALS
            #include "assign/template/GB_bitmap_assign_IxJ_template.c"
            #undef  GB_NO_CNVALS
        }

    }
    else if (C_iso)
    {

        //----------------------------------------------------------------------
        // C=A(I,J) iso numeric with A and C bitmap/full
        //----------------------------------------------------------------------

        if (GB_C_IS_BITMAP)
        { 
            // iso case where C and A are bitmap
            int64_t cnvals = 0 ;
            memcpy (C->x, cscalar, ctype->size) ;
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pA,pC)                                      \
            {                                                               \
                int8_t ab = Ab [pA] ;                                       \
                Cb [pC] = ab ;                                              \
                task_cnvals += ab ;                                         \
            }
            #include "assign/template/GB_bitmap_assign_IxJ_template.c"
            C->nvals = cnvals ;
        }
        else
        { 
            // iso case where C and A are full
            memcpy (C->x, cscalar, ctype->size) ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // C=A(I,J) non-iso numeric with A and C bitmap/full
        //----------------------------------------------------------------------

        ASSERT (ctype == A->type) ;

        // via the JIT kernel
        info = GB_subref_bitmap_jit (C, A,
            I, I_is_32, nI, Ikind, Icolon,
            J, J_is_32, nJ, Jkind, Jcolon, Werk) ;

        // via the generic kernel
        if (info == GrB_NO_VALUE)
        { 
            // using the generic kernel
            GBURBLE ("(generic subref) ") ;
            const size_t csize = C->type->size ; // C and A have the same type
            const GB_void *restrict Ax = (GB_void *) A->x ;
                  GB_void *restrict Cx = (GB_void *) C->x ;
            #define GB_COPY_ENTRY(pC,pA)                                    \
                memcpy (Cx + (pC)*csize, Ax + (pA)*csize, csize) ;
            #include "extract/template/GB_bitmap_subref_template.c"
            info = GrB_SUCCESS ;
        }
    }

    GB_OK (info) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (C, "C output for bitmap subref C=A(I,J)", GB0) ;
    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

