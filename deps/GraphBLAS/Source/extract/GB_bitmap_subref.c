//------------------------------------------------------------------------------
// GB_bitmap_subref: C = A(I,J) where A is bitmap or full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// C=A(I,J), where A is bitmap or full, symbolic and numeric.

#include "extract/GB_subref.h"
#include "jitifyer/GB_stringify.h"

#define GB_FREE_WORKSPACE                               \
{                                                       \
    GB_FREE_WORK (&TaskList_IxJ, TaskList_IxJ_size) ;   \
}

#define GB_FREE_ALL                                     \
{                                                       \
    GB_FREE_WORKSPACE                                   \
    GB_phybix_free (C) ;                                \
}

GrB_Info GB_bitmap_subref       // C = A(I,J): either symbolic or numeric
(
    // output
    GrB_Matrix C,               // output matrix, static header
    // input, not modified
    const bool C_iso,           // if true, C is iso
    const GB_void *cscalar,     // scalar value of C, if iso
    const bool C_is_csc,        // requested format of C
    const GrB_Matrix A,
    const GrB_Index *I,         // index list for C = A(I,J), or GrB_ALL, etc.
    const int64_t ni,           // length of I, or special
    const GrB_Index *J,         // index list for C = A(I,J), or GrB_ALL, etc.
    const int64_t nj,           // length of J, or special
    const bool symbolic,        // if true, construct C as symbolic
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    ASSERT (C != NULL && (C->static_header || GBNSTATIC)) ;
    ASSERT_MATRIX_OK (A, "A for C=A(I,J) bitmap subref", GB0) ;
    ASSERT (GB_IS_BITMAP (A) || GB_IS_FULL (A)) ;
    ASSERT (!GB_IS_SPARSE (A)) ;
    ASSERT (!GB_IS_HYPERSPARSE (A)) ;
    ASSERT (!GB_ZOMBIES (A)) ;
    ASSERT (!GB_JUMBLED (A)) ;
    ASSERT (!GB_PENDING (A)) ;

    //--------------------------------------------------------------------------
    // workspace for GB_bitmap_assign_IxJ_template.c
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
    GB_ijlength (I, ni, avlen, &nI, &Ikind, Icolon) ;
    GB_ijlength (J, nj, avdim, &nJ, &Jkind, Jcolon) ;

    bool I_unsorted, I_has_dupl, I_contig, J_unsorted, J_has_dupl, J_contig ;
    int64_t imin, imax, jmin, jmax ;

    info = GB_ijproperties (I, ni, nI, avlen, &Ikind, Icolon,
        &I_unsorted, &I_has_dupl, &I_contig, &imin, &imax, Werk) ;
    if (info != GrB_SUCCESS)
    { 
        // I invalid
        return (info) ;
    }

    info = GB_ijproperties (J, nj, nJ, avdim, &Jkind, Jcolon,
        &J_unsorted, &J_has_dupl, &J_contig, &jmin, &jmax, Werk) ;
    if (info != GrB_SUCCESS)
    { 
        // J invalid
        return (info) ;
    }

    #define GB_I_KIND Ikind
    #define GB_J_KIND Jkind
    #define GB_C_IS_BITMAP (sparsity == GxB_BITMAP)
    #define GB_C_IS_FULL   (sparsity == GxB_FULL)

    //--------------------------------------------------------------------------
    // allocate C
    //--------------------------------------------------------------------------

    int64_t cnzmax ;
    bool ok = GB_int64_multiply ((GrB_Index *) (&cnzmax), nI, nJ) ;
    if (!ok) cnzmax = INT64_MAX ;
    GrB_Type ctype = symbolic ? GrB_INT64 : A->type ;
    int sparsity = GB_IS_BITMAP (A) ? GxB_BITMAP : GxB_FULL ;
    // set C->iso = C_iso   OK
    GB_OK (GB_new_bix (&C, // bitmap or full, existing header
        ctype, nI, nJ, GB_Ap_null, C_is_csc,
        sparsity, true, A->hyper_switch, -1, cnzmax, true, C_iso)) ;

    //--------------------------------------------------------------------------
    // get C
    //--------------------------------------------------------------------------

    int8_t *restrict Cb = C->b ;

    // In GB_bitmap_assign_IxJ_template, vlen is the vector length of the
    // submatrix C(I,J), but here the template is used to access A(I,J), and so
    // the vector length is A->vlen, not C->vlen.  The pointers pA and pC are
    // swapped in GB_IXJ_WORK macro below, since C=A(I,J) is being computed,
    // instead of C(I,J)=A for the bitmap assignment.

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
        // C are bitmap is not needed.  The code is left here in case it is
        // needed in the future.

        ASSERT (GB_C_IS_FULL) ;

        int64_t cnvals = 0 ;
        #if 0
        if (GB_C_IS_BITMAP)
        {
            // C=A(I,J) symbolic with A and C bitmap
            ASSERT (GB_DEAD_CODE) ;
            int64_t *restrict Cx = (int64_t *) C->x ;
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pA,pC)                                      \
            {                                                               \
                int8_t ab = Ab [pA] ;                                       \
                Cb [pC] = ab ;                                              \
                Cx [pC] = pA ;                                              \
                task_cnvals += ab ;                                         \
            }
            #include "template/GB_bitmap_assign_IxJ_template.c"
        }
        else
        #endif
        { 
            // C=A(I,J) symbolic with A and C full
            int64_t *restrict Cx = (int64_t *) C->x ;
            #undef  GB_IXJ_WORK
            #define GB_IXJ_WORK(pA,pC)                                      \
            {                                                               \
                Cx [pC] = pA ;                                              \
            }
            #define GB_NO_CNVALS
            #include "template/GB_bitmap_assign_IxJ_template.c"
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
            #include "template/GB_bitmap_assign_IxJ_template.c"
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

        // via the JIT kernel
        info = GB_subref_bitmap_jit (C, A,
            I, nI, Ikind, Icolon, J, nJ, Jkind, Jcolon, Werk) ;

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

    if (info != GrB_SUCCESS)
    { 
        // out of memory or JIT kernel failed
        GB_FREE_ALL ;
        return (info) ;
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->magic = GB_MAGIC ;
    ASSERT_MATRIX_OK (C, "C output for bitmap subref C=A(I,J)", GB0) ;
    GB_FREE_WORKSPACE ;
    return (GrB_SUCCESS) ;
}

