//------------------------------------------------------------------------------
// GB_bitmap_assign_1_template: C bitmap, M bitmap/full, with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<M>(I,J) += A           assign
// C(I,J)<M> += A           subassign

// C<M,repl>(I,J) += A      assign
// C(I,J)<M,repl> += A      subassign

// C<!M>(I,J) += A          assign
// C(I,J)<!M> += A          subassign

// C<!M,repl>(I,J) += A     assign
// C(I,J)<!M,repl> += A     subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, bitmap or full (not hypersparse or sparse)
// Mask_comp:   true or false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign, row assign, col assign, or subassign

// If C were full: entries can be deleted only if C_replace is true.

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_ALL_FOR_BITMAP

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_A_SCALAR_FOR_BITMAP
    GB_GET_MASK
    GB_GET_ACCUM_FOR_BITMAP

    //--------------------------------------------------------------------------
    // to get the effective value of the mask entry mij
    //--------------------------------------------------------------------------

    #define GB_GET_MIJ(mij,pM)                                  \
        bool mij = (GBB_M (Mb, pM) && GB_MCAST (Mx, pM, msize)) ^ GB_MASK_COMP ;

    //--------------------------------------------------------------------------
    // assignment phase
    //--------------------------------------------------------------------------

    if (GB_SCALAR_ASSIGN)
    {

        //----------------------------------------------------------------------
        // scalar assignment: C<M or !M>(I,J) += scalar
        //----------------------------------------------------------------------

        // for all IxJ
        //  get the effective value of the mask, via GB_GET_MIJ:
        //      for row assign: get mij = m(jC,0)
        //      for col assign: get mij = m(iC,0)
        //      for assign: get mij = M(iC,jC)
        //      for subassign: get mij = M(i,j)
        //      if complemented: mij = !mij
        //  if mij == 1:
        //      if Cb(p) == 0
        //          Cx(p) = scalar
        //          Cb(p) = 1       // C(iC,jC) is now present, insert
        //      else // if Cb(p) == 1:
        //          Cx(p) += scalar // C(iC,jC) still present, updated

        // FUTURE: if C FULL: Cb is effectively all 1's and stays that way

        #undef  GB_IXJ_WORK
        #define GB_IXJ_WORK(pC,pA)                          \
        {                                                   \
            int64_t pM = GB_GET_pM ;                        \
            GB_GET_MIJ (mij, pM) ;                          \
            if (mij)                                        \
            {                                               \
                int8_t cb = Cb [pC] ;                       \
                if (cb == 0)                                \
                {                                           \
                    /* Cx [pC] = scalar */                  \
                    GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ;   \
                    Cb [pC] = 1 ;                           \
                    task_cnvals++ ;                         \
                }                                           \
                else /* (cb == 1) */                        \
                {                                           \
                    /* Cx [pC] += scalar */                 \
                    GB_ACCUMULATE_scalar (Cx, pC, ywork, C_iso) ;  \
                }                                           \
            }                                               \
        }

        ASSERT (GB_ASSIGN_KIND == GB_ASSIGN || GB_ASSIGN_KIND == GB_SUBASSIGN) ;

        switch (GB_ASSIGN_KIND)
        {
            case GB_ASSIGN : 
                // C<M>(I,J) += scalar where M has the same size as C
                #undef  GB_GET_pM
                #define GB_GET_pM pC
                #include "template/GB_bitmap_assign_IxJ_template.c"
                break ;
            case GB_SUBASSIGN : 
                // C(I,J)<M> += scalar where M has the same size as A
                #undef  GB_GET_pM
                #define GB_GET_pM pA
                #include "template/GB_bitmap_assign_IxJ_template.c"
                break ;
            default: ;
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // matrix assignment: C<M or !M>(I,J) += A
        //----------------------------------------------------------------------

        // for all entries aij in A (A can be hyper, sparse, bitmap, or full)
        //     get the effective value of the mask, via GB_GET_MIJ:
        //         for row assign: get mij = m(jC,0)
        //         for col assign: get mij = m(iC,0)
        //         for assign: get mij = M(iC,jC)
        //         for subassign: get mij = M(i,j)
        //         if complemented: mij = !mij
        //     if mij == 1:
        //         if Cb(p) == 0
        //             Cx(p) = aij
        //             Cb(p) = 1       // C(iC,jC) is now present, insert
        //             task_cnvals++
        //         else // if Cb(p) == 1:
        //             Cx(p) += aij    // C(iC,jC) still present, updated

        // FUTURE: if C FULL: Cb is effectively all 1's and stays that way

        #define GB_AIJ_WORK(pC,pA)                                      \
        {                                                               \
            int64_t pM = GB_GET_pM ;                                    \
            GB_GET_MIJ (mij, pM) ;                                      \
            if (mij)                                                    \
            {                                                           \
                int8_t cb = Cb [pC] ;                                   \
                if (cb == 0)                                            \
                {                                                       \
                    /* Cx [pC] = Ax [pA] */                             \
                    GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork, C_iso) ; \
                    Cb [pC] = 1 ;                                       \
                    task_cnvals++ ;                                     \
                }                                                       \
                else /* (cb == 1) */                                    \
                {                                                       \
                    /* Cx [pC] += Ax [pA] */                            \
                    GB_ACCUMULATE_aij (Cx, pC, Ax, pA, A_iso, ywork, C_iso) ;  \
                }                                                       \
            }                                                           \
        }

        switch (GB_ASSIGN_KIND)
        {
            case GB_ROW_ASSIGN : 
                // C<m>(i,J) += A where m is a 1-by-C->vdim row vector
                #undef  GB_GET_pM
                #define GB_GET_pM jC
                #include "template/GB_bitmap_assign_A_template.c"
                break ;
            case GB_COL_ASSIGN : 
                // C<m>(I,j) += A where m is a C->vlen-by-1 column vector
                #undef  GB_GET_pM
                #define GB_GET_pM iC
                #include "template/GB_bitmap_assign_A_template.c"
                break ;
            case GB_ASSIGN : 
                // C<M>(I,J) += A where M has the same size as C
                #undef  GB_GET_pM
                #define GB_GET_pM pC
                #include "template/GB_bitmap_assign_A_template.c"
                break ;
            case GB_SUBASSIGN : 
                // C(I,J)<M> += A where M has the same size as A
                #undef  GB_GET_pM
                #define GB_GET_pM (iA + jA * nI)
                #include "template/GB_bitmap_assign_A_template.c"
                break ;
            default: ;
        }
    }

    //--------------------------------------------------------------------------
    // C_replace phase
    //--------------------------------------------------------------------------

    if (C_replace)
    { 
        // FUTURE: if C FULL: use two passes: first pass checks if any
        // entry must be deleted.  If none: do nothing.  Else:  change C
        // to bitmap and do 2nd pass as below.

        // for row assign: for all entries in C(i,:)
        // for col assign: for all entries in C(:,j)
        // for assign: for all entries in C(:,:)
        // for subassign: for all entries in C(I,J)
        //      get effective value mij of the mask via GB_GET_MIJ
        //      if mij == 0 set Cb(p) = 0
        #define GB_CIJ_WORK(pC)             \
        {                                   \
            if (!mij)                       \
            {                               \
                int8_t cb = Cb [pC] ;       \
                Cb [pC] = 0 ;               \
                task_cnvals -= (cb == 1) ;  \
            }                               \
        }
        #include "template/GB_bitmap_assign_C_template.c"
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    ASSERT_MATRIX_OK (C, "final C for bitmap assign, M full, accum", GB0) ;
    GB_FREE_ALL ;
    return (GrB_SUCCESS) ;
}

