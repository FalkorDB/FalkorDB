//------------------------------------------------------------------------------
// GB_bitmap_assign_2_template: C bitmap, M bitmap/full, no accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<M>(I,J) = A       assign
// C(I,J)<M> = A       subassign

// C<M,repl>(I,J) = A       assign
// C(I,J)<M,repl> = A       subassign

// C<!M>(I,J) = A       assign
// C(I,J)<!M> = A       subassign

// C<!M,repl>(I,J) = A       assign
// C(I,J)<!M,repl> = A       subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, bitmap or full (not hypersparse or sparse)
// Mask_comp:   true or false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       not present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign, row assign, col assign, or subassign

// If C were full: entries can be deleted if C_replace is true,
// or if A is not full and missing at least one entry.

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_ALL_FOR_BITMAP

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_A_SCALAR_FOR_BITMAP
    GB_GET_MASK

    //--------------------------------------------------------------------------
    // to get the effective value of the mask entry mij
    //--------------------------------------------------------------------------

    #undef  GB_GET_MIJ
    #define GB_GET_MIJ(mij,pM)                                  \
        bool mij = (GBB_M (Mb, pM) && GB_MCAST (Mx, pM, msize)) ^ GB_MASK_COMP ;

    //--------------------------------------------------------------------------
    // C_replace phase
    //--------------------------------------------------------------------------

    if (C_replace)
    { 
        // FUTURE: if C FULL: use two passes: first pass checks if any
        // entry must be deleted.  If none: do nothing.  Else:  change C
        // to bitmap and do 2nd pass as below.

        // for row assign: set Cb(i,:) to zero if mij == 0
        // for col assign: set Cb(:,j) to zero if mij == 0
        // for assign: set Cb(:,:) to zero if mij == 0
        // for subassign set Cb(I,J) to zero if mij == 0
        #undef  GB_CIJ_WORK
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
    // assignment phase
    //--------------------------------------------------------------------------

    if (GB_SCALAR_ASSIGN)
    {

        //----------------------------------------------------------------------
        // scalar assignment: C<M or !M>(I,J) = scalar
        //----------------------------------------------------------------------

        // FUTURE: if C FULL: Cb is effectively all 1's and stays that way

        // for all entries in IxJ
        #undef  GB_IXJ_WORK
        #define GB_IXJ_WORK(pC,pA)                      \
        {                                               \
            int64_t pM = GB_GET_pM ;                    \
            GB_GET_MIJ (mij, pM) ;                      \
            if (mij)                                    \
            {                                           \
                int8_t cb = Cb [pC] ;                   \
                /* Cx [pC] = scalar */                  \
                GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ; \
                Cb [pC] = 1 ;                           \
                task_cnvals += (cb == 0) ;              \
            }                                           \
        }

        ASSERT (GB_ASSIGN_KIND == GB_ASSIGN || GB_ASSIGN_KIND == GB_SUBASSIGN) ;

        switch (GB_ASSIGN_KIND)
        {
            case GB_ASSIGN : 
                // C<M>(I,J) = scalar where M has the same size as C
                #undef  GB_GET_pM
                #define GB_GET_pM pC
                #include "template/GB_bitmap_assign_IxJ_template.c"
                break ;
            case GB_SUBASSIGN : 
                // C(I,J)<M> = scalar where M has the same size as A
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
        // matrix assignment: C<M or !M>(I,J) = A
        //----------------------------------------------------------------------

        // assign A into C:

            //  for all entries aij in A
            //      get the effective value of the mask:
            //          for row assign: get mij = m(jC,0)
            //          for col assign: get mij = m(iC,0)
            //          for assign: get mij = M(iC,jC)
            //          for subassign: get mij = M(i,j)
            //          if complemented: mij = !mij
            //      if mij == 1:
            //          Cx(p) = aij     // C(iC,jC) inserted or updated
            //          Cb(p) = 4

        // clear entries from C that were not in A:

            // for all entries in IxJ
                // get the effective value of the mask
                // if mij == 1
                    // 0 -> 0
                    // 1 -> 0           delete because aij not present
                    // 4 -> 1

        // TODO: if A is bitmap or full, use a single pass

        #define GB_AIJ_WORK(pC,pA)                                  \
        {                                                           \
            int64_t pM = GB_GET_pM ;                                \
            GB_GET_MIJ (mij, pM) ;                                  \
            if (mij)                                                \
            {                                                       \
                int8_t cb = Cb [pC] ;                               \
                /* Cx [pC] = Ax [pA] */                             \
                GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork, C_iso) ;   \
                Cb [pC] = 4 ;                                       \
                task_cnvals += (cb == 0) ;                          \
            }                                                       \
        }

        #undef  GB_IXJ_WORK
        #define GB_IXJ_WORK(pC,pA)          \
        {                                   \
            int64_t pM = GB_GET_pM ;        \
            GB_GET_MIJ (mij, pM) ;          \
            if (mij)                        \
            {                               \
                int8_t cb = Cb [pC] ;       \
                Cb [pC] = (cb > 1) ;        \
                task_cnvals -= (cb == 1) ;  \
            }                               \
        }

        switch (GB_ASSIGN_KIND)
        {
            case GB_ROW_ASSIGN : 
                // C<m>(i,J) = A where m is a 1-by-C->vdim row vector
                #undef  GB_GET_pM
                #define GB_GET_pM jC
                #include "template/GB_bitmap_assign_A_template.c"
                #include "template/GB_bitmap_assign_IxJ_template.c"
                break ;

            case GB_COL_ASSIGN : 
                // C<m>(I,j) = A where m is a C->vlen-by-1 column vector
                #undef  GB_GET_pM
                #define GB_GET_pM iC
                #include "template/GB_bitmap_assign_A_template.c"
                #include "template/GB_bitmap_assign_IxJ_template.c"
                break ;

            case GB_ASSIGN : 
                // C<M>(I,J) = A where M has the same size as C
                #undef  GB_GET_pM
                #define GB_GET_pM pC
                #include "template/GB_bitmap_assign_A_template.c"
                #include "template/GB_bitmap_assign_IxJ_template.c"
                break ;

            case GB_SUBASSIGN : 
                // C(I,J)<M> = A where M has the same size as A
                #undef  GB_GET_pM
                #define GB_GET_pM (iA + jA * nI)
                #include "template/GB_bitmap_assign_A_template.c"
                #include "template/GB_bitmap_assign_IxJ_template.c"
                break ;

            default: ;
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "final C for bitmap assign, M full, noaccum", GB0) ;
    return (GrB_SUCCESS) ;
}

