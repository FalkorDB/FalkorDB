//------------------------------------------------------------------------------
// GB_bitmap_assign_2_whole_template: C bitmap, M bitmap/full, no accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<M> = A       assign
// C<M> = A       subassign

// C<M,repl> = A       assign
// C<M,repl> = A       subassign

// C<!M> = A       assign
// C<!M> = A       subassign

// C<!M,repl> = A       assign
// C<!M,repl> = A       subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, bitmap or full (not hypersparse or sparse)
// Mask_comp:   true or false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       not present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign or subassign (same action)

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

    #define GB_GET_MIJ(mij,pM)                                  \
        bool mij = (GBB_M (Mb, pM) && GB_MCAST (Mx, pM, msize)) ^ GB_MASK_COMP ;

    //--------------------------------------------------------------------------
    // assignment phase
    //--------------------------------------------------------------------------

    if (GB_SCALAR_ASSIGN)
    {

        //----------------------------------------------------------------------
        // scalar assignment: C<M or !M> = scalar
        //----------------------------------------------------------------------

        if (C_replace)
        { 

            //------------------------------------------------------------------
            // C<M or !M, replace> = scalar
            //------------------------------------------------------------------

            #undef  GB_CIJ_WORK
            #define GB_CIJ_WORK(pC)                         \
            {                                               \
                int8_t cb = Cb [pC] ;                       \
                if (mij)                                    \
                {                                           \
                    /* Cx [pC] = scalar */                  \
                    GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ; \
                    Cb [pC] = 1 ;                           \
                    task_cnvals += (cb == 0) ;              \
                }                                           \
                else                                        \
                {                                           \
                    /* delete C(i,j) if present */          \
                    Cb [pC] = 0 ;                           \
                    task_cnvals -= (cb == 1) ;              \
                }                                           \
            }
            #include "template/GB_bitmap_assign_C_whole_template.c"

        }
        else
        { 

            //------------------------------------------------------------------
            // C<M or !M> = scalar
            //------------------------------------------------------------------

            #undef  GB_CIJ_WORK
            #define GB_CIJ_WORK(pC)                         \
            {                                               \
                if (mij)                                    \
                {                                           \
                    /* Cx [pC] = scalar */                  \
                    int8_t cb = Cb [pC] ;                   \
                    GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ; \
                    Cb [pC] = 1 ;                           \
                    task_cnvals += (cb == 0) ;              \
                }                                           \
            }
            #include "template/GB_bitmap_assign_C_whole_template.c"
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // matrix assignment: C<M or !M> = A
        //----------------------------------------------------------------------

        if (GB_IS_BITMAP (A) || GB_IS_FULL (A))
        {

            //------------------------------------------------------------------
            // matrix assignment: C<M or !M> = A where A is bitmap or full
            //------------------------------------------------------------------

            if (C_replace)
            { 

                //--------------------------------------------------------------
                // C<M or !M,replace> = A where A is bitmap or full
                //--------------------------------------------------------------

                #undef  GB_CIJ_WORK
                #define GB_CIJ_WORK(pC)                                     \
                {                                                           \
                    int8_t cb = Cb [pC] ;                                   \
                    if (mij && GBB_A (Ab, pC))                              \
                    {                                                       \
                        /* Cx [pC] = Ax [pC] */                             \
                        GB_COPY_aij_to_C (Cx,pC,Ax,pC,A_iso,cwork,C_iso) ;  \
                        Cb [pC] = 1 ;                                       \
                        task_cnvals += (cb == 0) ;                          \
                    }                                                       \
                    else                                                    \
                    {                                                       \
                        /* delete C(i,j) if present */                      \
                        Cb [pC] = 0 ;                                       \
                        task_cnvals -= (cb == 1) ;                          \
                    }                                                       \
                }
                #include "template/GB_bitmap_assign_C_whole_template.c"

            }
            else
            { 

                //--------------------------------------------------------------
                // C<M or !M> = A where A is bitmap or full
                //--------------------------------------------------------------

                #undef  GB_CIJ_WORK
                #define GB_CIJ_WORK(pC)                                       \
                {                                                             \
                    if (mij)                                                  \
                    {                                                         \
                        int8_t cb = Cb [pC] ;                                 \
                        if (GBB_A (Ab, pC))                                   \
                        {                                                     \
                            /* Cx [pC] = Ax [pC] */                           \
                            GB_COPY_aij_to_C (Cx,pC,Ax,pC,A_iso,cwork,C_iso) ;\
                            Cb [pC] = 1 ;                                     \
                            task_cnvals += (cb == 0) ;                        \
                        }                                                     \
                        else                                                  \
                        {                                                     \
                            /* delete C(i,j) if present */                    \
                            Cb [pC] = 0 ;                                     \
                            task_cnvals -= (cb == 1) ;                        \
                        }                                                     \
                    }                                                         \
                }
                #include "template/GB_bitmap_assign_C_whole_template.c"
            }
        }
        else
        {

            //------------------------------------------------------------------
            // matrix assignment: C<M or !M> = A where A is sparse or hyper
            //------------------------------------------------------------------

            if (C_replace)
            { 

                //--------------------------------------------------------------
                // C<M or !M,replace> = A where A is sparse or hyper
                //--------------------------------------------------------------

                // clear C of all entries
                cnvals = 0 ;
                GB_memset (Cb, 0, cnzmax, nthreads_max) ;

                // C<M or !M> = A
                #undef  GB_AIJ_WORK
                #define GB_AIJ_WORK(pC,pA)                                  \
                {                                                           \
                    GB_GET_MIJ (mij, pC) ;                                  \
                    if (mij)                                                \
                    {                                                       \
                        /* Cx [pC] = Ax [pA] */                             \
                        GB_COPY_aij_to_C (Cx,pC,Ax,pA,A_iso,cwork,C_iso) ;  \
                        Cb [pC] = 1 ;                                       \
                        task_cnvals++ ;                                     \
                    }                                                       \
                }
                #include "template/GB_bitmap_assign_A_whole_template.c"

            }
            else
            { 

                //--------------------------------------------------------------
                // C<M or !M> = A where A is sparse or hyper
                //--------------------------------------------------------------

                // C<M or !M> = A, assign entries from A
                #undef  GB_AIJ_WORK
                #define GB_AIJ_WORK(pC,pA)                                  \
                {                                                           \
                    GB_GET_MIJ (mij, pC) ;                                  \
                    if (mij)                                                \
                    {                                                       \
                        /* Cx [pC] = Ax [pA] */                             \
                        int8_t cb = Cb [pC] ;                               \
                        GB_COPY_aij_to_C (Cx,pC,Ax,pA,A_iso,cwork,C_iso) ;  \
                        Cb [pC] = 4 ; /* keep this entry */                 \
                        task_cnvals += (cb == 0) ;                          \
                    }                                                       \
                }
                #include "template/GB_bitmap_assign_A_whole_template.c"

                // delete entries where M(i,j)=1 but not assigned by A
                #undef  GB_CIJ_WORK
                #define GB_CIJ_WORK(pC)                     \
                {                                           \
                    int8_t cb = Cb [pC] ;                   \
                    if (mij)                                \
                    {                                       \
                        Cb [pC] = (cb == 4) ;               \
                        task_cnvals -= (cb == 1) ;          \
                    }                                       \
                }
                #include "template/GB_bitmap_assign_C_whole_template.c"
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "final C bitmap assign, M full, noaccum, whole", GB0) ;
    return (GrB_SUCCESS) ;
}

