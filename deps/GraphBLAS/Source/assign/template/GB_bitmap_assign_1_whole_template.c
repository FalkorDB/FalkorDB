//------------------------------------------------------------------------------
// GB_bitmap_assign_1_whole_template: C bitmap, M bitmap/full, with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2025, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<M> += A           assign
// C<M> += A           subassign

// C<M,repl> += A      assign
// C<M,repl> += A      subassign

// C<!M> += A          assign
// C<!M> += A          subassign

// C<!M,repl> += A     assign
// C<!M,repl> += A     subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, bitmap or full (not hypersparse or sparse)
// Mask_comp:   true or false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign or subassign (same action)

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

    #define GB_GET_MIJ(mij,pC)                                  \
        bool mij = (GBb_M (Mb, pC) && GB_MCAST (Mx, pC, msize)) ^ GB_MASK_COMP ;

    //--------------------------------------------------------------------------
    // slice
    //--------------------------------------------------------------------------

    //--------------------------------------------------------------------------
    // assignment phase
    //--------------------------------------------------------------------------

    if (GB_SCALAR_ASSIGN)
    {

        //----------------------------------------------------------------------
        // scalar assignment: C<M or !M> += scalar
        //----------------------------------------------------------------------

        if (C_replace)
        { 

            //------------------------------------------------------------------
            // C<M,replace> += scalar
            //------------------------------------------------------------------

            #undef  GB_CIJ_WORK
            #define GB_CIJ_WORK(pC)                                     \
            {                                                           \
                int8_t cb = Cb [pC] ;                                   \
                if (mij)                                                \
                {                                                       \
                    if (cb == 0)                                        \
                    {                                                   \
                        /* Cx [pC] = scalar */                          \
                        GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ;     \
                        Cb [pC] = 1 ;                                   \
                        task_cnvals++ ;                                 \
                    }                                                   \
                    else /* (cb == 1) */                                \
                    {                                                   \
                        /* Cx [pC] += scalar */                         \
                        GB_ACCUMULATE_scalar (Cx, pC, ywork, C_iso) ;   \
                    }                                                   \
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

            //------------------------------------------------------------------
            // C<M> += scalar
            //------------------------------------------------------------------

            #undef  GB_CIJ_WORK
            #define GB_CIJ_WORK(pC)                                     \
            {                                                           \
                if (mij)                                                \
                {                                                       \
                    if (Cb [pC])                                        \
                    {                                                   \
                        /* Cx [pC] += scalar */                         \
                        GB_ACCUMULATE_scalar (Cx, pC, ywork, C_iso) ;   \
                    }                                                   \
                    else                                                \
                    {                                                   \
                        /* Cx [pC] = scalar */                          \
                        GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ;     \
                        Cb [pC] = 1 ;                                   \
                        task_cnvals++ ;                                 \
                    }                                                   \
                }                                                       \
            }
            #include "template/GB_bitmap_assign_C_whole_template.c"
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // matrix assignment: C<M or !M> += A
        //----------------------------------------------------------------------

        if (GB_IS_BITMAP (A) || GB_IS_FULL (A))
        {
            if (C_replace)
            { 

                //--------------------------------------------------------------
                // C<M or !M,replace> += A where A is bitmap or full
                //--------------------------------------------------------------

                #undef  GB_CIJ_WORK
                #define GB_CIJ_WORK(pC)                                     \
                {                                                           \
                    int8_t cb = Cb [pC] ;                                   \
                    if (mij)                                                \
                    {                                                       \
                        if (GBb_A (Ab, pC))                                 \
                        {                                                   \
                            /* mij true and A(i,j) present */               \
                            if (cb)                                         \
                            {                                               \
                                /* Cx [pC] += Ax [pC] */                    \
                                GB_ACCUMULATE_aij (Cx,pC,Ax,pC,A_iso,ywork, \
                                    C_iso) ;                                \
                            }                                               \
                            else                                            \
                            {                                               \
                                /* Cx [pC] = Ax [pC] */                     \
                                GB_COPY_aij_to_C (Cx,pC,Ax,pC,A_iso,cwork,  \
                                   C_iso) ;                                 \
                                Cb [pC] = 1 ;                               \
                                task_cnvals++ ;                             \
                            }                                               \
                        }                                                   \
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
                // C<M or !M> += A where A is bitmap or full
                //--------------------------------------------------------------

                #undef  GB_CIJ_WORK
                #define GB_CIJ_WORK(pC)                                     \
                {                                                           \
                    if (mij && GBb_A (Ab, pC))                              \
                    {                                                       \
                        /* mij true and A(i,j) present */                   \
                        if (Cb [pC])                                        \
                        {                                                   \
                            /* Cx [pC] += Ax [pC] */                        \
                            GB_ACCUMULATE_aij (Cx,pC,Ax,pC,A_iso,ywork,C_iso) ;\
                        }                                                   \
                        else                                                \
                        {                                                   \
                            /* Cx [pC] = Ax [pC] */                         \
                            GB_COPY_aij_to_C (Cx,pC,Ax,pC,A_iso,cwork,C_iso) ;\
                            Cb [pC] = 1 ;                                   \
                            task_cnvals++ ;                                 \
                        }                                                   \
                    }                                                       \
                }
                #include "template/GB_bitmap_assign_C_whole_template.c"
            }

        }
        else
        {

            //------------------------------------------------------------------
            // C<M or !M,replace or !replace> += A, where A is sparse/hyper
            //------------------------------------------------------------------

            // assign entries from A
            #undef  GB_AIJ_WORK
            #define GB_AIJ_WORK(pC,pA)                                      \
            {                                                               \
                GB_GET_MIJ (mij, pC) ;                                      \
                if (mij)                                                    \
                {                                                           \
                    /* mij true and A(i,j) present */                       \
                    if (Cb [pC])                                            \
                    {                                                       \
                        /* Cx [pC] += Ax [pA] */                            \
                        GB_ACCUMULATE_aij (Cx,pC,Ax,pA,A_iso,ywork,C_iso) ; \
                    }                                                       \
                    else                                                    \
                    {                                                       \
                        /* Cx [pC] = Ax [pA] */                             \
                        GB_COPY_aij_to_C (Cx,pC,Ax,pA,A_iso,cwork,C_iso) ;  \
                        Cb [pC] = 1 ;                                       \
                        task_cnvals++ ;                                     \
                    }                                                       \
                }                                                           \
            }
            #include "template/GB_bitmap_assign_A_whole_template.c"

            // clear the mask and delete entries not assigned
            if (C_replace)
            { 
                #undef  GB_CIJ_WORK
                #define GB_CIJ_WORK(pC)                 \
                {                                       \
                    if (!mij)                           \
                    {                                   \
                        /* delete C(i,j) if present */  \
                        int8_t cb = Cb [pC] ;           \
                        Cb [pC] = 0 ;                   \
                        task_cnvals -= (cb == 1) ;      \
                    }                                   \
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
    ASSERT_MATRIX_OK (C, "final C, bitmap assign, M full, accum, whole", GB0) ;
    return (GrB_SUCCESS) ;
}

