//------------------------------------------------------------------------------
// GB_bitmap_assign_7_whole_template: C bitmap, !M sparse/hyper, with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<!M> += A       assign
// C<!M> += A       subassign

// C<!M,repl> += A       assign
// C<!M,repl> += A       subassign
//------------------------------------------------------------------------------

// C:           bitmap
// M:           present, hypersparse or sparse (not bitmap or full)
// Mask_comp:   true
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
    GB_SLICE_M_FOR_BITMAP
    GB_GET_ACCUM_FOR_BITMAP

    //--------------------------------------------------------------------------
    // scatter M
    //--------------------------------------------------------------------------

    // Cb [pC] += 2 for each entry M(i,j) in the mask
    GB_bitmap_M_scatter_whole (C, M, GB_MASK_STRUCT,
        GB_BITMAP_M_SCATTER_PLUS_2, M_ek_slicing, M_ntasks, M_nthreads) ;
    // the bitmap of C now contains:
    //  Cb (i,j) = 0:   cij not present, mij zero
    //  Cb (i,j) = 1:   cij present, mij zero
    //  Cb (i,j) = 2:   cij not present, mij 1
    //  Cb (i,j) = 3:   cij present, mij 1

    //--------------------------------------------------------------------------
    // do the assignment
    //--------------------------------------------------------------------------

    if (GB_SCALAR_ASSIGN)
    {

        //----------------------------------------------------------------------
        // scalar assignment: C<!M, replace or !replace> += scalar
        //----------------------------------------------------------------------

        if (C_replace)
        { 

            //------------------------------------------------------------------
            // C<!M,replace> += scalar
            //------------------------------------------------------------------

            #undef  GB_CIJ_WORK
            #define GB_CIJ_WORK(pC)                                 \
            {                                                       \
                switch (Cb [pC])                                    \
                {                                                   \
                    case 0: /* C(i,j) not present, !M(i,j) = 1 */   \
                        /* Cx [pC] = scalar */                      \
                        GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ; \
                        Cb [pC] = 1 ;                               \
                        task_cnvals++ ;                             \
                        break ;                                     \
                    case 1: /* C(i,j) present, !M(i,j) = 1 */       \
                        /* Cx [pC] += scalar */                     \
                        GB_ACCUMULATE_scalar (Cx, pC, ywork, C_iso) ;      \
                        break ;                                     \
                    case 2: /* C(i,j) not present, !M(i,j) = 0 */   \
                        /* clear the mask from C */                 \
                        Cb [pC] = 0 ;                               \
                        break ;                                     \
                    case 3: /* C(i,j) present, !M(i,j) = 0 */       \
                        /* delete this entry */                     \
                        Cb [pC] = 0 ;                               \
                        task_cnvals-- ;                             \
                        break ;                                     \
                    default: ;                                      \
                }                                                   \
            }
            #include "template/GB_bitmap_assign_C_whole_template.c"

        }
        else
        { 

            //------------------------------------------------------------------
            // C<!M> += scalar
            //------------------------------------------------------------------

            #undef  GB_CIJ_WORK
            #define GB_CIJ_WORK(pC)                                 \
            {                                                       \
                switch (Cb [pC])                                    \
                {                                                   \
                    case 0: /* C(i,j) not present, !M(i,j) = 1 */   \
                        /* Cx [pC] = scalar */                      \
                        GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ; \
                        Cb [pC] = 1 ;                               \
                        task_cnvals++ ;                             \
                        break ;                                     \
                    case 1: /* C(i,j) present, !M(i,j) = 1 */       \
                        /* Cx [pC] += scalar */                     \
                        GB_ACCUMULATE_scalar (Cx, pC, ywork, C_iso) ;      \
                        break ;                                     \
                    case 2: /* C(i,j) not present, !M(i,j) = 0 */   \
                        /* clear the mask from C */                 \
                        Cb [pC] = 0 ;                               \
                        break ;                                     \
                    case 3: /* C(i,j) present, !M(i,j) = 0 */       \
                        /* C(i,j) remains; clear the mask from C */ \
                        Cb [pC] = 1 ;                               \
                        break ;                                     \
                    default: ;                                      \
                }                                                   \
            }
            #include "template/GB_bitmap_assign_C_whole_template.c"
        }

    }
    else
    {

        //----------------------------------------------------------------------
        // matrix assignment: C<!M, replace or !replace> += A
        //----------------------------------------------------------------------

        if (GB_IS_BITMAP (A) || GB_IS_FULL (A))
        {

            //------------------------------------------------------------------
            // C<!M, replace or !replace> += A where A is bitmap or full
            //------------------------------------------------------------------

            if (C_replace)
            { 

                //--------------------------------------------------------------
                // C<!M, replace> += A where A is bitmap or full
                //--------------------------------------------------------------

                #undef  GB_CIJ_WORK
                #define GB_CIJ_WORK(pC)                                       \
                {                                                             \
                    switch (Cb [pC])                                          \
                    {                                                         \
                        case 0: /* C(i,j) not present, !M(i,j) = 1 */         \
                            if (GBB_A (Ab, pC))                               \
                            {                                                 \
                                /* Cx [pC] = Ax [pC] */                       \
                                GB_COPY_aij_to_C (Cx,pC,Ax,pC,A_iso,cwork,    \
                                    C_iso) ;                                  \
                                Cb [pC] = 1 ;                                 \
                                task_cnvals++ ;                               \
                            }                                                 \
                            break ;                                           \
                        case 1: /* C(i,j) present, !M(i,j) = 1 */             \
                            if (GBB_A (Ab, pC))                               \
                            {                                                 \
                                /* Cx [pC] += Ax [pC] */                      \
                                GB_ACCUMULATE_aij (Cx,pC,Ax,pC,A_iso,ywork,   \
                                    C_iso) ;                                  \
                            }                                                 \
                            break ;                                           \
                        case 2: /* C(i,j) not present, !M(i,j) = 0 */         \
                            /* clear the mask from C */                       \
                            Cb [pC] = 0 ;                                     \
                            break ;                                           \
                        case 3: /* C(i,j) present, !M(i,j) = 0 */             \
                            /* delete this entry */                           \
                            Cb [pC] = 0 ;                                     \
                            task_cnvals-- ;                                   \
                            break ;                                           \
                        default: ;                                            \
                    }                                                         \
                }
                #include "template/GB_bitmap_assign_C_whole_template.c"

            }
            else
            { 

                //--------------------------------------------------------------
                // C<!M> += A where A is bitmap or full
                //--------------------------------------------------------------

                #undef  GB_CIJ_WORK
                #define GB_CIJ_WORK(pC)                                       \
                {                                                             \
                    switch (Cb [pC])                                          \
                    {                                                         \
                        case 0: /* C(i,j) not present, !M(i,j) = 1 */         \
                            if (GBB_A (Ab, pC))                               \
                            {                                                 \
                                /* Cx [pC] = Ax [pC] */                       \
                                GB_COPY_aij_to_C (Cx,pC,Ax,pC,A_iso,cwork,    \
                                    C_iso) ;                                  \
                                Cb [pC] = 1 ;                                 \
                                task_cnvals++ ;                               \
                            }                                                 \
                            break ;                                           \
                        case 1: /* C(i,j) present, !M(i,j) = 1 */             \
                            if (GBB_A (Ab, pC))                               \
                            {                                                 \
                                /* Cx [pC] += Ax [pC] */                      \
                                GB_ACCUMULATE_aij (Cx,pC,Ax,pC,A_iso,ywork,   \
                                    C_iso) ;                                  \
                            }                                                 \
                            Cb [pC] = 1 ;                                     \
                            break ;                                           \
                        case 2: /* C(i,j) not present, !M(i,j) = 0 */         \
                            /* clear the mask from C */                       \
                            Cb [pC] = 0 ;                                     \
                            break ;                                           \
                        case 3: /* C(i,j) present, !M(i,j) = 0 */             \
                            /* keep the entry */                              \
                            Cb [pC] = 1 ;                                     \
                            break ;                                           \
                        default: ;                                            \
                    }                                                         \
                }
                #include "template/GB_bitmap_assign_C_whole_template.c"
            }

        }
        else
        {

            //------------------------------------------------------------------
            // C<!M, replace or !replace> += A where A is sparse or hyper
            //------------------------------------------------------------------

            // assign or accumulate entries from A into C
            #undef  GB_AIJ_WORK
            #define GB_AIJ_WORK(pC,pA)                                  \
            {                                                           \
                int8_t cb = Cb [pC] ;                                   \
                if (cb == 0)                                            \
                {                                                       \
                    /* Cx [pC] = Ax [pA] */                             \
                    GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork, C_iso) ;   \
                    Cb [pC] = 1 ;                                       \
                    task_cnvals++ ;                                     \
                }                                                       \
                else if (cb == 1)                                       \
                {                                                       \
                    /* Cx [pC] += Ax [pA] */                            \
                    GB_ACCUMULATE_aij (Cx, pC, Ax, pA, A_iso, ywork, C_iso) ;  \
                }                                                       \
            }
            #include "template/GB_bitmap_assign_A_whole_template.c"

            if (C_replace)
            { 
                // clear the mask and delete entries not assigned
                #undef  GB_MASK_WORK
                #define GB_MASK_WORK(pC)                \
                {                                       \
                    int8_t cb = Cb [pC] ;               \
                    Cb [pC] = 0 ;                       \
                    task_cnvals -= (cb == 3) ;          \
                }
                #include "template/GB_bitmap_assign_M_all_template.c"
            }
            else
            { 
                // clear the mask
                // Cb [pC] -= 2 for each entry M(i,j) in the mask
                GB_bitmap_M_scatter_whole (C, M, GB_MASK_STRUCT,
                    GB_BITMAP_M_SCATTER_MINUS_2,
                    M_ek_slicing, M_ntasks, M_nthreads) ;
            }
        }
    }

    //--------------------------------------------------------------------------
    // free workspace and return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "final C for bitmap assign, !M, accum, whole", GB0) ;
    return (GrB_SUCCESS) ;
}

