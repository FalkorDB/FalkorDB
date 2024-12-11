//------------------------------------------------------------------------------
// GB_bitmap_assign_5_whole_template: C bitmap, no M, with accum
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2024, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------
// C<> += A            assign
// C<> += A            subassign

// C<repl> += A        assign
// C<repl> += A        subassign

// C<!> += A           assign: no work to do
// C<!> += A           subassign: no work to do

// C<!,repl> += A      assign: just clear C of all entries, not done here
// C<!,repl> += A      subassign: just clear C of all entries, not done here
//------------------------------------------------------------------------------

// C:           bitmap
// M:           none
// Mask_comp:   true or false
// Mask_struct: true or false
// C_replace:   true or false
// accum:       present
// A:           matrix (hyper, sparse, bitmap, or full), or scalar
// kind:        assign or subassign (same action)

// If Mask_comp is true, then an empty mask is complemented.  This case has
// already been handled by GB_assign_prep, which calls GB_clear, and thus
// Mask_comp is always false in this method.

// If C were full: entries can be deleted only if C_replace is true.

#undef  GB_FREE_ALL
#define GB_FREE_ALL GB_FREE_ALL_FOR_BITMAP

{

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_A_SCALAR_FOR_BITMAP
    GB_GET_ACCUM_FOR_BITMAP

    //--------------------------------------------------------------------------
    // do the assignment
    //--------------------------------------------------------------------------

    if (!GB_MASK_COMP)
    {

        //----------------------------------------------------------------------
        // C += A or += scalar
        //----------------------------------------------------------------------

        if (GB_SCALAR_ASSIGN)
        { 

            //------------------------------------------------------------------
            // scalar assignment: C += scalar
            //------------------------------------------------------------------

            #undef  GB_CIJ_WORK
            #define GB_CIJ_WORK(pC)                         \
            {                                               \
                int8_t cb = Cb [pC] ;                       \
                if (cb == 0)                                \
                {                                           \
                    /* Cx [pC] = scalar */                  \
                    GB_COPY_cwork_to_C (Cx, pC, cwork, C_iso) ; \
                }                                           \
                else                                        \
                {                                           \
                    /* Cx [pC] += scalar */                 \
                    GB_ACCUMULATE_scalar (Cx, pC, ywork, C_iso) ;  \
                }                                           \
            }
            if (!C_iso)
            {
                #include "template/GB_bitmap_assign_C_whole_template.c"
            }

            // free the bitmap or set it to all ones
            GB_bitmap_assign_to_full (C, nthreads_max) ;

        }
        else
        {

            //------------------------------------------------------------------
            // matrix assignment: C += A
            //------------------------------------------------------------------

            if (GB_IS_FULL (A))
            { 

                //--------------------------------------------------------------
                // C += A where C is bitmap and A is full
                //--------------------------------------------------------------

                #undef  GB_CIJ_WORK
                #define GB_CIJ_WORK(pC)                                     \
                {                                                           \
                    int8_t cb = Cb [pC] ;                                   \
                    if (cb == 0)                                            \
                    {                                                       \
                        /* Cx [pC] = Ax [pC] */                             \
                        GB_COPY_aij_to_C (Cx,pC,Ax,pC,A_iso,cwork,C_iso) ;  \
                    }                                                       \
                    else                                                    \
                    {                                                       \
                        /* Cx [pC] += Ax [pC] */                            \
                        GB_ACCUMULATE_aij (Cx,pC,Ax,pC,A_iso,ywork,C_iso) ; \
                    }                                                       \
                }
                if (!C_iso)
                {
                    #include "template/GB_bitmap_assign_C_whole_template.c"
                }

                // free the bitmap or set it to all ones
                GB_bitmap_assign_to_full (C, nthreads_max) ;

            }
            else if (GB_IS_BITMAP (A))
            { 

                //--------------------------------------------------------------
                // C += A where C and A are bitmap
                //--------------------------------------------------------------

                #undef  GB_CIJ_WORK
                #define GB_CIJ_WORK(pC)                                        \
                {                                                              \
                    if (Ab [pC])                                               \
                    {                                                          \
                        int8_t cb = Cb [pC] ;                                  \
                        if (cb == 0)                                           \
                        {                                                      \
                            /* Cx [pC] = Ax [pC] */                            \
                            GB_COPY_aij_to_C (Cx,pC,Ax,pC,A_iso,cwork,C_iso) ; \
                            Cb [pC] = 1 ;                                      \
                            task_cnvals++ ;                                    \
                        }                                                      \
                        else                                                   \
                        {                                                      \
                            /* Cx [pC] += Ax [pC] */                           \
                            GB_ACCUMULATE_aij (Cx,pC,Ax,pC,A_iso,ywork,C_iso) ;\
                        }                                                      \
                    }                                                          \
                }
                #include "template/GB_bitmap_assign_C_whole_template.c"
                C->nvals = cnvals ;

            }
            else
            { 

                //--------------------------------------------------------------
                // C += A where C is bitmap and A is sparse or hyper
                //--------------------------------------------------------------

                #undef  GB_AIJ_WORK
                #define GB_AIJ_WORK(pC,pA)                                  \
                {                                                           \
                    int8_t cb = Cb [pC] ;                                   \
                    if (cb == 0)                                            \
                    {                                                       \
                        /* Cx [pC] = Ax [pA] */                             \
                        GB_COPY_aij_to_C (Cx,pC,Ax,pA,A_iso,cwork,C_iso) ;  \
                        Cb [pC] = 1 ;                                       \
                        task_cnvals++ ;                                     \
                    }                                                       \
                    else                                                    \
                    {                                                       \
                        /* Cx [pC] += Ax [pA] */                            \
                        GB_ACCUMULATE_aij (Cx,pC,Ax,pA,A_iso,ywork,C_iso) ; \
                    }                                                       \
                }
                #include "template/GB_bitmap_assign_A_whole_template.c"
                C->nvals = cnvals ;
            }
        }
    }

#if 0
    else if (C_replace)
    {
        // The mask is not present yet complemented: C_replace phase only.  all
        // entries are deleted.  This is done by GB_clear in GB_assign_prep
        // and is thus not needed here.
        GB_memset (Cb, 0, cnzmax, nthreads_max) ;
        cnvals = 0 ;
    }
#endif

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    GB_FREE_ALL ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign, no M, accum, whole", GB0) ;
    return (GrB_SUCCESS) ;
}

