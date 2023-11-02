//------------------------------------------------------------------------------
// GB_bitmap_assign_fullM_noaccum_whole: assign to C bitmap, M is bitmap or full
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
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

// JIT: needed.

#include "GB_bitmap_assign_methods.h"
#include "GB_assign_shared_definitions.h"

#undef  GB_FREE_ALL
#define GB_FREE_ALL ;

GrB_Info GB_bitmap_assign_fullM_noaccum_whole
(
    // input/output:
    GrB_Matrix C,               // input/output matrix in bitmap format
    const bool C_replace,       // descriptor for C
    // inputs:
    const GrB_Matrix M,         // mask matrix, which is present here
    const bool Mask_comp,       // true for !M, false for M
    const bool Mask_struct,     // true if M is structural, false if valued
//  const GrB_BinaryOp accum,   // not present
    const GrB_Matrix A,         // input matrix, not transposed
    const void *scalar,         // input scalar
    const GrB_Type scalar_type, // type of input scalar
    GB_Werk Werk
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GBURBLE_BITMAP_ASSIGN ("bit2:whole", M, Mask_comp, NULL,
        GB_ALL, GB_ALL, GB_ASSIGN) ;
    ASSERT (GB_IS_BITMAP (M) || GB_IS_FULL (M)) ;
    ASSERT_MATRIX_OK (C, "C for bitmap assign, M full, noaccum, whole", GB0) ;
    ASSERT_MATRIX_OK (M, "M for bitmap assign, M full, noaccum, whole", GB0) ;
    ASSERT_MATRIX_OK_OR_NULL (A, "A for bitmap assign, M full, noaccum, whole",
        GB0) ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    GB_GET_C_BITMAP ;           // C must be bitmap
    GB_GET_M
    GB_GET_A_AND_SCALAR

    //--------------------------------------------------------------------------
    // to get the effective value of the mask entry mij
    //--------------------------------------------------------------------------

    #define GB_GET_MIJ(mij,pM)                                  \
        bool mij = (GBB (Mb, pM) && GB_MCAST (Mx, pM, msize)) ^ Mask_comp ;

    //--------------------------------------------------------------------------
    // assignment phase
    //--------------------------------------------------------------------------

    if (A == NULL)
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
                    GB_COPY_scalar_to_C (Cx, pC, cwork) ;   \
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
            #include "GB_bitmap_assign_C_whole_template.c"

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
                    GB_COPY_scalar_to_C (Cx, pC, cwork) ;   \
                    Cb [pC] = 1 ;                           \
                    task_cnvals += (cb == 0) ;              \
                }                                           \
            }
            #include "GB_bitmap_assign_C_whole_template.c"
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
                    if (mij && GBB (Ab, pC))                                \
                    {                                                       \
                        /* Cx [pC] = Ax [pC] */                             \
                        GB_COPY_aij_to_C (Cx, pC, Ax, pC, A_iso, cwork) ;   \
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
                #include "GB_bitmap_assign_C_whole_template.c"

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
                        if (GBB (Ab, pC))                                     \
                        {                                                     \
                            /* Cx [pC] = Ax [pC] */                           \
                            GB_COPY_aij_to_C (Cx, pC, Ax, pC, A_iso, cwork) ; \
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
                #include "GB_bitmap_assign_C_whole_template.c"
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
                        GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork) ;   \
                        Cb [pC] = 1 ;                                       \
                        task_cnvals++ ;                                     \
                    }                                                       \
                }
                #include "GB_bitmap_assign_A_whole_template.c"

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
                        GB_COPY_aij_to_C (Cx, pC, Ax, pA, A_iso, cwork) ;   \
                        Cb [pC] = 4 ; /* keep this entry */                 \
                        task_cnvals += (cb == 0) ;                          \
                    }                                                       \
                }
                #include "GB_bitmap_assign_A_whole_template.c"

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
                #include "GB_bitmap_assign_C_whole_template.c"
            }
        }
    }

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    C->nvals = cnvals ;
    ASSERT_MATRIX_OK (C, "final C bitmap assign, M full, noaccum, whole", GB0) ;
    return (GrB_SUCCESS) ;
}

