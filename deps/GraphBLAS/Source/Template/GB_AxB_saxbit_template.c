//------------------------------------------------------------------------------
// GB_AxB_saxbit_template.c: C<#M>=A*B when C is bitmap
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2023, All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

//------------------------------------------------------------------------------

// GB_AxB_saxpy_sparsity determines the sparsity structure for C<M or !M>=A*B
// or C=A*B, and this template is used when C is bitmap.

// FUTURE: C could be modified in-place if the accum operator is the same as
// the monoid.

// C is bitmap.
// M is anything: present or not, complemented or not, structural or valued,
//      and any format (sparse, hyper, bitmap, or full)
// A is sparse, hypersparse, bitmap, or full.
// B is bitmap or full.  It is not sparse or hyper.

// Either A or B may be iso-valued.  C can be iso-valued but only for the
// ANY_PAIR_ISO semiring.

{

    //--------------------------------------------------------------------------
    // get C, M, A, and B
    //--------------------------------------------------------------------------

    ASSERT (GB_IS_BITMAP (C)) ;                 // C is always bitmap
    int8_t *restrict Cb = C->b ;
    const int64_t cvlen = C->vlen ;
    ASSERT (C->vlen == A->vlen) ;
    ASSERT (C->vdim == B->vdim) ;
    ASSERT (A->vdim == B->vlen) ;
    int64_t cnvals = C->nvals ;

    const int8_t *restrict Bb = B->b ;
    const int64_t bvlen = B->vlen ;
    const int64_t bvdim = B->vdim ;
    GB_B_NHELD (bnz) ;      // const int64_t bnz = GB_nnz_held (B) ;

    #ifdef GB_JIT_KERNEL
    #define B_iso GB_B_ISO
    #define B_is_bitmap GB_B_IS_BITMAP
    #else
    const bool B_iso = B->iso ;
    const bool B_is_bitmap = GB_IS_BITMAP (B) ;
    #endif

    // B is bitmap or full
    ASSERT (!GB_IS_SPARSE (B)) ;
    ASSERT (!GB_IS_HYPERSPARSE (B)) ;

    const int64_t *restrict Ap = A->p ;
    const int64_t *restrict Ah = A->h ;
    const int8_t  *restrict Ab = A->b ;
    const int64_t *restrict Ai = A->i ;
    const int64_t anvec = A->nvec ;
    const int64_t avlen = A->vlen ;
    const int64_t avdim = A->vdim ;
    const bool A_jumbled = A->jumbled ;
    GB_A_NHELD (anz) ;      // const int64_t anz = GB_nnz_held (A) ;

    #ifdef GB_JIT_KERNEL
    #define A_iso GB_A_ISO
    #define A_is_hyper  GB_A_IS_HYPER
    #define A_is_sparse GB_A_IS_SPARSE
    #define A_is_bitmap GB_A_IS_BITMAP
    #define A_is_sparse_or_hyper (GB_A_IS_SPARSE || GB_A_IS_HYPER)
    #else
    const bool A_iso = A->iso ;
    const bool A_is_hyper = GB_IS_HYPERSPARSE (A) ;
    const bool A_is_sparse = GB_IS_SPARSE (A) ;
    const bool A_is_bitmap = GB_IS_BITMAP (A) ;
    const bool A_is_sparse_or_hyper = A_is_sparse || A_is_hyper ;
    #endif

    const int64_t *restrict Mp = NULL ;
    const int64_t *restrict Mh = NULL ;
    const int8_t  *restrict Mb = NULL ;
    const int64_t *restrict Mi = NULL ;
    const GB_M_TYPE *restrict Mx = NULL ;
    size_t msize = 0 ;
    int64_t mnvec = 0 ;
    int64_t mvlen = 0 ;

    #ifdef GB_JIT_KERNEL
    #define Mask_comp   GB_MASK_COMP
    #define Mask_struct GB_MASK_STRUCT
    #define M_is_present (!GB_NO_MASK)
    #if GB_NO_MASK
        #define M_is_hyper  0
        #define M_is_sparse 0
        #define M_is_bitmap 0
        #define M_is_full   0
        #define M_is_sparse_or_hyper        0
        #define GB_MASK_IS_SPARSE_OR_HYPER  0
        #define GB_MASK_IS_BITMAP_OR_FULL   0
    #else
        #define M_is_hyper  GB_M_IS_HYPER
        #define M_is_sparse GB_M_IS_SPARSE
        #define M_is_bitmap GB_M_IS_BITMAP
        #define M_is_full   GB_M_IS_FULL
        #define M_is_sparse_or_hyper        (GB_M_IS_SPARSE || GB_M_IS_HYPER)
        #define GB_MASK_IS_SPARSE_OR_HYPER  (GB_M_IS_SPARSE || GB_M_IS_HYPER)
        #define GB_MASK_IS_BITMAP_OR_FULL   (GB_M_IS_BITMAP || GB_M_IS_FULL)
    #endif
    #else
    const bool M_is_hyper  = GB_IS_HYPERSPARSE (M) ;
    const bool M_is_sparse = GB_IS_SPARSE (M) ;
    const bool M_is_bitmap = GB_IS_BITMAP (M) ;
    const bool M_is_full   = GB_IS_FULL (M) ;
    const bool M_is_sparse_or_hyper = M_is_sparse || M_is_hyper ;
    const bool M_is_present = (M != NULL) ;
    #endif

    if (M_is_present)
    {
        ASSERT (C->vlen == M->vlen) ;
        ASSERT (C->vdim == M->vdim) ;
        Mp = M->p ;
        Mh = M->h ;
        Mb = M->b ;
        Mi = M->i ;
        Mx = (GB_M_TYPE *) (Mask_struct ? NULL : (M->x)) ;
        msize = M->type->size ;
        mnvec = M->nvec ;
        mvlen = M->vlen ;

        // if M is sparse or hypersparse, scatter it into the C bitmap
        if (M_is_sparse_or_hyper)
        { 
            // Cb [pC] += 2 for each entry M(i,j) in the mask
            GB_bitmap_M_scatter (C,
                NULL, 0, GB_ALL, NULL, NULL, 0, GB_ALL, NULL,
                M, Mask_struct, GB_ASSIGN, GB_BITMAP_M_SCATTER_PLUS_2,
                M_ek_slicing, M_ntasks, M_nthreads) ;
            // the bitmap of C now contains:
            //  Cb (i,j) = 0:   cij not present, mij zero
            //  Cb (i,j) = 1:   cij present, mij zero
            //  Cb (i,j) = 2:   cij not present, mij 1
            //  Cb (i,j) = 3:   cij present, mij 1
        }
    }

    #if !GB_A_IS_PATTERN
    const GB_A_TYPE *restrict Ax = (GB_A_TYPE *) A->x ;
    #endif
    #if !GB_B_IS_PATTERN
    const GB_B_TYPE *restrict Bx = (GB_B_TYPE *) B->x ;
    #endif
    #if !GB_IS_ANY_PAIR_SEMIRING
          GB_C_TYPE *restrict Cx = (GB_C_TYPE *) C->x ;
    #endif

    //--------------------------------------------------------------------------
    // select the method
    //--------------------------------------------------------------------------

    #ifdef GB_JIT_KERNEL
    {

        //----------------------------------------------------------------------
        // JIT kernel
        //----------------------------------------------------------------------

        #if (!GB_NO_MASK) && GB_MASK_IS_SPARSE_OR_HYPER && (!GB_MASK_COMP)
        #define keep 3
        #else
        #define keep 1
        #endif

        #if ( GB_A_IS_SPARSE || GB_A_IS_HYPER )
        {
            #include "GB_AxB_saxbit_A_sparse_B_bitmap_template.c"
        }
        #else
        {
            #include "GB_AxB_saxbit_A_bitmap_B_bitmap_template.c"
        }
        #endif

    }
    #else
    {

        //----------------------------------------------------------------------
        // factory or generic kernel 
        //----------------------------------------------------------------------

        if (A_is_sparse_or_hyper)
        {

            //-----------------------------------------------------
            // C                =               A     *     B
            //-----------------------------------------------------

            // bitmap           .               hyper       bitmap
            // bitmap           .               sparse      bitmap
            // bitmap           .               hyper       full 
            // bitmap           .               sparse      full

            //-----------------------------------------------------
            // C               <M>=             A     *     B
            //-----------------------------------------------------

            // bitmap           any             hyper       bitmap
            // bitmap           any             sparse      bitmap
            // bitmap           bitmap/full     hyper       full
            // bitmap           bitmap/full     sparse      full

            //-----------------------------------------------------
            // C               <!M>=            A     *     B
            //-----------------------------------------------------

            // bitmap           any             hyper       bitmap
            // bitmap           any             sparse      bitmap
            // bitmap           any             hyper       full 
            // bitmap           any             sparse      full

            if (M == NULL)
            {

                //--------------------------------------------------------------
                // C = A*B, no mask, A sparse/hyper, B bitmap/full
                //--------------------------------------------------------------

                #define GB_NO_MASK 1
                #define GB_MASK_IS_SPARSE_OR_HYPER 0
                #define GB_MASK_IS_BITMAP_OR_FULL  0
                #undef  keep
                #define keep 1
                if (B_is_bitmap)
                { 
                    // A is sparse/hyper, B is bitmap, no mask
                    #undef  GB_B_IS_BITMAP
                    #define GB_B_IS_BITMAP 1
                    #include "GB_AxB_saxbit_A_sparse_B_bitmap_template.c"
                }
                else
                { 
                    // A is sparse/hyper, B is full, no mask
                    #undef  GB_B_IS_BITMAP
                    #define GB_B_IS_BITMAP 0
                    #include "GB_AxB_saxbit_A_sparse_B_bitmap_template.c"
                }
                #undef GB_MASK_IS_SPARSE_OR_HYPER
                #undef GB_MASK_IS_BITMAP_OR_FULL
                #undef GB_NO_MASK

            }
            else if (M_is_sparse_or_hyper)
            {

                //--------------------------------------------------------------
                // C<M> or <!M> = A*B, M and A are sparse/hyper, B bitmap/full
                //--------------------------------------------------------------

                #define GB_NO_MASK 0
                #define GB_MASK_IS_SPARSE_OR_HYPER 1
                #define GB_MASK_IS_BITMAP_OR_FULL  0
                #undef  keep
                const int8_t keep = (Mask_comp) ? 1 : 3 ;
                if (B_is_bitmap)
                { 
                    // A is sparse/hyper, B is bitmap, M is sparse/hyper
                    #undef  GB_B_IS_BITMAP
                    #define GB_B_IS_BITMAP 1
                    #include "GB_AxB_saxbit_A_sparse_B_bitmap_template.c"
                }
                else
                { 
                    // A is sparse/hyper, B is full, M is sparse/hyper
                    #undef  GB_B_IS_BITMAP
                    #define GB_B_IS_BITMAP 0
                    #include "GB_AxB_saxbit_A_sparse_B_bitmap_template.c"
                }
                #undef GB_MASK_IS_SPARSE_OR_HYPER
                #undef GB_MASK_IS_BITMAP_OR_FULL

            }
            else
            {

                //--------------------------------------------------------------
                // C<M> or <!M> = A*B, M bitmap, A sparse, B bitmap
                //--------------------------------------------------------------

                #define GB_MASK_IS_SPARSE_OR_HYPER 0
                #define GB_MASK_IS_BITMAP_OR_FULL  1
                #undef  keep
                #define keep 1
                if (B_is_bitmap)
                { 
                    // A is sparse/hyper, B is bitmap, M is bitmap/full
                    #undef  GB_B_IS_BITMAP
                    #define GB_B_IS_BITMAP 1
                    #include "GB_AxB_saxbit_A_sparse_B_bitmap_template.c"
                }
                else
                { 
                    // A is sparse/hyper, B is full, M is bitmap/full
                    #undef  GB_B_IS_BITMAP
                    #define GB_B_IS_BITMAP 0
                    #include "GB_AxB_saxbit_A_sparse_B_bitmap_template.c"
                }
                #undef GB_MASK_IS_SPARSE_OR_HYPER
                #undef GB_MASK_IS_BITMAP_OR_FULL
                #undef GB_NO_MASK
            }

            #undef GB_B_IS_BITMAP

        }
        else
        {

            //-----------------------------------------------------
            // C                =               A     *     B
            //-----------------------------------------------------

            // bitmap           .               bitmap      bitmap
            // bitmap           .               full        bitmap
            // bitmap           .               bitmap      full
            // full             .               full        full

            //-----------------------------------------------------
            // C               <M>=             A     *     B
            //-----------------------------------------------------

            // bitmap           any             bitmap      bitmap
            // bitmap           any             full        bitmap
            // bitmap           bitmap/full     bitmap      full
            // bitmap           bitmap/full     full        full

            //-----------------------------------------------------
            // C               <!M>=            A     *     B
            //-----------------------------------------------------

            // bitmap           any             bitmap      bitmap
            // bitmap           any             full        bitmap
            // bitmap           any             bitmap      full
            // bitmap           any             full        full

            if (M == NULL)
            { 

                //--------------------------------------------------------------
                // C = A*B, no mask, A and B bitmap/full
                //--------------------------------------------------------------

                #define GB_MASK_IS_SPARSE_OR_HYPER 0
                #define GB_MASK_IS_BITMAP_OR_FULL  0
                #undef  keep
                #define keep 1
                #include "GB_AxB_saxbit_A_bitmap_B_bitmap_template.c"
                #undef GB_MASK_IS_SPARSE_OR_HYPER
                #undef GB_MASK_IS_BITMAP_OR_FULL

            }
            else if (M_is_sparse_or_hyper)
            { 

                //--------------------------------------------------------------
                // C<M> or <!M> = A*B, M sparse/hyper, A and B bitmap/full
                //--------------------------------------------------------------

                #define GB_MASK_IS_SPARSE_OR_HYPER 1
                #define GB_MASK_IS_BITMAP_OR_FULL  0
                #undef  keep
                const int8_t keep = (Mask_comp) ? 1 : 3 ;
                #include "GB_AxB_saxbit_A_bitmap_B_bitmap_template.c"
                #undef GB_MASK_IS_SPARSE_OR_HYPER
                #undef GB_MASK_IS_BITMAP_OR_FULL

            }
            else
            { 

                //--------------------------------------------------------------
                // C<M> or <!M> = A*B, all matrices bitmap/full
                //--------------------------------------------------------------

                #define GB_MASK_IS_SPARSE_OR_HYPER 0
                #define GB_MASK_IS_BITMAP_OR_FULL  1
                #undef  keep
                #define keep 1
                #include "GB_AxB_saxbit_A_bitmap_B_bitmap_template.c"
                #undef GB_MASK_IS_SPARSE_OR_HYPER
                #undef GB_MASK_IS_BITMAP_OR_FULL
            }
        }
    }
    #endif

    C->nvals = cnvals ;

    //--------------------------------------------------------------------------
    // if M is sparse, clear it from the C bitmap
    //--------------------------------------------------------------------------

    if (M_is_sparse_or_hyper)
    { 
        // Cb [pC] -= 2 for each entry M(i,j) in the mask
        GB_bitmap_M_scatter (C,
            NULL, 0, GB_ALL, NULL, NULL, 0, GB_ALL, NULL,
            M, Mask_struct, GB_ASSIGN, GB_BITMAP_M_SCATTER_MINUS_2,
            M_ek_slicing, M_ntasks, M_nthreads) ;
    }
}

